"""Build the Phase 4 review-rating data in Turkish and English, then upload it under finetune/.

    python scripts/prepare_finetune.py                # build, then ask for a token and upload
    python scripts/prepare_finetune.py --no-upload    # build only (inspect data/finetune/ first)

Sources:
    tr  turkish-nlp-suite/SentiTurca, subset e-commerce (Trendyol / Hepsiburada reviews, 1-5 stars)
    en  Yelp/yelp_review_full (Yelp reviews, 1-5 stars)

The two languages are made the same size: the English train / val / test sets are seeded
samples with exactly as many reviews as the Turkish ones (English val comes from the Yelp
train split, disjoint from English train). A review whose prompt is longer than
--max-prompt-tokens is skipped in both languages, so the length rule is identical.

Output per language:
    train/tokens.npy, train/mask.npy   packed windows for training (only the answer digit and <|end|> learned)
    val/tokens.npy, val/mask.npy       the same for the validation loss during training
    val.jsonl, test.jsonl              {"text", "stars"} per review, for accuracy (scored without generation)
plus finetune/manifest.json with counts, star distributions and SHA-256 of every file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from getpass import getpass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glm_ladder.chat import encode_conversation, prompt_ids  # noqa: E402
from glm_ladder.checkpoint import sha256_file  # noqa: E402
from glm_ladder.classify import STARS, conversation  # noqa: E402
from glm_ladder.sft_data import pack_conversations  # noqa: E402
from glm_ladder.tokenizer import load_tokenizer  # noqa: E402

SEED = 1234
SENTITURCA = "turkish-nlp-suite/SentiTurca"
YELP = "Yelp/yelp_review_full"


# ----------------------------------------------------------------- reading the sources

def to_stars(labels: list, names: list[str] | None = None) -> list[int]:
    """Map raw labels to 1-5 stars, refusing anything ambiguous.

    Integers 0-4 (class indices) become 1-5; integers 1-5 stay. Strings, or indices with
    ClassLabel names, must contain exactly one digit 1-5 (e.g. "4_star").
    """
    if names is not None:
        labels = [names[label] for label in labels]
    if all(isinstance(label, (int, np.integer)) for label in labels):
        values = set(int(label) for label in labels)
        if values <= {0, 1, 2, 3, 4} and 0 in values:
            return [int(label) + 1 for label in labels]
        if values <= set(STARS):
            return [int(label) for label in labels]
        raise ValueError(f"cannot map integer labels {sorted(values)} to stars")
    stars = []
    for label in labels:
        digits = re.findall(r"[1-5]", str(label))
        if len(digits) != 1:
            raise ValueError(f"cannot read stars from label {label!r}")
        stars.append(int(digits[0]))
    return stars


def load_sentiturca() -> dict[str, list[tuple[str, int]]]:
    from datasets import load_dataset

    files = {split: f"hf://datasets/{SENTITURCA}/e-commerce/{split}*.jsonl" for split in ("train", "validation", "test")}
    data = load_dataset("json", data_files=files)
    out = {}
    for split, name in (("train", "train"), ("validation", "val"), ("test", "test")):
        rows = data[split]
        stars = to_stars(list(rows["label"]))
        print(f"  SentiTurca {split}: {len(rows):,} rows; raw labels {sorted(Counter(rows['label']).items())[:5]} "
              f"-> stars {sorted(Counter(stars).items())}")
        out[name] = list(zip(rows["text"], stars))
    return out


def clean_yelp(text: str) -> str:
    # The Yelp CSV escapes line breaks and quotes as literal backslash sequences.
    return text.replace("\\n", "\n").replace('\\"', '"').strip()


def load_yelp() -> dict[str, list[tuple[str, int]]]:
    from datasets import load_dataset

    data = load_dataset(YELP)
    out = {}
    for split in ("train", "test"):
        rows = data[split]
        names = rows.features["label"].names
        stars = to_stars(list(rows["label"]), names)
        print(f"  Yelp {split}: {len(rows):,} rows; label names {names} -> stars {sorted(Counter(stars).items())}")
        out[split] = [(clean_yelp(text), star) for text, star in zip(rows["text"], stars)]
    return out


# ----------------------------------------------------------------- encoding and packing

def encodable(tokenizer, text: str, stars: int, language: str, max_prompt_tokens: int, stats: Counter):
    """(ids, learn) of the full example, or None (counted in stats) if it must be skipped."""
    if not text.strip():
        stats["skipped_empty"] += 1
        return None
    messages = conversation(text, language, stars)
    try:
        prompt_length = len(prompt_ids(tokenizer, messages[:1]))
        ids, learn = encode_conversation(tokenizer, messages)
    except ValueError:                                  # a special token inside the review
        stats["skipped_invalid"] += 1
        return None
    if prompt_length > max_prompt_tokens:
        stats["skipped_too_long"] += 1
        return None
    stats["kept"] += 1
    stats["tokens"] += len(ids) + 1
    return ids, learn


def select(tokenizer, rows, language: str, max_prompt_tokens: int, *, limit: int | None = None,
           shuffle: bool = False, exclude: set[int] | None = None):
    """Keep encodable rows (optionally a seeded shuffle, at most `limit`). Returns (kept, used row ids, stats)."""
    order = np.random.default_rng([SEED, len(rows)]).permutation(len(rows)) if shuffle else range(len(rows))
    stats: Counter = Counter()
    kept, used = [], set()
    for index in order:
        index = int(index)
        if exclude and index in exclude:
            continue
        text, stars = rows[index]
        encoded = encodable(tokenizer, text, stars, language, max_prompt_tokens, stats)
        if encoded is None:
            continue
        kept.append((text, stars, *encoded))
        used.add(index)
        if limit is not None and len(kept) == limit:
            break
    if limit is not None and len(kept) < limit:
        raise SystemExit(f"{language}: only {len(kept):,} usable rows, {limit:,} needed")
    return kept, used, stats


def write_split(out: Path, language: str, split: str, kept, stats: Counter, window: int, *, packed: bool,
                examples: bool) -> dict:
    directory = out / language
    directory.mkdir(parents=True, exist_ok=True)
    info = {"examples": len(kept), "stars": dict(sorted(Counter(stars for _, stars, _, _ in kept).items())),
            "skipped": {key: stats[key] for key in ("skipped_empty", "skipped_invalid", "skipped_too_long")},
            "tokens": stats["tokens"], "sha256": {}}
    if packed:
        # Shuffle before packing so a window does not hold a run of reviews with the same label.
        order = np.random.default_rng([SEED, len(kept), 1]).permutation(len(kept))
        windows = list(pack_conversations(((kept[i][2], kept[i][3]) for i in order), window))
        (directory / split).mkdir(exist_ok=True)
        np.save(directory / split / "tokens.npy", np.stack([tokens for tokens, _ in windows]))
        np.save(directory / split / "mask.npy", np.stack([mask.astype(np.uint8) for _, mask in windows]))
        info["windows"] = len(windows)
        info["fill"] = round(stats["tokens"] / (len(windows) * window), 4)
        for file in ("tokens.npy", "mask.npy"):
            info["sha256"][f"{split}/{file}"] = sha256_file(directory / split / file)
    if examples:
        path = directory / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for text, stars, _, _ in kept:
                handle.write(json.dumps({"text": text, "stars": stars}, ensure_ascii=False) + "\n")
        info["sha256"][f"{split}.jsonl"] = sha256_file(path)
    print(f"  [{language}/{split}] {json.dumps({k: v for k, v in info.items() if k != 'sha256'})}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="Tbt-AI/glm-ladder-data")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--out", default=str(ROOT / "data" / "finetune"))
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    tokenizer_path = ROOT / "tokenizer" / "glm-ladder-16k.json"
    receipt = json.loads((ROOT / "artifacts" / "receipts" / "tokenizer-16k.json").read_text())
    if sha256_file(tokenizer_path) != receipt["sha256"]:
        raise SystemExit("tokenizer file does not match its receipt")
    tokenizer = load_tokenizer(tokenizer_path)
    window, out, cap = args.seq_len + 1, Path(args.out), args.max_prompt_tokens
    manifest = {"schema_version": "1.0", "kind": "finetune_data", "task": "review stars 1-5",
                "seq_len": args.seq_len, "max_prompt_tokens": cap, "seed": SEED,
                "tokenizer_sha256": receipt["sha256"],
                "datasets": {"tr": f"{SENTITURCA} (e-commerce)", "en": YELP}, "languages": {"tr": {}, "en": {}}}

    print("Turkish (SentiTurca e-commerce)")
    turkish = load_sentiturca()
    sizes = {}
    for split in ("train", "val", "test"):
        kept, _, stats = select(tokenizer, turkish[split], "tr", cap)
        sizes[split] = len(kept)
        manifest["languages"]["tr"][split] = write_split(out, "tr", split, kept, stats, window,
                                                         packed=split != "test", examples=split != "train")

    print("English (Yelp), sampled to the Turkish sizes", sizes)
    yelp = load_yelp()
    train, used, stats = select(tokenizer, yelp["train"], "en", cap, limit=sizes["train"], shuffle=True)
    manifest["languages"]["en"]["train"] = write_split(out, "en", "train", train, stats, window, packed=True, examples=False)
    val, _, stats = select(tokenizer, yelp["train"], "en", cap, limit=sizes["val"], shuffle=True, exclude=used)
    manifest["languages"]["en"]["val"] = write_split(out, "en", "val", val, stats, window, packed=True, examples=True)
    test, _, stats = select(tokenizer, yelp["test"], "en", cap, limit=sizes["test"], shuffle=True)
    manifest["languages"]["en"]["test"] = write_split(out, "en", "test", test, stats, window, packed=False, examples=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({lang: {split: {k: v for k, v in info.items() if k != "sha256"} for split, info in splits.items()}
                      for lang, splits in manifest["languages"].items()}, indent=2))
    for language in ("tr", "en"):
        first = np.load(out / language / "train" / "tokens.npy", mmap_mode="r")[0]
        print(f"\nfirst {language} training window (start):",
              tokenizer.decode(first[:120].tolist(), skip_special_tokens=False))

    if args.no_upload:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=getpass("Hugging Face write token: "))
    api.upload_folder(folder_path=str(out), path_in_repo="finetune", repo_id=args.repo, repo_type="dataset",
                      commit_message="Fine-tuning data: review stars, SentiTurca e-commerce (tr) + Yelp (en)")
    print(f"uploaded to {args.repo}/finetune")


if __name__ == "__main__":
    main()
