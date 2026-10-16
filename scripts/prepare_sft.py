"""Tokenize and pack the SFT data, then upload it to the private dataset repo under sft/.

    python scripts/prepare_sft.py                 # build, then ask for a token and upload
    python scripts/prepare_sft.py --no-upload     # build only (inspect data/sft/ first)

Sources (streamed, nothing is downloaded in full):
    english  HuggingFaceTB/smol-smoltalk   train split -> train, first 2,000 of test -> val
    turkish  turkish-nlp-suite/InstrucTurca first 1,000 usable rows -> val, the rest -> train
The Turkish train set stops at 1/9 of the English train windows, so Turkish is ~10% of
SFT tokens, matching the pretraining mix.

Output, per source and split: tokens.npy (uint16) and mask.npy (uint8), both
[windows, seq_len + 1], plus sft/manifest.json with counts and SHA-256 of every file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from getpass import getpass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glm_ladder.chat import END_TOKEN, encode_conversation, from_instructurca, special_id  # noqa: E402
from glm_ladder.sft_data import pack_conversations  # noqa: E402
from glm_ladder.tokenizer import load_tokenizer  # noqa: E402

DATASETS = {
    "english": ("HuggingFaceTB/smol-smoltalk", None),
    "turkish": ("turkish-nlp-suite/InstrucTurca", None),
}
ENGLISH_VAL_CONVERSATIONS = 2_000
TURKISH_VAL_CONVERSATIONS = 1_000
TURKISH_FRACTION = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stream(name: str, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    path, config = DATASETS[name]
    yield from load_dataset(path, name=config, split=split, streaming=True)


def encode_all(tokenizer, conversations: Iterable[list[dict]], window: int, stats: Counter
               ) -> Iterator[tuple[list[int], list[bool]]]:
    """Encode conversations, counting (and skipping) the ones the template rejects or that are too long.

    A conversation longer than a window is cut after its last complete assistant turn that
    still fits (turns are never split); if even the first answer does not fit, it is skipped.
    """
    end_id = special_id(tokenizer, END_TOKEN)
    for messages in conversations:
        stats["seen"] += 1
        try:
            ids, learn = encode_conversation(tokenizer, messages)
        except ValueError:
            stats["skipped_invalid"] += 1
            continue
        if len(ids) + 1 > window:
            # Ends of assistant turns are learned <|end|> tokens; prefix ids[:i + 1] plus EOS must fit.
            cuts = [i + 1 for i, (token, learned) in enumerate(zip(ids, learn))
                    if token == end_id and learned and i + 2 <= window]
            if not cuts:
                stats["skipped_too_long"] += 1
                continue
            ids, learn = ids[:cuts[-1]], learn[:cuts[-1]]
            stats["truncated_at_turn"] += 1
        if not any(learn):
            stats["skipped_nothing_to_learn"] += 1
            continue
        stats["kept"] += 1
        stats["tokens"] += len(ids) + 1
        stats["learned_tokens"] += sum(learn)
        yield ids, learn


def build(name: str, split: str, conversations: Iterable[list[dict]], tokenizer, window: int,
          out_root: Path, max_windows: int | None = None) -> dict:
    stats: Counter = Counter()
    tokens, masks = [], []
    started = time.perf_counter()
    for tokens_row, mask_row in pack_conversations(encode_all(tokenizer, conversations, window, stats), window):
        tokens.append(tokens_row)
        masks.append(mask_row.astype(np.uint8))
        if len(tokens) % 20_000 == 0:
            print(f"  [{name}/{split}] {len(tokens):,} windows, {stats['kept']:,} conversations, "
                  f"{time.perf_counter() - started:.0f} s")
        if max_windows is not None and len(tokens) >= max_windows:
            break   # counts then include the one conversation that opened the next, unsaved window
    directory = out_root / name / split
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "tokens.npy", np.stack(tokens))
    np.save(directory / "mask.npy", np.stack(masks))
    info = {
        "dataset": DATASETS[name][0], "split": split, "windows": len(tokens),
        "conversations": stats["kept"], "truncated_at_turn": stats["truncated_at_turn"],
        "skipped": {key: stats[key] for key in ("skipped_invalid", "skipped_too_long", "skipped_nothing_to_learn")},
        "tokens": stats["tokens"], "learned_tokens": stats["learned_tokens"],
        "fill": round(stats["tokens"] / (len(tokens) * window), 4),
        "sha256": {file: sha256_file(directory / file) for file in ("tokens.npy", "mask.npy")},
    }
    print(f"  [{name}/{split}] done: {json.dumps({k: v for k, v in info.items() if k != 'sha256'})}")
    return info


def english(split: str, limit: int | None) -> Iterator[list[dict]]:
    for index, row in enumerate(stream("english", split)):
        if limit is not None and index >= limit:
            return
        yield [{"role": m["role"], "content": m["content"]} for m in row["messages"]]


def turkish_rows() -> Iterator[list[dict]]:
    for row in stream("turkish", "train"):
        if (row.get("Input") or "").strip() and (row.get("Output") or "").strip():
            yield from_instructurca(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="Tbt-AI/glm-ladder-data")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--out", default=str(ROOT / "data" / "sft"))
    parser.add_argument("--english-limit", type=int, help="use only the first N English train rows (for a quick test)")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    tokenizer_path = ROOT / "tokenizer" / "glm-ladder-16k.json"
    receipt = json.loads((ROOT / "artifacts" / "receipts" / "tokenizer-16k.json").read_text())
    if sha256_file(tokenizer_path) != receipt["sha256"]:
        raise SystemExit("tokenizer file does not match its receipt")
    tokenizer = load_tokenizer(tokenizer_path)
    window, out_root = args.seq_len + 1, Path(args.out)

    manifest = {"schema_version": "1.0", "kind": "sft_data", "seq_len": args.seq_len,
                "tokenizer_sha256": receipt["sha256"], "sources": {"english": {}, "turkish": {}}}
    print("english / val")
    manifest["sources"]["english"]["val"] = build(
        "english", "val", english("test", ENGLISH_VAL_CONVERSATIONS), tokenizer, window, out_root)
    print("english / train")
    manifest["sources"]["english"]["train"] = build(
        "english", "train", english("train", args.english_limit), tokenizer, window, out_root)

    # Turkish: one stream, first rows for validation, then train up to the target size.
    rows = turkish_rows()
    target = round(manifest["sources"]["english"]["train"]["windows"] * TURKISH_FRACTION / (1 - TURKISH_FRACTION))
    val_rows = [next(rows) for _ in range(TURKISH_VAL_CONVERSATIONS)]
    print("turkish / val")
    manifest["sources"]["turkish"]["val"] = build("turkish", "val", val_rows, tokenizer, window, out_root)
    print(f"turkish / train (target {target:,} windows)")
    manifest["sources"]["turkish"]["train"] = build(
        "turkish", "train", rows, tokenizer, window, out_root, max_windows=target)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

    # Show the first English training window, so the template can be checked by eye.
    first = np.load(out_root / "english" / "train" / "tokens.npy", mmap_mode="r")[0]
    print("\nfirst window (start):", tokenizer.decode(first[:200].tolist(), skip_special_tokens=False))

    if args.no_upload:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=getpass("Hugging Face write token: "))
    api.upload_folder(folder_path=str(out_root), path_in_repo="sft", repo_id=args.repo, repo_type="dataset",
                      commit_message="SFT data: smol-smoltalk + InstrucTurca, packed")
    print(f"uploaded to {args.repo}/sft")


if __name__ == "__main__":
    main()