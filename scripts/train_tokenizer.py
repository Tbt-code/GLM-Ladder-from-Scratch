#!/usr/bin/env python3
"""Train the shared byte-level BPE tokenizer on the 60/30/10 mix and write a receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_ladder.data_sources import MIX, SOURCES, iter_documents, mixed_sample, take_bytes
from glm_ladder.tokenizer import SPECIAL_TOKENS, VOCAB_SIZE, bytes_per_token, train_bpe_tokenizer

EVAL_BYTES_PER_SOURCE = 2_000_000  # held-out text per source for the compression table


def evaluate(tokenizer, eval_texts: dict[str, list[str]]) -> dict[str, float]:
    return {name: round(bytes_per_token(tokenizer, texts), 3) for name, texts in eval_texts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-mb", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("tokenizer/glm-ladder-16k.json"))
    parser.add_argument("--receipt", type=Path, default=Path("artifacts/receipts/tokenizer-16k.json"))
    parser.add_argument(
        "--compare-without-turkish", action="store_true",
        help="also train on the same English and code documents without Turkish, for comparison",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; delete it to retrain")
    total_bytes = args.sample_mb * 1_000_000
    started = time.perf_counter()

    print(f"Sampling {args.sample_mb} MB with mix {MIX} ...", flush=True)
    streams = {name: iter_documents(name) for name in MIX}
    sample = mixed_sample(total_bytes, MIX, streams)
    # Held-out evaluation text: the documents that follow the sample in each stream.
    eval_texts = {name: list(take_bytes(streams[name], EVAL_BYTES_PER_SOURCE)) for name in MIX}
    sample_seconds = time.perf_counter() - started

    print(f"Training on {len(sample):,} documents ...", flush=True)
    tokenizer = train_bpe_tokenizer(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))
    compression = evaluate(tokenizer, eval_texts)
    print("bytes per token:", compression, flush=True)

    comparison = None
    if args.compare_without_turkish:
        # Same English and code budgets as above, so the streams yield the same documents.
        no_turkish_mix = {"english": 0.6 / 0.9, "code": 0.3 / 0.9}
        print("Training comparison tokenizer without Turkish ...", flush=True)
        no_turkish_sample = mixed_sample(
            int(total_bytes * 0.9), no_turkish_mix, {n: iter_documents(n) for n in no_turkish_mix}
        )
        comparison = {
            "mix": no_turkish_mix,
            "bytes_per_token": evaluate(train_bpe_tokenizer(no_turkish_sample), eval_texts),
        }
        print("without Turkish:", comparison["bytes_per_token"], flush=True)

    import tokenizers

    receipt = {
        "schema_version": "1.0",
        "kind": "tokenizer",
        "path": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": VOCAB_SIZE,
        "special_tokens": SPECIAL_TOKENS,
        "sources": {name: SOURCES[name][:2] for name in MIX},
        "mix": MIX,
        "sample_bytes": sum(len(t.encode("utf-8")) for t in sample),
        "sample_documents": len(sample),
        "eval_bytes_per_source": EVAL_BYTES_PER_SOURCE,
        "bytes_per_token": compression,
        "comparison_without_turkish": comparison,
        "sample_seconds": round(sample_seconds, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "tokenizers_version": tokenizers.__version__,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("vocab_size", "bytes_per_token", "total_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())