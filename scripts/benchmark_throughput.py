#!/usr/bin/env python3
"""Command-line wrapper around glm_ladder.benchmark (the notebook calls it directly)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_ladder.benchmark import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="s-50m,m-100m,l-200m")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_benchmark(
        [name.strip() for name in args.presets.split(",") if name.strip()],
        sequence_length=args.sequence_length, micro_batch=args.micro_batch, steps=args.steps,
        gradient_checkpointing=args.gradient_checkpointing, use_compile=args.compile,
    )
    text = json.dumps(receipt, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
