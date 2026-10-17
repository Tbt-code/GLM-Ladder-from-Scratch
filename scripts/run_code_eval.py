"""Run the executable code evaluation on a run from the Hub, on CPU, and save a receipt.

    python scripts/run_code_eval.py --run s50m-001 --mode completion       # pretrained model
    python scripts/run_code_eval.py --run sft-s50m-001 --mode completion   # same prompts after SFT
    python scripts/run_code_eval.py --run sft-s50m-001 --mode chat         # asked as a chat request

For each of the 40 tasks: one greedy answer (temperature 0) and --n sampled answers.
Reports greedy pass@1 and sampled pass@1 / pass@k. Writes artifacts/evals/code-<run>-<mode>.json
with every generated function and whether it passed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from getpass import getpass
from pathlib import Path

import torch
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from glm_ladder.chat import END_TOKEN, prompt_ids, special_id  # noqa: E402
from glm_ladder.code_eval import (  # noqa: E402
    extract_from_chat, extract_from_completion, load_tasks, run_tests, sample_many, summarize,
)
from glm_ladder.tokenizer import EOS_ID, load_tokenizer  # noqa: E402
from sample import load_run  # noqa: E402  (scripts/sample.py)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=("completion", "chat"), required=True)
    parser.add_argument("--repo", default="Tbt-AI/glm-ladder-runs")
    parser.add_argument("--checkpoint", help="default: latest")
    parser.add_argument("--n", type=int, default=10, help="sampled answers per task")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="only the first N tasks (quick check)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    api = HfApi(token=getpass("Hugging Face token: "))
    model, state = load_run(api, args.repo, args.run, args.checkpoint)
    tokenizer = load_tokenizer(ROOT / "tokenizer" / "glm-ladder-16k.json")
    tasks = load_tasks(ROOT / "evals" / "code_tasks.jsonl")[:args.limit]
    end_id = special_id(tokenizer, END_TOKEN)

    if args.mode == "completion":
        def stop(ids):  # a new line that starts without indentation ends the function
            return EOS_ID in ids or re.search(r"\n\S", tokenizer.decode(ids)) is not None
    else:
        def stop(ids):
            return ids[-1] in (end_id, EOS_ID)

    def to_code(task, ids):
        text = tokenizer.decode([t for t in ids if t not in (end_id, EOS_ID)])
        if args.mode == "completion":
            return extract_from_completion(task.completion_prompt(), text)
        return extract_from_chat(text, task.name)

    results, started = [], time.perf_counter()
    for index, task in enumerate(tasks):
        if args.mode == "completion":
            ids = [EOS_ID, *tokenizer.encode(task.completion_prompt()).ids]
        else:
            ids = prompt_ids(tokenizer, [{"role": "user", "content": task.chat_prompt()}])
        common = dict(max_new_tokens=args.max_tokens, top_k=args.top_k, seed=args.seed + index, stop=stop)
        greedy = sample_many(model, ids, n=1, temperature=0.0, **common)[0]
        sampled = sample_many(model, ids, n=args.n, temperature=args.temperature, **common)
        entry = {"task": task.name, "greedy": None, "samples": []}
        for out, target in [(greedy, "greedy"), *[(s, "samples") for s in sampled]]:
            code = to_code(task, out)
            record = {"code": code, "passed": code is not None and run_tests(code, task.tests)}
            if target == "greedy":
                entry["greedy"] = record
            else:
                entry["samples"].append(record)
        results.append(entry)
        hits = sum(s["passed"] for s in entry["samples"])
        print(f"[{index + 1:2d}/{len(tasks)}] {task.name:22s} greedy {'PASS' if entry['greedy']['passed'] else 'fail'}"
              f"  sampled {hits}/{args.n}  ({time.perf_counter() - started:.0f} s)")

    summary = {"greedy_pass@1": sum(r["greedy"]["passed"] for r in results) / len(results),
               **{f"sampled_{key}": value for key, value in
                  summarize([[s["passed"] for s in r["samples"]] for r in results], [1, 5, 10]).items()}}
    receipt = {
        "schema_version": "1.0", "kind": "code_eval", "run": args.run, "checkpoint": state["checkpoint"],
        "mode": args.mode, "tasks": len(tasks), "n": args.n, "temperature": args.temperature,
        "top_k": args.top_k, "max_tokens": args.max_tokens, "seed": args.seed, "summary": summary,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.limit is None:
        out = ROOT / "artifacts" / "evals" / f"code-{args.run}-{args.mode}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
