"""Run the executable code evaluation on a run from the Hub and save a receipt.

    python scripts/run_code_eval.py --run s50m-001 --mode completion       # pretrained model
    python scripts/run_code_eval.py --run sft-s50m-001 --mode completion   # same prompts after SFT
    python scripts/run_code_eval.py --run sft-s50m-001 --mode chat         # asked as a chat request

For each of the 40 tasks: the loss of the reference solution, one greedy answer
(temperature 0) and --n sampled answers. Reports greedy pass@1 and sampled pass@1 / pass@k.
Writes artifacts/evals/code-<run>-<mode>.json with every generated function and its result.
Runs on CPU by default; notebooks/04_code_eval.ipynb calls `evaluate` on a GPU.
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
    answer_loss, extract_from_chat, extract_from_completion, load_tasks, run_tests, sample_many, summarize,
)
from glm_ladder.graph_decode import GraphDecoder  # noqa: E402
from glm_ladder.tokenizer import EOS_ID, load_tokenizer  # noqa: E402
from sample import load_run  # noqa: E402  (scripts/sample.py)


def evaluate(model, tokenizer, tasks, *, mode: str, n: int = 10, temperature: float = 0.8, top_k: int = 50,
             max_tokens: int = 160, seed: int = 0, loss_only: bool = False, verbose: bool = False,
             log=print, graphs: bool | None = None) -> tuple[list[dict], dict]:
    """Per-task results and the summary (reference loss, greedy pass@1, sampled pass@k).

    graphs: decode from CUDA graphs (default: when the model is on a GPU).
    """
    end_id = special_id(tokenizer, END_TOKEN)
    if graphs is None:
        graphs = next(model.parameters()).is_cuda
    decoders = {}

    def decoder(batch):
        if not graphs:
            return None
        if batch not in decoders:
            decoders[batch] = GraphDecoder(model, batch)
        return decoders[batch]

    if mode == "completion":
        def stop(ids):  # a new line that starts without indentation ends the function
            return EOS_ID in ids or re.search(r"\n\S", tokenizer.decode(ids)) is not None
    else:
        def stop(ids):
            return ids[-1] in (end_id, EOS_ID)

    def to_code(task, ids):
        text = tokenizer.decode([t for t in ids if t not in (end_id, EOS_ID)])
        if mode == "completion":
            return extract_from_completion(task.completion_prompt(), text)
        return extract_from_chat(text, task.name)

    def reference_ids(task, ids):
        """Token ids of the reference solution, as the model would write it in this mode."""
        if mode == "chat":
            answer = f"```python\ndef {task.name}({task.args}):\n{task.reference}```"
            return tokenizer.encode(answer).ids + [end_id]
        # Encode prompt + body together (as in training text); the prompt's tokens are a prefix.
        full = [EOS_ID, *tokenizer.encode(task.completion_prompt() + "\n" + task.reference.rstrip("\n")).ids]
        if full[:len(ids)] != ids:
            raise ValueError(f"{task.name}: prompt tokens are not a prefix of prompt + reference")
        return full[len(ids):]

    results, started = [], time.perf_counter()
    for index, task in enumerate(tasks):
        if mode == "completion":
            ids = [EOS_ID, *tokenizer.encode(task.completion_prompt()).ids]
        else:
            ids = prompt_ids(tokenizer, [{"role": "user", "content": task.chat_prompt()}])
        reference_loss = answer_loss(model, ids, reference_ids(task, ids))
        if loss_only:
            results.append({"task": task.name, "reference_loss": reference_loss})
            log(f"[{index + 1:2d}/{len(tasks)}] {task.name:22s} reference loss {reference_loss:.3f}")
            continue
        common = dict(max_new_tokens=max_tokens, top_k=top_k, seed=seed + index, stop=stop)
        greedy = sample_many(model, ids, n=1, temperature=0.0, decoder=decoder(1), **common)[0]
        sampled = sample_many(model, ids, n=n, temperature=temperature, decoder=decoder(n), **common)
        entry = {"task": task.name, "reference_loss": reference_loss, "greedy": None, "samples": []}
        for out, target in [(greedy, "greedy"), *[(s, "samples") for s in sampled]]:
            code = to_code(task, out)
            record = {"code": code, "passed": code is not None and run_tests(code, task.tests)}
            if target == "greedy":
                entry["greedy"] = record
            else:
                entry["samples"].append(record)
        results.append(entry)
        if verbose:
            log(f"--- {task.name}: raw greedy output ---\n{tokenizer.decode(greedy)!r}\n--- code tested ---\n"
                f"{entry['greedy']['code']}\n---")
        hits = sum(s["passed"] for s in entry["samples"])
        log(f"[{index + 1:2d}/{len(tasks)}] {task.name:22s} greedy {'PASS' if entry['greedy']['passed'] else 'fail'}"
            f"  sampled {hits}/{n}  ({time.perf_counter() - started:.0f} s)")

    summary = {"reference_loss": sum(r["reference_loss"] for r in results) / len(results)}
    if not loss_only:
        summary["greedy_pass@1"] = sum(r["greedy"]["passed"] for r in results) / len(results)
        summary.update({f"sampled_{key}": value for key, value in
                        summarize([[s["passed"] for s in r["samples"]] for r in results], [1, 5, 10]).items()})
    return results, summary


def receipt(run: str, checkpoint: str, mode: str, results: list[dict], summary: dict, *, device: str,
            n: int, temperature: float, top_k: int, max_tokens: int, seed: int, loss_only: bool) -> dict:
    return {
        "schema_version": "1.0", "kind": "code_eval", "run": run, "checkpoint": checkpoint, "mode": mode,
        "device": device, "tasks": len(results), "n": n, "temperature": temperature, "top_k": top_k,
        "max_tokens": max_tokens, "seed": seed, "loss_only": loss_only, "summary": summary, "results": results,
    }


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
    parser.add_argument("--loss-only", action="store_true",
                        help="only the reference-solution loss (seconds instead of minutes)")
    parser.add_argument("--verbose", action="store_true", help="print the raw greedy output and the code that was tested")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    api = HfApi(token=getpass("Hugging Face token: "))
    model, state = load_run(api, args.repo, args.run, args.checkpoint)
    tokenizer = load_tokenizer(ROOT / "tokenizer" / "glm-ladder-16k.json")
    tasks = load_tasks(ROOT / "evals" / "code_tasks.jsonl")[:args.limit]
    settings = dict(n=args.n, temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_tokens,
                    seed=args.seed, loss_only=args.loss_only)
    results, summary = evaluate(model, tokenizer, tasks, mode=args.mode, verbose=args.verbose, **settings)
    print(json.dumps(summary, indent=2))
    if args.limit is None:
        suffix = "-loss" if args.loss_only else ""
        out = ROOT / "artifacts" / "evals" / f"code-{args.run}-{args.mode}{suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = receipt(args.run, state["checkpoint"], args.mode, results, summary, device="cpu", **settings)
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()