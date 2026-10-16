"""Talk to a fine-tuned (SFT) run on CPU, or ask it a fixed set of questions.

    python scripts/talk.py --run sft-s50m-001                     # interactive, multi-turn
    python scripts/talk.py --run sft-s50m-001 --questions --out artifacts/samples/sft-s50m-001.md

Interactive commands: /reset clears the conversation, /quit exits.
The model sees exactly the training template: history, then <|assistant|>; it answers
until it produces <|end|>. Only the run's latest checkpoint weights are downloaded.
"""
from __future__ import annotations

import argparse
import sys
from getpass import getpass
from pathlib import Path

import torch
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from glm_ladder.chat import END_TOKEN, prompt_ids, special_id  # noqa: E402
from glm_ladder.generate import generate  # noqa: E402
from glm_ladder.tokenizer import EOS_ID, load_tokenizer  # noqa: E402
from sample import load_run  # noqa: E402  (scripts/sample.py)

QUESTIONS = [
    "What is the capital of France?",
    "Give me three tips for staying focused while studying.",
    "Explain what photosynthesis is in two sentences.",
    "Rewrite this sentence to sound more formal: hey, can u send me the report asap?",
    "Türkiye'nin başkenti neresidir?",
    "Sağlıklı beslenme için üç öneri ver.",
    "Write a Python function that returns the sum of a list of numbers.",
    "Write a Python function is_even(n) that returns True if n is even, otherwise False.",
]


def reply(model, tokenizer, messages, *, max_new_tokens: int, temperature: float, top_k: int, seed: int) -> str:
    end_id = special_id(tokenizer, END_TOKEN)
    new_ids = generate(model, prompt_ids(tokenizer, messages), max_new_tokens=max_new_tokens,
                       temperature=temperature, top_k=top_k, seed=seed, stop_id=end_id)
    finished = bool(new_ids) and new_ids[-1] == end_id
    if EOS_ID in new_ids:                         # <|endoftext|> also ends the answer
        new_ids, finished = new_ids[:new_ids.index(EOS_ID)], True
    text = tokenizer.decode([t for t in new_ids if t != end_id]).strip()
    return text if finished else text + " [cut: max tokens]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="an SFT run, e.g. sft-s50m-001")
    parser.add_argument("--repo", default="Tbt-AI/glm-ladder-runs")
    parser.add_argument("--checkpoint", help="default: latest")
    parser.add_argument("--questions", action="store_true", help="answer the fixed question set and exit")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="with --questions: also write Markdown (UTF-8)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    api = HfApi(token=getpass("Hugging Face token: "))
    model, state = load_run(api, args.repo, args.run, args.checkpoint)
    tokenizer = load_tokenizer(ROOT / "tokenizer" / "glm-ladder-16k.json")
    header = (f"{args.run} @ {state['checkpoint']}; temperature {args.temperature}, top-k {args.top_k}, "
              f"seed {args.seed}, at most {args.max_tokens} tokens per answer")
    print(header + "\n")
    settings = dict(max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, seed=args.seed)

    if args.questions:
        blocks = [f"# Chat samples: {args.run}\n\n{header}\n"]
        for question in QUESTIONS:
            answer = reply(model, tokenizer, [{"role": "user", "content": question}], **settings)
            print(f"USER: {question}\nMODEL: {answer}\n" + "-" * 80)
            blocks.append(f"**User:** {question}\n\n```text\n{answer}\n```\n")
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(blocks), encoding="utf-8")
            print("wrote", out)
        return

    messages: list[dict[str, str]] = []
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/reset":
            messages = []
            print("(conversation cleared)")
            continue
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        answer = reply(model, tokenizer, messages, **settings)
        print(f"model> {answer}\n")
        messages.append({"role": "assistant", "content": answer.removesuffix(" [cut: max tokens]")})


if __name__ == "__main__":
    main()