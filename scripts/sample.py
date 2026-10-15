"""Generate text from a pretraining run stored on the Hugging Face Hub, on CPU.

Downloads only model.safetensors and trainer.json of the run's latest checkpoint
(cached by huggingface_hub, so a second call is instant), checks the SHA-256,
and continues a few prompts:

    python scripts/sample.py --run s50m-001
    python scripts/sample.py --run s50m-001 --prompt "Bir varmış, bir yokmuş" --temperature 0.7

The token is read with a hidden prompt and kept in memory only.
"""
from __future__ import annotations

import argparse
import json
import sys
from getpass import getpass
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glm_ladder import GLMLadderModel, ModelConfig  # noqa: E402
from glm_ladder.checkpoint import MODEL_FILE, STATE_FILE, sha256_file  # noqa: E402
from glm_ladder.generate import generate  # noqa: E402
from glm_ladder.runs import RunStore  # noqa: E402
from glm_ladder.tokenizer import EOS_ID, load_tokenizer  # noqa: E402

DEFAULT_PROMPTS = [
    "The history of the Roman Empire",
    "Photosynthesis is the process by which",
    "Türkiye'nin en kalabalık şehri",
    "Yapay zeka, günümüzde",
    "def fibonacci(n):\n",
    "import numpy as np\n\n\ndef normalize(x):\n",
]


def load_run(api: HfApi, repo_id: str, run_name: str, checkpoint: str | None) -> tuple[GLMLadderModel, dict]:
    name = checkpoint or RunStore(api, repo_id, run_name).latest()
    if name is None:
        raise SystemExit(f"no checkpoints for {run_name} in {repo_id}")

    def download(filename: str) -> Path:
        return Path(api.hf_hub_download(repo_id=repo_id, filename=f"runs/{run_name}/{name}/{filename}",
                                        repo_type="model"))

    state = json.loads(download(STATE_FILE).read_text())
    weights = download(MODEL_FILE)
    if sha256_file(weights) != state["model_sha256"]:
        raise SystemExit("model.safetensors does not match the SHA-256 in trainer.json")
    model = GLMLadderModel(ModelConfig.from_dict(state["model_config"]))
    load_model(model, str(weights), strict=True)
    model.eval()
    return model, {**state, "checkpoint": name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", default="Tbt-AI/glm-ladder-runs")
    parser.add_argument("--checkpoint", help="e.g. step-004044 (default: latest)")
    parser.add_argument("--prompt", action="append", help="repeatable; default: a fixed EN/TR/code set")
    parser.add_argument("--tokens", type=int, default=80, help="new tokens per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    api = HfApi(token=getpass("Hugging Face token: "))
    model, state = load_run(api, args.repo, args.run, args.checkpoint)
    tokenizer = load_tokenizer(ROOT / "tokenizer" / "glm-ladder-16k.json")
    print(f"{args.run} @ {state['checkpoint']}: step {state['step']}, {state['tokens_seen']:,} tokens; "
          f"temperature {args.temperature}, top-k {args.top_k}, seed {args.seed}\n")

    for prompt in args.prompt or DEFAULT_PROMPTS:
        # Every training document follows an <|endoftext|>, so start the context the same way.
        prompt_ids = [EOS_ID, *tokenizer.encode(prompt).ids]
        new_ids = generate(model, prompt_ids, max_new_tokens=args.tokens, temperature=args.temperature,
                           top_k=args.top_k, seed=args.seed)
        ended = " [end of document]" if new_ids and new_ids[-1] == EOS_ID else ""
        print("=" * 80)
        print(prompt + tokenizer.decode(new_ids) + ended)
    print("=" * 80)


if __name__ == "__main__":
    main()
