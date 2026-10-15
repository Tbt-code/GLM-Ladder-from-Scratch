"""Rebuild a pretraining receipt from a run stored on the Hugging Face Hub.

Only trainer.json and metrics.jsonl are downloaded (a few MB), never the weights,
so this runs on any laptop:

    python scripts/pretrain_receipt.py --run s50m-001

The token is read with a hidden prompt and kept in memory only.
"""
from __future__ import annotations

import argparse
import json
import sys
from getpass import getpass
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm_ladder import ModelConfig, expected_parameter_counts  # noqa: E402
from glm_ladder.runs import MetricsLog, RunStore  # noqa: E402


def build_receipt(api: HfApi, repo_id: str, run_name: str) -> dict:
    store = RunStore(api, repo_id, run_name)
    latest = store.latest()
    if latest is None:
        raise SystemExit(f"no checkpoints for {run_name} in {repo_id}")

    def download(*parts: str) -> Path:
        filename = "/".join(["runs", run_name, *parts])
        return Path(api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", force_download=True))

    state = json.loads(download(latest, "trainer.json").read_text())
    rows = MetricsLog(download("metrics.jsonl")).rows()
    train = [row for row in rows if "loss" in row and row["step"] <= state["step"]]
    evals = [row for row in rows if "eval" in row and row["step"] <= state["step"]]
    final = evals[-1] if evals else {}
    seconds = sum(row["seconds"] for row in train)
    config = state["pretrain_config"]
    tokens_per_step = config["batch_rows"] * config["seq_len"]
    total_steps = -(-config["total_tokens"] // tokens_per_step)   # ceil division
    metadata = state.get("metadata", {})

    return {
        "schema_version": "1.0", "kind": "pretraining_run", "run_name": run_name,
        "commit": metadata.get("commit"), "preset": metadata.get("preset"), "gpu": metadata.get("gpu"),
        "parameters": expected_parameter_counts(ModelConfig.from_dict(state["model_config"])),
        "pretrain_config": config,
        "steps": state["step"], "tokens_seen": state["tokens_seen"], "finished": state["step"] == total_steps,
        "train_hours": round(seconds / 3600, 2),
        "mean_tokens_per_second": round(len(train) * tokens_per_step / seconds) if seconds else None,
        "final_eval_step": final.get("step"),
        "final_eval": final.get("eval"), "final_eval_mixed": final.get("eval_mixed"),
        "checkpoints": f"{repo_id}/runs/{run_name} @ {latest}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run name, e.g. s50m-001")
    parser.add_argument("--repo", default="Tbt-AI/glm-ladder-runs")
    parser.add_argument("--out", help="default: artifacts/receipts/pretrain-<run>.json")
    args = parser.parse_args()

    api = HfApi(token=getpass("Hugging Face token: "))
    receipt = build_receipt(api, args.repo, args.run)
    out = Path(args.out or Path(__file__).resolve().parents[1] / "artifacts" / "receipts" / f"pretrain-{args.run}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
