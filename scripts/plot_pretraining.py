"""Plot validation curves of several pretraining runs from the Hugging Face Hub on shared axes.

Downloads only metrics.jsonl and the latest trainer.json of each run, so it runs on a laptop:

    python scripts/plot_pretraining.py --runs s50m-001 m100m-001

Writes artifacts/plots/pretraining.png: mixed validation loss (nats per token) and, per
source, bits per UTF-8 byte, which is comparable across languages and tokenizers.
"""
from __future__ import annotations

import argparse
import json
import math
from getpass import getpass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("english", "code", "turkish")

# Categorical slots in fixed order: run 1 blue, run 2 orange, run 3 aqua. Each run also gets
# its own line style, so the runs stay distinguishable without color.
SERIES = [("#2a78d6", "--"), ("#eb6834", "-"), ("#1baf7a", ":")]
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def fetch_run(api, repo_id: str, run_name: str) -> dict:
    """Return {"label", "tokens", "mixed", "<source>": [...]} for the run's eval rows."""
    def download(filename: str) -> Path:
        return Path(api.hf_hub_download(repo_id=repo_id, filename=f"runs/{run_name}/{filename}",
                                        repo_type="model", force_download=True))

    latest = json.loads(download("latest.json").read_text())["latest"]
    state = json.loads(download(f"{latest}/trainer.json").read_text())
    rows = [json.loads(line) for line in download("metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    config = state["pretrain_config"]
    tokens_per_step = config["batch_rows"] * config["seq_len"]
    evals = [row for row in rows if "eval" in row and row["step"] <= state["step"]]
    preset = state.get("metadata", {}).get("preset", run_name)
    return {
        "label": preset,
        "tokens": [row["step"] * tokens_per_step for row in evals],
        "mixed": [row["eval_mixed"] for row in evals],
        **{source: [row["eval"][source] for row in evals] for source in SOURCES},
    }


def plot(runs: list[dict], bytes_per_token: dict[str, float], out: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    panels = [("mixed", "Validation loss, token-weighted mix", "nats per token")]
    panels += [(source, f"{source.capitalize()} validation", "bits per byte") for source in SOURCES]

    for ax, (key, title, unit) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for run, (color, style) in zip(runs, SERIES):
            values = run[key]
            if key != "mixed":
                values = [loss / math.log(2) / bytes_per_token[key] for loss in values]
            billions = [t / 1e9 for t in run["tokens"]]
            ax.plot(billions, values, color=color, linestyle=style, linewidth=2, label=run["label"])
            ax.plot(billions[-1], values[-1], "o", color=color, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{values[-1]:.3f}", (billions[-1], values[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("training tokens (billions)", color=MUTED)
        ax.set_ylabel(unit, color=MUTED)
        # Skip the first eval (step 250): its loss is far above the rest and flattens the curves.
        tail = [v for run in runs for v in (run[key][1:] if key == "mixed" else
                [x / math.log(2) / bytes_per_token[key] for x in run[key][1:]])]
        if tail:
            low, high = min(tail), max(tail)
            ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
        ax.set_xlim(left=0, right=max(t for run in runs for t in run["tokens"]) / 1e9 * 1.12)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="in plotting order, e.g. s50m-001 m100m-001")
    parser.add_argument("--repo", default="Tbt-AI/glm-ladder-runs")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "plots" / "pretraining.png"))
    args = parser.parse_args()
    if len(args.runs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs per plot")

    api = HfApi(token=getpass("Hugging Face token: "))
    receipt = json.loads((ROOT / "artifacts" / "receipts" / "tokenizer-16k.json").read_text())
    plot([fetch_run(api, args.repo, run) for run in args.runs], receipt["bytes_per_token"], Path(args.out))


if __name__ == "__main__":
    main()
