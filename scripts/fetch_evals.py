"""Download the code-eval receipts that notebooks/04_code_eval.ipynb uploaded to the Hub.

    python scripts/fetch_evals.py

Copies evals/*.json from the runs repo into artifacts/evals/, ready to commit.
The token is read with a hidden prompt and kept in memory only.
"""
from __future__ import annotations

import argparse
import shutil
from getpass import getpass
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="Tbt-AI/glm-ladder-runs")
    args = parser.parse_args()

    folder = Path(snapshot_download(repo_id=args.repo, repo_type="model", allow_patterns=["evals/*.json"],
                                    token=getpass("Hugging Face token: ")))
    target = ROOT / "artifacts" / "evals"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted((folder / "evals").glob("*.json")):
        shutil.copyfile(path, target / path.name)
        print("wrote", target / path.name)


if __name__ == "__main__":
    main()