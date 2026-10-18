"""Download receipts that a GPU notebook uploaded to the runs repo, ready to commit.

    python scripts/fetch_evals.py                                                    # notebook 04: code evals
    python scripts/fetch_evals.py --pattern "finetune/*.json" --to artifacts/finetune  # notebook 05

Copies the matching files from the runs repo into the target folder (default artifacts/evals/).
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
    parser.add_argument("--pattern", default="evals/*.json", help="files in the runs repo, e.g. finetune/*.json")
    parser.add_argument("--to", default="artifacts/evals", help="target folder, relative to the repository root")
    args = parser.parse_args()

    folder = Path(snapshot_download(repo_id=args.repo, repo_type="model", allow_patterns=[args.pattern],
                                    token=getpass("Hugging Face token: ")))
    target = ROOT / args.to
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(folder.glob(args.pattern)):
        shutil.copyfile(path, target / path.name)
        print("wrote", target / path.name)


if __name__ == "__main__":
    main()