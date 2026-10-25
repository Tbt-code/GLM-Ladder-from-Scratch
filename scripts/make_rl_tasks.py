"""Write the generated RL tasks to evals/rl_tasks/{train,test_id,test_ood}.jsonl (deterministic).

    python scripts/make_rl_tasks.py

Same format as evals/code_tasks.jsonl, so load_tasks reads them. The files are committed:
anyone can check exactly which tasks RL trained on and which it was tested on.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glm_ladder.code_eval import load_tasks  # noqa: E402
from glm_ladder.task_gen import OOD_FAMILIES, SEED, generate_splits  # noqa: E402


def main() -> None:
    hand_written = {task.name for task in load_tasks(ROOT / "evals" / "code_tasks.jsonl")}
    splits = generate_splits()
    out = ROOT / "evals" / "rl_tasks"
    out.mkdir(parents=True, exist_ok=True)
    for split, tasks in splits.items():
        clashes = [task.name for task in tasks if task.name in hand_written]
        if clashes:
            raise SystemExit(f"generated tasks reuse hand-written names: {clashes}")
        with open(out / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as handle:
            for task in tasks:
                handle.write(json.dumps({"name": task.name, "args": task.args, "doc": task.doc,
                                         "tests": list(task.tests), "reference": task.reference}) + "\n")
        print(f"{split:9s} {len(tasks):4d} tasks -> {out / f'{split}.jsonl'}")
    summary = {"seed": SEED, "ood_families": list(OOD_FAMILIES),
               "counts": {split: len(tasks) for split, tasks in splits.items()}}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    first = Counter(name.split("_")[0] for name in (t.name for t in splits["train"]))
    print("most common name stems in train:", first.most_common(8))


if __name__ == "__main__":
    main()
