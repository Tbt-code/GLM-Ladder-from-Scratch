"""Run bookkeeping: a JSON-lines metrics log and a checkpoint store on the Hugging Face Hub."""
from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import MODEL_FILE, OPTIMIZER_FILE, STATE_FILE


class MetricsLog:
    """Append-only JSON-lines log; every row has a "step"."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def truncate(self, step: int) -> None:
        """Drop rows logged after `step`: after a resume they would be repeated."""
        kept = [row for row in self.rows() if row["step"] <= step]
        self.path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept), encoding="utf-8")


class RunStore:
    """Mirror of one run's checkpoints in a Hugging Face model repo under runs/<run_name>/.

    Order of operations keeps the remote consistent if a session dies mid-way:
    upload the checkpoint folder -> update latest.json -> delete stale checkpoints.
    """

    def __init__(self, api, repo_id: str, run_name: str, *, keep_last: int = 2) -> None:
        self.api, self.repo_id, self.run_name, self.keep_last = api, repo_id, run_name, keep_last

    def _remote(self, *parts: str) -> str:
        return "/".join(["runs", self.run_name, *parts])

    def index(self) -> dict:
        name = self._remote("latest.json")
        if not self.api.file_exists(self.repo_id, name, repo_type="model"):
            return {"latest": None, "resume": [], "milestones": []}
        path = self.api.hf_hub_download(repo_id=self.repo_id, filename=name, repo_type="model", force_download=True)
        return json.loads(Path(path).read_text())

    def latest(self) -> str | None:
        return self.index()["latest"]

    def push(self, checkpoint_dir: str | Path, *, metrics: str | Path | None = None, milestone: bool = False) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        name = checkpoint_dir.name
        self.api.upload_folder(
            folder_path=str(checkpoint_dir), path_in_repo=self._remote(name),
            repo_id=self.repo_id, repo_type="model", commit_message=f"{self.run_name}: {name}",
        )
        if metrics is not None and Path(metrics).exists():
            self.api.upload_file(
                path_or_fileobj=str(metrics), path_in_repo=self._remote("metrics.jsonl"),
                repo_id=self.repo_id, repo_type="model", commit_message=f"{self.run_name}: metrics at {name}",
            )
        index = self.index()
        index["latest"] = name
        (index["milestones"] if milestone else index["resume"]).append(name)
        stale = index["resume"][:-self.keep_last]
        index["resume"] = index["resume"][-self.keep_last:]
        self.api.upload_file(
            path_or_fileobj=json.dumps(index, indent=2).encode(), path_in_repo=self._remote("latest.json"),
            repo_id=self.repo_id, repo_type="model", commit_message=f"{self.run_name}: latest -> {name}",
        )
        for old in stale:
            if old not in index["milestones"]:
                self.api.delete_folder(
                    path_in_repo=self._remote(old), repo_id=self.repo_id, repo_type="model",
                    commit_message=f"{self.run_name}: drop {old}",
                )

    def pull(self, name: str, local_root: str | Path) -> Path:
        """Download one checkpoint (and the metrics log, if any) under local_root; return its directory."""
        local_root = Path(local_root)
        for filename in (MODEL_FILE, OPTIMIZER_FILE, STATE_FILE):
            self.api.hf_hub_download(
                repo_id=self.repo_id, filename=self._remote(name, filename),
                repo_type="model", local_dir=str(local_root),
            )
        metrics = self._remote("metrics.jsonl")
        if self.api.file_exists(self.repo_id, metrics, repo_type="model"):
            self.api.hf_hub_download(repo_id=self.repo_id, filename=metrics, repo_type="model",
                                     local_dir=str(local_root), force_download=True)
        return local_root / "runs" / self.run_name / name