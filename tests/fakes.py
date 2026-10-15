"""Test doubles shared by several test files."""
from __future__ import annotations

import shutil
from pathlib import Path


class FakeHub:
    """Same methods as HfApi, backed by a local folder."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _p(self, name: str) -> Path:
        return self.root / name

    def file_exists(self, repo_id, filename, *, repo_type=None) -> bool:
        return self._p(filename).is_file()

    def upload_folder(self, *, folder_path, path_in_repo, **_) -> None:
        shutil.copytree(folder_path, self._p(path_in_repo), dirs_exist_ok=True)

    def upload_file(self, *, path_or_fileobj, path_in_repo, **_) -> None:
        target = self._p(path_in_repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(path_or_fileobj, bytes):
            target.write_bytes(path_or_fileobj)
        else:
            shutil.copyfile(path_or_fileobj, target)

    def hf_hub_download(self, *, filename, local_dir=None, **_) -> str:
        if local_dir is None:
            return str(self._p(filename))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._p(filename), target)
        return str(target)

    def delete_folder(self, *, path_in_repo, **_) -> None:
        shutil.rmtree(self._p(path_in_repo))
