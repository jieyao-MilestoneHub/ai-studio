"""Delivery/incoming pruning: the always-on host must not fill its disk."""

from __future__ import annotations

import os
import time
from pathlib import Path

from ai_studio.storage.retention import sweep_old_files


def _touch(path: Path, *, age_days: float) -> Path:
    path.write_bytes(b"x" * 1024)
    old = time.time() - age_days * 86_400
    os.utime(path, (old, old))
    return path


def test_only_files_older_than_the_window_go(tmp_path: Path) -> None:
    old = _touch(tmp_path / "old.mp4", age_days=10)
    fresh = _touch(tmp_path / "fresh.mp4", age_days=1)

    result = sweep_old_files(tmp_path, max_age_days=7)

    assert not old.exists() and fresh.exists()
    assert result.removed == 1 and result.kept == 1
    assert result.freed_bytes == 1024


def test_a_protected_file_is_never_removed(tmp_path: Path) -> None:
    """An image-to-video photo a still-live job points at outlives the window."""
    keep = _touch(tmp_path / "photo.jpg", age_days=30)

    result = sweep_old_files(tmp_path, max_age_days=7, keep={str(keep.resolve())})

    assert keep.exists() and result.removed == 0 and result.kept == 1


def test_dry_run_removes_nothing(tmp_path: Path) -> None:
    old = _touch(tmp_path / "old.mp4", age_days=10)
    result = sweep_old_files(tmp_path, max_age_days=7, dry_run=True)
    assert old.exists() and result.removed == 1  # reported, not deleted


def test_a_missing_directory_is_a_clean_zero(tmp_path: Path) -> None:
    result = sweep_old_files(tmp_path / "nope", max_age_days=7)
    assert result.removed == 0 and result.kept == 0


def test_subdirectories_are_left_alone(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _touch(tmp_path / "sub" / "old.mp4", age_days=99)
    result = sweep_old_files(tmp_path, max_age_days=7)
    assert result.removed == 0 and (tmp_path / "sub" / "old.mp4").exists()
