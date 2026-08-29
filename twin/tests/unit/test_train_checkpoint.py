"""Checkpoint completeness/discovery/encryption — pure `file://` fixture
tests, no subprocess/GPU/model. SPEC.md §7.4, §8."""

from __future__ import annotations

from pathlib import Path

import pytest

from twin.core.encryption import InvalidToken, generate_key
from twin.train.checkpoint import (
    ADAPTER_SAVE_ARTIFACTS,
    COMPLETION_MARKER,
    REQUIRED_CHECKPOINT_ARTIFACTS,
    download_checkpoint_for_resume,
    find_latest_complete_checkpoint,
    is_checkpoint_complete,
    list_checkpoints,
    prune_remote_checkpoints,
    upload_adapter,
    upload_checkpoint,
)


def _write_local_checkpoint(local_dir: Path) -> None:
    local_dir.mkdir(parents=True)
    for artifact in REQUIRED_CHECKPOINT_ARTIFACTS:
        (local_dir / artifact).write_text("x", encoding="utf-8")


def test_upload_checkpoint_then_is_checkpoint_complete(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_checkpoint(local)
    remote = f"file://{tmp_path / 'remote' / 'checkpoint-1'}"

    upload_checkpoint(str(local), remote, encryption_key=generate_key())

    assert is_checkpoint_complete(remote) is True


def test_upload_checkpoint_rejects_incomplete_local_dir(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "adapter_model.safetensors").write_text("x", encoding="utf-8")  # missing the other 5

    with pytest.raises(AssertionError):
        upload_checkpoint(str(local), f"file://{tmp_path / 'remote'}", encryption_key=generate_key())


def test_is_checkpoint_complete_false_when_marker_missing(tmp_path: Path) -> None:
    """The exact torn-upload case: the archive landed but the process was
    killed before the completion marker was written last."""
    local = tmp_path / "local"
    _write_local_checkpoint(local)
    remote_dir = tmp_path / "remote" / "checkpoint-1"
    upload_checkpoint(str(local), f"file://{remote_dir}", encryption_key=generate_key())

    (remote_dir / COMPLETION_MARKER).unlink()

    assert is_checkpoint_complete(f"file://{remote_dir}") is False


def test_is_checkpoint_complete_false_when_directory_does_not_exist(tmp_path: Path) -> None:
    assert is_checkpoint_complete(f"file://{tmp_path / 'never-existed'}") is False


def test_list_checkpoints_is_empty_for_a_fresh_run_root(tmp_path: Path) -> None:
    assert list_checkpoints(f"file://{tmp_path / 'never-existed'}") == []


def test_list_checkpoints_sorted_ascending_by_step(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for step in (30, 10, 20):
        local = tmp_path / f"local-{step}"
        _write_local_checkpoint(local)
        upload_checkpoint(str(local), f"file://{run_root}/checkpoint-{step}", encryption_key=generate_key())

    pairs = list_checkpoints(f"file://{run_root}")

    assert [step for step, _uri in pairs] == [10, 20, 30]


def test_find_latest_complete_checkpoint_skips_incomplete_newest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    local_10 = tmp_path / "local-10"
    _write_local_checkpoint(local_10)
    upload_checkpoint(str(local_10), f"file://{run_root}/checkpoint-10", encryption_key=generate_key())
    local_20 = tmp_path / "local-20"
    _write_local_checkpoint(local_20)
    upload_checkpoint(str(local_20), f"file://{run_root}/checkpoint-20", encryption_key=generate_key())
    (run_root / "checkpoint-20" / COMPLETION_MARKER).unlink()  # torn upload, newest by step

    latest = find_latest_complete_checkpoint(f"file://{run_root}")

    assert latest is not None
    assert latest.endswith("checkpoint-10")


def test_find_latest_complete_checkpoint_none_for_fresh_run(tmp_path: Path) -> None:
    assert find_latest_complete_checkpoint(f"file://{tmp_path / 'never-existed'}") is None


def test_download_checkpoint_for_resume_round_trips(tmp_path: Path) -> None:
    key = generate_key()
    local = tmp_path / "local"
    _write_local_checkpoint(local)
    remote = f"file://{tmp_path / 'remote' / 'checkpoint-5'}"
    upload_checkpoint(str(local), remote, encryption_key=key)

    downloaded = download_checkpoint_for_resume(remote, str(tmp_path / "scratch"), encryption_key=key)

    for artifact in REQUIRED_CHECKPOINT_ARTIFACTS:
        assert (Path(downloaded) / artifact).read_text(encoding="utf-8") == "x"


def test_download_checkpoint_for_resume_fails_loudly_with_wrong_key(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_checkpoint(local)
    remote = f"file://{tmp_path / 'remote' / 'checkpoint-5'}"
    upload_checkpoint(str(local), remote, encryption_key=generate_key())

    with pytest.raises(InvalidToken):
        download_checkpoint_for_resume(remote, str(tmp_path / "scratch"), encryption_key=generate_key())


def test_upload_adapter_narrower_artifact_check(tmp_path: Path) -> None:
    """A bare `save_pretrained` output has none of the optimizer/scheduler/
    RNG/trainer-state files a Trainer checkpoint does — `upload_adapter`
    MUST NOT require them."""
    local = tmp_path / "local_adapter"
    local.mkdir()
    for artifact in ADAPTER_SAVE_ARTIFACTS:
        (local / artifact).write_text("x", encoding="utf-8")
    remote = f"file://{tmp_path / 'remote' / 'final'}"

    upload_adapter(str(local), remote, encryption_key=generate_key())

    assert is_checkpoint_complete(remote) is True


def test_upload_adapter_rejects_missing_artifacts(tmp_path: Path) -> None:
    local = tmp_path / "local_adapter"
    local.mkdir()

    with pytest.raises(AssertionError):
        upload_adapter(str(local), f"file://{tmp_path / 'remote'}", encryption_key=generate_key())


def test_prune_remote_checkpoints_keeps_latest_two_complete(tmp_path: Path) -> None:
    key = generate_key()
    run_root = f"file://{tmp_path / 'remote'}"
    for step in (30, 55, 80, 105):
        local = tmp_path / f"local-{step}"
        _write_local_checkpoint(local)
        upload_checkpoint(str(local), f"{run_root}/checkpoint-{step}", encryption_key=key)
    # a torn newest upload: archive present, no completion marker
    torn = tmp_path / "remote" / "checkpoint-130"
    torn.mkdir()
    (torn / "adapter_weights.tar.enc").write_bytes(b"partial")

    removed = prune_remote_checkpoints(run_root)

    remaining = sorted(p.name for p in (tmp_path / "remote").iterdir())
    assert remaining == ["checkpoint-105", "checkpoint-130", "checkpoint-80"]
    assert sorted(removed) == [f"{run_root}/checkpoint-30", f"{run_root}/checkpoint-55"]
    assert find_latest_complete_checkpoint(run_root) == f"{run_root}/checkpoint-105"
