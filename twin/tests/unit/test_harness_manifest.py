"""Run manifests. eval-harness skill step 0/1."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from twin.harness.manifest import RunManifest, find_existing_run, write_manifest_once
from twin.harness.schema import HarnessError


def _manifest(**overrides: object) -> RunManifest:
    defaults: dict[str, object] = dict(
        run_id="run-001",
        base_model="some-org/some-8b-model",
        adapter_hash="a" * 64,
        dataset_hash="b" * 64,
        eval_set_version="v1",
        rubric_hash="c" * 64,
        date=datetime(2026, 8, 27, tzinfo=UTC),
        split="heldout",
        shard_ids=["shard-000"],
    )
    defaults.update(overrides)
    return RunManifest(**defaults)  # type: ignore[arg-type]


def test_write_manifest_once_then_find_existing_run(tmp_path: Path) -> None:
    manifest = _manifest()
    write_manifest_once(manifest, f"file://{tmp_path}/run-001.json")

    found = find_existing_run(
        dataset_hash="b" * 64, rubric_hash="c" * 64, split="heldout", runs_root=f"file://{tmp_path}"
    )
    assert found is not None
    assert found.run_id == "run-001"


def test_find_existing_run_returns_none_when_nothing_matches(tmp_path: Path) -> None:
    write_manifest_once(_manifest(), f"file://{tmp_path}/run-001.json")
    found = find_existing_run(
        dataset_hash="different", rubric_hash="c" * 64, split="heldout", runs_root=f"file://{tmp_path}"
    )
    assert found is None


def test_find_existing_run_on_an_empty_directory_returns_none(tmp_path: Path) -> None:
    found = find_existing_run(
        dataset_hash="b" * 64, rubric_hash="c" * 64, split="heldout", runs_root=f"file://{tmp_path}/nonexistent"
    )
    assert found is None


def test_write_manifest_once_refuses_to_overwrite(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/run-001.json"
    write_manifest_once(_manifest(), uri)
    with pytest.raises(HarnessError, match="already exists"):
        write_manifest_once(_manifest(run_id="run-001-edited"), uri)
