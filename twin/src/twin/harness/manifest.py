"""Run manifests. eval-harness skill step 0/1:

- step 0: refuse to re-run an identical (dataset_hash, rubric_hash, split) —
  "上次分數不好想再跑一次" is EVAL.md §12 anti-pattern 10, not a valid reason.
- step 1: a manifest MUST be written before any judge runs, and MUST NOT be
  modified after — changing it means a new run_id, not an edit.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

import fsspec
from pydantic import BaseModel, ConfigDict

from twin.harness.schema import HarnessError


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    base_model: str
    adapter_hash: str
    dataset_hash: str
    eval_set_version: str
    rubric_hash: str
    date: datetime
    split: Literal["heldout", "sealed"]
    shard_ids: list[str]


def find_existing_run(
    *, dataset_hash: str, rubric_hash: str, split: str, runs_root: str
) -> RunManifest | None:
    """Scans `runs_root` (a directory of `<run_id>.json` manifests) for one
    matching this exact (dataset_hash, rubric_hash, split) triple."""
    pattern = f"{runs_root.rstrip('/')}/*.json"
    for open_file in fsspec.open_files(pattern, mode="r", encoding="utf-8"):
        with open_file as f:
            data = json.loads(f.read())
        manifest = RunManifest.model_validate(data)
        if manifest.dataset_hash == dataset_hash and manifest.rubric_hash == rubric_hash and manifest.split == split:
            return manifest
    return None


def write_manifest_once(manifest: RunManifest, uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        raise HarnessError(
            f"a manifest already exists at {uri} — manifests MUST NOT be "
            f"modified once written (eval-harness skill step 1); write a new "
            f"run_id instead of editing this one"
        )
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json())
