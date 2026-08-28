"""The eval/ file-writing layer: shard writing, rubric_hash, and reading a
judged shard back — including the eval-judge agent's real `reason` field
mapping onto `JudgedItem.rationale`.
"""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import pytest

from twin.harness.eval_io import compute_rubric_hash, read_judged_shard, write_report, write_shards
from twin.harness.report import EvalReport
from twin.harness.schema import HarnessError, StrippedSample


def _stripped(sample_id: str) -> StrippedSample:
    return StrippedSample(sample_id=sample_id, content=f"content-{sample_id}", suite="s1")


def test_write_shards_writes_one_file_per_shard_under_run_id(tmp_path: Path) -> None:
    shards = [[_stripped("a"), _stripped("b")], [_stripped("c")]]
    uris = write_shards(shards, run_id="run-1", root_uri=f"file://{tmp_path}")

    assert uris == [
        f"file://{tmp_path}/in/run-1/shard-000.jsonl",
        f"file://{tmp_path}/in/run-1/shard-001.jsonl",
    ]
    with fsspec.open(uris[0], "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert [line["sample_id"] for line in lines] == ["a", "b"]


def test_compute_rubric_hash_changes_when_shard_size_changes(tmp_path: Path) -> None:
    rubric_uri = f"file://{tmp_path}/rubric.md"
    with fsspec.open(rubric_uri, "w", encoding="utf-8") as f:
        f.write("judge instructions")

    assert compute_rubric_hash(rubric_uri, shard_size=20) != compute_rubric_hash(rubric_uri, shard_size=30)
    assert compute_rubric_hash(rubric_uri, shard_size=20) == compute_rubric_hash(rubric_uri, shard_size=20)


def _write_judged_lines(uri: str, lines: list[dict[str, object]]) -> None:
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line))
            f.write("\n")


def test_read_judged_shard_maps_reason_to_rationale(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/shard-000.jsonl"
    _write_judged_lines(
        uri,
        [{"sample_id": "a", "verdict": "match", "reason": "agrees with R2", "confidence": "high", "flags": []}],
    )

    judged, quality = read_judged_shard(uri, unjudgeable_verdict="unjudgeable")

    assert judged[0].sample_id == "a"
    assert judged[0].verdict == "match"
    assert judged[0].rationale == "agrees with R2"
    assert quality.total == 1
    assert quality.unjudgeable == 0
    assert quality.low_confidence == 0


def test_read_judged_shard_raises_on_a_line_missing_required_fields(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/shard-000.jsonl"
    _write_judged_lines(uri, [{"sample_id": "a", "verdict": "match"}])  # no "reason"

    with pytest.raises(HarnessError, match="reason"):
        read_judged_shard(uri, unjudgeable_verdict="unjudgeable")


def test_read_judged_shard_tallies_flags_and_low_confidence(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/shard-000.jsonl"
    _write_judged_lines(
        uri,
        [
            {"sample_id": "a", "verdict": "match", "reason": "x", "confidence": "low", "flags": ["source_leak"]},
            {"sample_id": "b", "verdict": "unjudgeable", "reason": "y", "confidence": "high", "flags": []},
        ],
    )

    judged, quality = read_judged_shard(uri, unjudgeable_verdict="unjudgeable")

    assert len(judged) == 2
    assert quality.unjudgeable == 1
    assert quality.low_confidence == 1
    assert quality.flag_counts == {"source_leak": 1}


def test_write_report_renders_to_disk(tmp_path: Path) -> None:
    report = EvalReport(
        run_id="run-1",
        base_model="Qwen/Qwen3-8B",
        adapter_hash="none",
        dataset_hash="x" * 64,
        eval_set_version="s1-v1",
        date="2026-08-28T00:00:00Z",  # type: ignore[arg-type]
        judge_agreement=0.9,
        judge_rubric_hash="y" * 64,
        confidence="high",
        gate_level="L0",  # type: ignore[arg-type]
    )
    uri = f"file://{tmp_path}/report.md"

    write_report(report, uri)

    with fsspec.open(uri, "r", encoding="utf-8") as f:
        content = f.read()
    assert "run_id: run-1" in content
