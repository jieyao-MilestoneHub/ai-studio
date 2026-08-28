"""The eval/ file-writing layer. `harness.shard.split_into_shards` only
splits in memory; nothing before this module wrote `eval/in/<run_id>/
shard-NNN.jsonl` or read `eval/out/<run_id>/shard-NNN.jsonl` to/from disk.

Follows the `eval-harness` skill's nested-shard directory contract — more
specific than EVAL.md §6.2's flat shorthand, and the shape the `eval-judge`
agent's dispatch actually reads/writes against:

    eval/in/<run_id>/shard-NNN.jsonl
    eval/out/<run_id>/shard-NNN.jsonl
    eval/report/<run_id>.md
"""

from __future__ import annotations

import hashlib
import json

import fsspec
from pydantic import BaseModel, ConfigDict

from twin.harness.report import EvalReport, render_report_md
from twin.harness.schema import HarnessError, JudgedItem, StrippedSample


def _ensure_parent(uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)


def write_shards(shards: list[list[StrippedSample]], *, run_id: str, root_uri: str) -> list[str]:
    """Writes one `<root_uri>/in/<run_id>/shard-NNN.jsonl` per shard, returns
    the written URIs — the value `RunManifest.shard_ids` records."""
    uris: list[str] = []
    for index, shard in enumerate(shards):
        uri = f"{root_uri.rstrip('/')}/in/{run_id}/shard-{index:03d}.jsonl"
        _ensure_parent(uri)
        with fsspec.open(uri, "w", encoding="utf-8") as f:
            for sample in shard:
                f.write(sample.model_dump_json())
                f.write("\n")
        uris.append(uri)
    return uris


def compute_rubric_hash(rubric_uri: str, *, shard_size: int) -> str:
    """eval-harness skill step 2: shard size MUST feed `rubric_hash` — it
    determines how much context the judge sees per call, which is as much
    part of "the rubric" as the file's text. Hashes the rubric file's actual
    bytes via fsspec (same byte-hashing approach as `core.hashing.
    adapter_hash`), not its path, so a rename doesn't spuriously change the
    hash and a silent edit can't hide behind an unchanged one."""
    hasher = hashlib.sha256()
    with fsspec.open(rubric_uri, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    hasher.update(str(shard_size).encode("utf-8"))
    return hasher.hexdigest()


class JudgeLineQuality(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    unjudgeable: int
    low_confidence: int
    flag_counts: dict[str, int]


_REQUIRED_FIELDS = ("sample_id", "verdict", "reason")


def read_judged_shard(uri: str, *, unjudgeable_verdict: str) -> tuple[list[JudgedItem], JudgeLineQuality]:
    """Reads one `eval/out/<run_id>/shard-NNN.jsonl` file written by the
    `eval-judge` agent. Its actual output line is `{"sample_id", "verdict",
    "reason", "confidence", "flags"}` (`.claude/agents/eval-judge.md`) —
    `harness.schema.JudgedItem` has `{sample_id, verdict, rationale}`. This
    function is the explicit adapter: `reason` -> `rationale`. `confidence`
    and `flags` are tallied into `JudgeLineQuality` rather than discarded, so
    a `source_leak` flag or a `low` confidence line stays visible to whoever
    reads the aggregation output."""
    judged: list[JudgedItem] = []
    total = 0
    unjudgeable = 0
    low_confidence = 0
    flag_counts: dict[str, int] = {}

    with fsspec.open(uri, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            missing = [field for field in _REQUIRED_FIELDS if field not in data]
            if missing:
                raise HarnessError(f"{uri}:{line_number} is missing required field(s) {missing}")

            total += 1
            judged.append(JudgedItem(sample_id=data["sample_id"], verdict=data["verdict"], rationale=data["reason"]))
            if data["verdict"] == unjudgeable_verdict:
                unjudgeable += 1
            if data.get("confidence") == "low":
                low_confidence += 1
            for flag in data.get("flags", []):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return judged, JudgeLineQuality(
        total=total, unjudgeable=unjudgeable, low_confidence=low_confidence, flag_counts=flag_counts
    )


def write_report(report: EvalReport, uri: str) -> None:
    _ensure_parent(uri)
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        f.write(render_report_md(report))
