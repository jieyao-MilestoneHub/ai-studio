"""Fragment persistence round-trip. SPEC.md §7.2 — paths are URIs via fsspec."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from twin.core.enums import Modality, SourceClass, Split
from twin.core.fragment import EventTime, Fragment
from twin.ingest.store import read_fragments_jsonl, write_fragments_jsonl


def _fragment(content: str) -> Fragment:
    return Fragment(
        principal_id="p1",
        source_class=SourceClass.SELF_REPORT,
        modality=Modality.TEXT,
        content=content,
        event_time=EventTime(value="2024-06", precision="month", confidence=0.8),
        ingest_time=datetime(2026, 8, 27, tzinfo=UTC),
        split=Split.TRAIN,
    )


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    fragments = [_fragment("a"), _fragment("b"), _fragment("c")]
    uri = f"file://{tmp_path}/fragments.jsonl"

    written = write_fragments_jsonl(fragments, uri)
    assert written == 3

    read_back = list(read_fragments_jsonl(uri))
    assert [f.fragment_id for f in read_back] == [f.fragment_id for f in fragments]
    assert [f.content for f in read_back] == ["a", "b", "c"]


def test_write_overwrites_rather_than_appends(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/fragments.jsonl"
    write_fragments_jsonl([_fragment("first-run")], uri)
    write_fragments_jsonl([_fragment("second-run-a"), _fragment("second-run-b")], uri)

    read_back = list(read_fragments_jsonl(uri))
    assert [f.content for f in read_back] == ["second-run-a", "second-run-b"]
