"""Fragment persistence round-trip. SPEC.md §7.2 — paths are URIs via fsspec."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from twin.core.enums import Modality, SourceClass, Split
from twin.core.fragment import EventTime, Fragment
from twin.ingest.store import read_fragments_jsonl, write_fragments_jsonl


def _fragment(content: str, *, source_class: SourceClass = SourceClass.SELF_REPORT) -> Fragment:
    return Fragment(
        principal_id="p1",
        source_class=source_class,
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


def test_refuses_to_write_a_self_report_fragment_to_a_non_file_uri() -> None:
    """INTERVIEW.md §6.3/§8 I-D: interview-transcript text becomes
    SELF_REPORT `Fragment.content` verbatim (SPEC.md D26) — the actual
    enforcement point for "MUST NOT enter cross-cloud storage" is here, not
    just the `transcript_store_uri` setting (which only constrains raw
    audio/transcript *files*, not `Fragment` records)."""
    with pytest.raises(ValueError, match="SELF_REPORT"):
        write_fragments_jsonl([_fragment("secret")], "r2://twin-checkpoints/fragments.jsonl")


def test_refuses_before_writing_any_byte_on_a_mixed_batch(tmp_path: Path) -> None:
    """Fail-fast: materializes and checks the whole batch before opening the
    destination for writing — a rejected batch MUST NOT leave a partial file
    behind, matching the rest of this codebase's write-once/fail-fast
    discipline (e.g. `harness.item_bank.write_item_bank_once`)."""
    target = "r2://twin-checkpoints/fragments.jsonl"
    with pytest.raises(ValueError, match="SELF_REPORT"):
        write_fragments_jsonl(
            [_fragment("behavioral", source_class=SourceClass.BEHAVIOR), _fragment("self-report content")], target
        )


def test_allows_non_self_report_fragments_on_a_non_file_uri() -> None:
    """The guard is scoped to SELF_REPORT — ordinary behavioral fragments
    (e.g. LINE messages) are still expected to sync cross-cloud per
    `twin/CLAUDE.md`'s "Cross-cloud hub" design; this test pins that the new
    guard doesn't overreach into blocking that. Uses fsspec's built-in
    in-memory filesystem (`memory://`, no network/credentials needed) so a
    genuinely non-`file://` scheme is exercised for real, not just implied."""
    written = write_fragments_jsonl(
        [_fragment("behavioral", source_class=SourceClass.BEHAVIOR)], "memory://fragments.jsonl"
    )
    assert written == 1
