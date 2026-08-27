"""Naive recall. SPEC.md §3.1/C4."""

from __future__ import annotations

from datetime import UTC, datetime

from twin.core.enums import Modality, SourceClass, Split
from twin.core.fragment import EventTime, Fragment
from twin.memory.retrieve import retrieve


def _fragment(content: str, event_time_value: str) -> Fragment:
    return Fragment(
        principal_id="p1",
        source_class=SourceClass.SELF_REPORT,
        modality=Modality.TEXT,
        content=content,
        event_time=EventTime(value=event_time_value, precision="day", confidence=0.9),
        ingest_time=datetime(2026, 8, 27, tzinfo=UTC),
        split=Split.TRAIN,
    )


FRAGMENTS = [
    _fragment("went hiking with Alice", "2026-01-01"),
    _fragment("had coffee with Bob", "2026-03-15"),
    _fragment("went hiking again, this time alone", "2026-06-20"),
]


def test_retrieve_matches_substring_case_insensitively() -> None:
    results = retrieve("HIKING", FRAGMENTS)
    assert len(results) == 2
    assert all("hiking" in f.content for f in results)


def test_retrieve_returns_no_matches_as_empty_list() -> None:
    assert retrieve("skydiving", FRAGMENTS) == []


def test_retrieve_orders_most_recent_first() -> None:
    results = retrieve("hiking", FRAGMENTS)
    assert results[0].content == "went hiking again, this time alone"
    assert results[1].content == "went hiking with Alice"


def test_retrieve_respects_limit() -> None:
    results = retrieve("", FRAGMENTS, limit=1)
    assert len(results) == 1


def test_retrieve_narrows_by_time_hint() -> None:
    results = retrieve("hiking", FRAGMENTS, time_hint="2026-06")
    assert len(results) == 1
    assert results[0].content == "went hiking again, this time alone"
