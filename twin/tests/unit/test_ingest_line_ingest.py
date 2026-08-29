"""twin.ingest.line_ingest — the Phase 1 end-to-end driver. Fictional names
and content only (SPEC.md §8 guardrail 2)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from twin.core.enums import Split
from twin.ingest.line_ingest import (
    IngestRefused,
    LineExportSource,
    ingest_line_export,
    ingest_line_exports,
)
from twin.ingest.store import read_fragments_jsonl

EXPORT = """2026.01.10 星期六
09:00 Alice Chen 早安
09:05 Bob 早
2026.03.10 星期二
10:00 Alice Chen 今天要不要去爬山
10:01 Bob 好啊
2026.05.20 星期三
11:00 Bob 最近好忙
11:02 Alice Chen 辛苦了
"""
KNOWN = ["Alice Chen", "Bob"]
TRAIN_CUTOFF = datetime(2026, 2, 1)
NOW = datetime(2026, 6, 1)  # heldout window Feb..Jun; last 20% (from ~May 8) is sealed


def _run(tmp_path: Path, **kw: object) -> tuple[str, object]:
    uri = f"file://{tmp_path}/data/fragments.jsonl"
    args: dict[str, object] = dict(
        uri=uri, principal_id="p", principal_display_name="Alice Chen", known_senders=KNOWN,
        train_cutoff=TRAIN_CUTOFF, now=NOW,
    )
    args.update(kw)
    return uri, ingest_line_export(EXPORT, **args)  # type: ignore[arg-type]


def test_writes_store_and_splits_by_cutoffs(tmp_path: Path) -> None:
    uri, summary = _run(tmp_path)
    stored = list(read_fragments_jsonl(uri))
    assert len(stored) == 6 == summary.total  # type: ignore[attr-defined]
    by_split = {s: sum(1 for f in stored if f.split == s) for s in Split}
    assert by_split == {Split.TRAIN: 2, Split.HELDOUT: 2, Split.SEALED: 2}
    assert summary.third_party_tagged == 3  # type: ignore[attr-defined]
    assert all(f.third_party_spans for f in stored if f.content.startswith("Bob:"))
    assert not any(f.third_party_spans for f in stored if f.content.startswith("Alice Chen:"))


def test_refuses_to_overwrite_existing_store(tmp_path: Path) -> None:
    uri, _ = _run(tmp_path)
    before = Path(uri.removeprefix("file://")).read_bytes()
    with pytest.raises(IngestRefused, match="already exists"):
        _run(tmp_path)
    assert Path(uri.removeprefix("file://")).read_bytes() == before
    _run(tmp_path, overwrite=True)


def test_refuses_when_no_heldout_fragment(tmp_path: Path) -> None:
    with pytest.raises(IngestRefused, match="0 held-out"):
        _run(tmp_path, train_cutoff=datetime(2026, 5, 25), now=datetime(2026, 5, 26))
    assert not (tmp_path / "data" / "fragments.jsonl").exists()


def test_refuses_principal_not_in_known_senders(tmp_path: Path) -> None:
    with pytest.raises(IngestRefused, match="known_senders"):
        _run(tmp_path, principal_display_name="Carol")


def test_render_reports_all_three_splits(tmp_path: Path) -> None:
    _, summary = _run(tmp_path)
    text = summary.render()  # type: ignore[attr-defined]
    for word in ("train", "heldout", "sealed", "third_party_spans", "event_time missing: 0"):
        assert word in text


def test_refuses_when_sealed_window_is_empty(tmp_path: Path) -> None:
    # now far past the export: the last 20% of Feb..2027 holds no message
    with pytest.raises(IngestRefused, match="0 sealed"):
        _run(tmp_path, now=datetime(2027, 6, 1))
    assert not (tmp_path / "data" / "fragments.jsonl").exists()


EXPORT_2 = """2026.04.01 星期三
08:00 Carol 嗨
08:01 Alice Chen 嗨
"""


def test_multiple_rooms_merge_into_one_store_sorted_by_time(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/data/fragments.jsonl"
    summary = ingest_line_exports(
        [
            LineExportSource(text=EXPORT, known_senders=KNOWN, label="a"),
            LineExportSource(text=EXPORT_2, known_senders=["Alice Chen", "Carol"], label="b"),
        ],
        uri=uri, principal_id="p", principal_display_name="Alice Chen",
        train_cutoff=TRAIN_CUTOFF, now=NOW,
    )
    stored = list(read_fragments_jsonl(uri))
    assert summary.total == len(stored) == 8
    times = [f.event_time.value for f in stored]
    assert times == sorted(times)
    assert sum(1 for f in stored if f.split == Split.HELDOUT) == 4
    assert sum(1 for f in stored if f.content.startswith("Carol:") and f.third_party_spans) == 1


def test_multiple_rooms_refuse_when_one_omits_principal(tmp_path: Path) -> None:
    with pytest.raises(IngestRefused, match="known_senders"):
        ingest_line_exports(
            [LineExportSource(text=EXPORT_2, known_senders=["Carol"], label="b")],
            uri=f"file://{tmp_path}/x.jsonl", principal_id="p", principal_display_name="Alice Chen",
            train_cutoff=TRAIN_CUTOFF, now=NOW,
        )
