"""The daily pod-open ledger: the backstop the monthly budget guard cannot
provide against a crash-looping worker opening a fresh pod every restart."""

from __future__ import annotations

import json
from pathlib import Path

from ai_studio.runtime.opens import RETAIN_S, PodOpenLedger


def test_counts_opens_since_a_timestamp(tmp_path: Path) -> None:
    ledger = PodOpenLedger(tmp_path / ".pod_opens.json")
    assert ledger.count_since(0.0) == 0

    ledger.record("pod-1", when=1_000.0)
    ledger.record("pod-2", when=2_000.0)

    assert ledger.count_since(0.0) == 2
    assert ledger.count_since(1_500.0) == 1
    assert ledger.count_since(3_000.0) == 0


def test_old_entries_are_pruned_on_write(tmp_path: Path) -> None:
    ledger = PodOpenLedger(tmp_path / ".pod_opens.json")
    ledger.record("stale", when=10.0)
    ledger.record("fresh", when=10.0 + RETAIN_S + 1)

    data = json.loads((tmp_path / ".pod_opens.json").read_text(encoding="utf-8"))
    assert [o["pod_id"] for o in data["opens"]] == ["fresh"]


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert PodOpenLedger(tmp_path / "nope.json").count_since(0.0) == 0
