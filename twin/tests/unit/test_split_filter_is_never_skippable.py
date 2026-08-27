"""PLAN.md §3.5: the split-filter test MUST NOT be skippable — this makes
that a mechanism, not a comment reminding someone not to add `@pytest.mark.skip`."""

from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "test_train_data_split_filter.py"

FORBIDDEN_MARKERS = ("pytest.mark.skip", "pytest.mark.xfail", "@pytest.mark.skipif")


def test_split_filter_test_file_exists() -> None:
    assert TARGET.is_file()


def test_split_filter_test_file_has_no_skip_or_xfail_markers() -> None:
    text = TARGET.read_text(encoding="utf-8")
    hits = [marker for marker in FORBIDDEN_MARKERS if marker in text]
    assert not hits, f"test_train_data_split_filter.py MUST NOT be skippable, found: {hits}"
