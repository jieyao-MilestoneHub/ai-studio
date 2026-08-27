"""Time-based split decision. SPEC.md §4.8/D21 — the failure mode this must
catch ("靜默污染, S1/S3 全面失真") is silent, so this coverage is what SPEC.md
§4.8 means by "此過濾 MUST 有測試覆蓋": if this test file is ever skipped or
deleted, nothing else in the codebase would notice the split logic breaking.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from twin.core.enums import Split
from twin.ingest.split import decide_split, sealed_cutoff_for

TRAIN_CUTOFF = datetime(2026, 1, 1, tzinfo=UTC)
SEALED_CUTOFF = datetime(2026, 7, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("event_time", "expected"),
    [
        (datetime(2025, 12, 31, tzinfo=UTC), Split.TRAIN),
        (datetime(2026, 1, 1, tzinfo=UTC), Split.HELDOUT),  # boundary is inclusive of heldout
        (datetime(2026, 6, 30, tzinfo=UTC), Split.HELDOUT),
        (datetime(2026, 7, 1, tzinfo=UTC), Split.SEALED),  # boundary is inclusive of sealed
        (datetime(2026, 8, 1, tzinfo=UTC), Split.SEALED),
    ],
)
def test_decide_split_boundaries(event_time: datetime, expected: Split) -> None:
    assert (
        decide_split(event_time, train_cutoff=TRAIN_CUTOFF, sealed_cutoff=SEALED_CUTOFF) == expected
    )


def test_decide_split_rejects_inverted_cutoffs() -> None:
    with pytest.raises(ValueError, match="before train_cutoff"):
        decide_split(
            datetime(2026, 3, 1, tzinfo=UTC),
            train_cutoff=SEALED_CUTOFF,
            sealed_cutoff=TRAIN_CUTOFF,
        )


def test_decide_split_rejects_mismatched_awareness() -> None:
    """Naive vs aware datetimes MUST raise rather than silently compare wrong —
    this is Python's own comparison semantics, exercised here to pin the
    behaviour ingest.sources.line's naive-local timestamps rely on."""
    with pytest.raises(TypeError):
        decide_split(datetime(2026, 3, 1), train_cutoff=TRAIN_CUTOFF, sealed_cutoff=SEALED_CUTOFF)


def test_sealed_cutoff_for_is_the_most_recent_slice() -> None:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    train_cutoff = now - timedelta(days=100)
    cutoff = sealed_cutoff_for(train_cutoff=train_cutoff, now=now, sealed_fraction=0.2)
    assert cutoff == now - timedelta(days=20)


def test_sealed_cutoff_for_rejects_bad_fraction() -> None:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    with pytest.raises(ValueError, match="sealed_fraction"):
        sealed_cutoff_for(train_cutoff=now - timedelta(days=10), now=now, sealed_fraction=1.0)


def test_sealed_cutoff_for_rejects_now_before_train_cutoff() -> None:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    with pytest.raises(ValueError, match="before train_cutoff"):
        sealed_cutoff_for(train_cutoff=now + timedelta(days=1), now=now)


def test_sealed_cutoff_end_to_end_with_decide_split() -> None:
    """The computed sealed_cutoff actually produces ~20% sealed of the heldout
    window when fed back through decide_split — the two functions must agree."""
    now = datetime(2026, 8, 27, tzinfo=UTC)
    train_cutoff = now - timedelta(days=100)
    sealed_cutoff = sealed_cutoff_for(train_cutoff=train_cutoff, now=now, sealed_fraction=0.2)

    just_inside_heldout = sealed_cutoff - timedelta(seconds=1)
    assert decide_split(just_inside_heldout, train_cutoff=train_cutoff, sealed_cutoff=sealed_cutoff) == Split.HELDOUT
    assert decide_split(sealed_cutoff, train_cutoff=train_cutoff, sealed_cutoff=sealed_cutoff) == Split.SEALED
