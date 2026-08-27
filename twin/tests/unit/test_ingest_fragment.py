"""ingest.fragment's source-agnostic assembly helper. SPEC.md §4.4 — the stored
`event_time.value` MUST actually reflect its stated `precision`, not a fuller
timestamp that happens to still parse. LINE (the only caller today) always
passes precision=MINUTE, so a bug here would stay invisible until a lower-
precision source (e.g. self-report) reuses this function — pinned directly
rather than relying on that future caller to notice.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from twin.core.enums import Modality, Precision, SourceClass
from twin.core.fragment import ThirdPartySpan
from twin.ingest.fragment import _format_event_time_value, fragment_from_text_record

EVENT_TIME = datetime(2024, 6, 15, 10, 23)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        (Precision.YEAR, "2024"),
        (Precision.MONTH, "2024-06"),
        (Precision.DAY, "2024-06-15"),
        (Precision.HOUR, "2024-06-15T10"),
        (Precision.MINUTE, "2024-06-15T10:23"),
    ],
)
def test_format_event_time_value_matches_stated_precision(precision: Precision, expected: str) -> None:
    assert _format_event_time_value(EVENT_TIME, precision) == expected


def _fragment(precision: Precision, **overrides: object):
    defaults = dict(
        principal_id="p1",
        content="hello",
        event_time=EVENT_TIME,
        precision=precision,
        confidence=0.8,
        source_class=SourceClass.SELF_REPORT,
        modality=Modality.TEXT,
        train_cutoff=datetime(2024, 1, 1),
        sealed_cutoff=datetime(2025, 1, 1),
    )
    defaults.update(overrides)
    return fragment_from_text_record(**defaults)  # type: ignore[arg-type]


def test_fragment_event_time_value_honours_low_precision() -> None:
    """A caller stating YEAR precision must not get back a minute-level string —
    that's exactly the "虛假的精確度" SPEC.md §4.4 warns against."""
    fragment = _fragment(Precision.YEAR)
    assert fragment.event_time.value == "2024"
    assert fragment.event_time.precision == Precision.YEAR


def test_third_party_spans_default_to_empty_when_not_passed() -> None:
    fragment = _fragment(Precision.DAY)
    assert fragment.third_party_spans == []


def test_third_party_spans_pass_through_when_given() -> None:
    span = ThirdPartySpan(start=0, end=5, party_ref="someone")
    fragment = _fragment(Precision.DAY, third_party_spans=[span])
    assert fragment.third_party_spans == [span]
