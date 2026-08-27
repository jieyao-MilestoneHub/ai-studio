"""Fragment schema. SPEC.md §4.4 — every MUST field must actually be enforced,
not just documented, and third_party_spans/entities must default to empty
rather than requiring every caller to pass them explicitly (§4.9)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from twin.core.enums import Modality, Precision, SourceClass, Split
from twin.core.fragment import Entities, EventTime, Fragment, ThirdPartySpan


def _make_fragment(**overrides: object) -> Fragment:
    defaults: dict[str, object] = dict(
        principal_id="p1",
        source_class=SourceClass.SELF_REPORT,
        modality=Modality.TEXT,
        content="hello",
        event_time=EventTime(value="2024-06", precision=Precision.MONTH, confidence=0.8),
        ingest_time=datetime(2026, 8, 27, tzinfo=UTC),
        split=Split.TRAIN,
    )
    defaults.update(overrides)
    return Fragment(**defaults)  # type: ignore[arg-type]


def test_fragment_id_is_generated_and_unique() -> None:
    a, b = _make_fragment(), _make_fragment()
    assert a.fragment_id and b.fragment_id
    assert a.fragment_id != b.fragment_id


def test_third_party_spans_defaults_to_empty_list() -> None:
    fragment = _make_fragment()
    assert fragment.third_party_spans == []


def test_entities_default_to_empty_lists() -> None:
    fragment = _make_fragment()
    assert fragment.entities == Entities(people=[], places=[], topics=[])


@pytest.mark.parametrize(
    "missing_field",
    ["principal_id", "source_class", "modality", "content", "event_time", "ingest_time", "split"],
)
def test_must_fields_are_required(missing_field: str) -> None:
    kwargs = dict(
        principal_id="p1",
        source_class=SourceClass.SELF_REPORT,
        modality=Modality.TEXT,
        content="hello",
        event_time=EventTime(value="2024-06", precision=Precision.MONTH, confidence=0.8),
        ingest_time=datetime(2026, 8, 27, tzinfo=UTC),
        split=Split.TRAIN,
    )
    del kwargs[missing_field]
    with pytest.raises(ValidationError):
        Fragment(**kwargs)  # type: ignore[arg-type]


def test_event_time_precision_is_required() -> None:
    """SPEC.md §4.4: "precision MUST 顯式表示." — no silent default."""
    with pytest.raises(ValidationError):
        EventTime(value="2024-06", confidence=0.8)  # type: ignore[call-arg]


def test_fragment_is_frozen() -> None:
    """data-contract skill rule 3: split/event_time MUST NOT be rewritten
    downstream. Enforced at the language level, not just by convention."""
    fragment = _make_fragment()
    with pytest.raises(ValidationError):
        fragment.split = Split.HELDOUT  # type: ignore[misc]


def test_third_party_span_fields() -> None:
    span = ThirdPartySpan(start=0, end=5, party_ref="friend_a")
    fragment = _make_fragment(third_party_spans=[span])
    assert fragment.third_party_spans == [span]


def test_salience_defaults_to_zero_and_is_bounded() -> None:
    assert _make_fragment().salience == 0.0
    with pytest.raises(ValidationError):
        _make_fragment(salience=1.5)


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        EventTime(value="2024-06", precision=Precision.MONTH, confidence=1.5)
