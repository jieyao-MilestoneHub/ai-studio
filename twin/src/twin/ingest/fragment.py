"""Assemble raw source records into `Fragment`s. SPEC.md §4.4, §4.8/D21.

The one place `ingest.split.decide_split` is actually called for text sources —
every source module (ingest.sources.*) hands its raw records here rather than
deciding split itself, so there is exactly one call site to audit.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

from twin.core.enums import Modality, Precision, SourceClass
from twin.core.fragment import EventTime, Fragment, ThirdPartySpan
from twin.ingest.sources.line import LineMessage, parse_line_export
from twin.ingest.split import decide_split


def _format_event_time_value(event_time: datetime, precision: Precision) -> str:
    """SPEC.md §4.4: the stored value MUST reflect its stated precision, not a
    fuller timestamp truncated by coincidence — a caller passing precision=YEAR
    but getting back a minute-level string would be exactly the "虛假的精確度"
    the spec warns against (data-contract skill rule 4)."""
    if precision is Precision.YEAR:
        return f"{event_time.year:04d}"
    if precision is Precision.MONTH:
        return f"{event_time.year:04d}-{event_time.month:02d}"
    if precision is Precision.DAY:
        return event_time.date().isoformat()
    if precision is Precision.HOUR:
        return event_time.isoformat(timespec="hours")
    return event_time.isoformat(timespec="minutes")


def fragment_from_text_record(
    *,
    principal_id: str,
    content: str,
    event_time: datetime,
    precision: Precision,
    confidence: float,
    source_class: SourceClass,
    modality: Modality,
    train_cutoff: datetime,
    sealed_cutoff: datetime,
    ingest_time: datetime | None = None,
    third_party_spans: list[ThirdPartySpan] | None = None,
) -> Fragment:
    split = decide_split(event_time, train_cutoff=train_cutoff, sealed_cutoff=sealed_cutoff)
    return Fragment(
        principal_id=principal_id,
        source_class=source_class,
        modality=modality,
        content=content,
        event_time=EventTime(
            value=_format_event_time_value(event_time, precision),
            precision=precision,
            confidence=confidence,
        ),
        ingest_time=ingest_time or datetime.now(UTC),
        split=split,
        third_party_spans=third_party_spans or [],
    )


def fragments_from_line_export(
    text: str,
    *,
    principal_id: str,
    principal_display_name: str,
    train_cutoff: datetime,
    sealed_cutoff: datetime,
    ingest_time: datetime | None = None,
) -> Iterator[Fragment]:
    """SPEC.md §6.4/D27 decided LINE as v1's Surface — its own chat export is
    the natural first real, plain-text source for Phase 1 (PLAN.md §2).

    `principal_display_name` is required, not defaulted: SPEC.md §4.9/§8
    guardrail 1 MUST tag third-party spans at ingest ("成本已付，且它是日後任何
    政策的前提") — every message from anyone other than the principal is, in
    full, third-party content. Without knowing the principal's own name as it
    appears in *this* export, that tagging cannot happen at all, so there is no
    safe default to fall back to. (What this does NOT cover: the principal's
    own messages *mentioning* a third party — that's entity extraction,
    ingest.entities, a later phase's job, not this one's.)
    """
    ingest_time = ingest_time or datetime.now(UTC)
    message: LineMessage
    for message in parse_line_export(text):
        content = f"{message.sender}: {message.content}"
        is_third_party = message.sender != principal_display_name
        spans = (
            [ThirdPartySpan(start=0, end=len(content), party_ref=message.sender)]
            if is_third_party
            else []
        )
        yield fragment_from_text_record(
            principal_id=principal_id,
            content=content,
            event_time=message.sent_at,
            precision=Precision.MINUTE,
            confidence=1.0,
            source_class=SourceClass.BEHAVIOR,
            modality=Modality.MESSAGE,
            train_cutoff=train_cutoff,
            sealed_cutoff=sealed_cutoff,
            ingest_time=ingest_time,
            third_party_spans=spans,
        )
