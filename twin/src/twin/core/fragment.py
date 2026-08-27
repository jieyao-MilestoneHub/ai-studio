"""Fragment — the minimal memory unit. SPEC.md §4.4, the single normative schema.

This is the *only* constructor for a fragment record (data-contract skill rule 1):
nothing else in this codebase MUST build a fragment from a dict literal. `Fragment`
is frozen so `split`, `event_time`, and `third_party_spans` — decided once at
ingest per SPEC.md §4.8/D21 and §4.9/D23 — cannot be silently rewritten downstream
(data-contract skill rule 3). A later correction (e.g. tagging a newly discovered
conflict per §4.7) MUST go through `model_copy(update=...)`, which still requires
the caller to state what changed and produces a new, distinct object rather than a
silent in-place edit.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from twin.core.enums import Modality, Precision, SourceClass, Split


class EventTime(BaseModel):
    """SPEC.md §4.4. A structured triple, not an ISO string — data-contract skill
    rule 4: collapsing this to `value` alone silently drops `precision`, which is
    exactly what EVAL.md §3.5's fuzziness-matching check measures."""

    model_config = ConfigDict(frozen=True)

    value: str = Field(description='ISO-8601, may be partial precision, e.g. "2024-06"')
    precision: Precision
    confidence: float = Field(ge=0.0, le=1.0)


class Entities(BaseModel):
    """SPEC.md §4.4. Extraction itself lands in a later phase (ingest.entities,
    §4.9); Phase 1 only needs the field to exist with empty-list defaults."""

    model_config = ConfigDict(frozen=True)

    people: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class ThirdPartySpan(BaseModel):
    """SPEC.md §4.9/D23. Marks *that* a span is third-party content; disposition
    (keep/anonymize/remove) is a separate, re-runnable step — never decided here."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    party_ref: str


class Fragment(BaseModel):
    """SPEC.md §4.4. A fragment with no `event_time` MUST NOT enter the memory
    store (§4.4) — enforced here by `event_time` being required, not optional:
    per SPEC.md §2.2's own definition, a fragment *is* "content that carries an
    event_time"; content without one is knowledge data (§4.1) and never becomes
    a `Fragment` at all."""

    model_config = ConfigDict(frozen=True)

    fragment_id: str = Field(default_factory=lambda: uuid4().hex)
    principal_id: str
    source_class: SourceClass
    modality: Modality
    content: str
    event_time: EventTime
    ingest_time: datetime
    split: Split
    entities: Entities = Field(default_factory=Entities)
    third_party_spans: list[ThirdPartySpan] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    source_uri: str | None = None
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
