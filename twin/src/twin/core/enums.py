"""Closed vocabularies for the Fragment schema. SPEC.md §4.1, §4.4, §2.5."""

from __future__ import annotations

from enum import StrEnum


class SourceClass(StrEnum):
    """Where a fragment's content came from. SPEC.md §4.1 — determines where it goes:
    self_report/behavior/exposure feed the memory store or trajectory set;
    knowledge MUST NOT enter LoRA training (§4.1) and has no event_time (§4.4)."""

    SELF_REPORT = "self_report"
    BEHAVIOR = "behavior"
    EXPOSURE = "exposure"
    KNOWLEDGE = "knowledge"


class Modality(StrEnum):
    """SPEC.md §4.4. All non-text modalities are reduced to text at ingest (§4.2) —
    this records what the content originally was, not what `content` is encoded as."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    DOC = "doc"
    MESSAGE = "message"
    AUDIO = "audio"


class Split(StrEnum):
    """SPEC.md §2.5, §4.8/D21. Written once at ingest; MUST NOT be decided at train
    time — see ingest.split.decide_split, the only place this is computed."""

    TRAIN = "train"
    HELDOUT = "heldout"
    SEALED = "sealed"


class Precision(StrEnum):
    """SPEC.md §4.4 event_time.precision — MUST be stated explicitly rather than
    defaulted, because a false precision is less like the principal than an honest
    "roughly last summer" (EVAL.md §3.5 fuzziness-matching failure)."""

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
