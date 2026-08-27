"""Closed vocabularies for the Fragment and Trajectory schemas, and for the two
gate mechanisms downstream of them. SPEC.md §4.1, §4.4, §2.5, §4.10-4.11, §6.5."""

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


class NegativeClass(StrEnum):
    """SPEC.md §4.10/§4.11. `hard` MUST carry real exposure evidence; a
    `no_action` with `exposure.evidence == absent` MUST be `trivial`, never
    `hard` — this is the project's named failure mode #1 if gotten wrong."""

    NONE = "none"
    HARD = "hard"
    TRIVIAL = "trivial"


class GroundTruthSource(StrEnum):
    """SPEC.md §4.10/D25. `teacher_synthesized` MUST NOT reach any EVAL.md
    suite — training may use it, evaluation MUST NOT (harness enforces this,
    not this enum; see harness.schema.reject_synthesized_for_eval)."""

    OBSERVED = "observed"
    PRINCIPAL_ANNOTATED = "principal_annotated"
    TEACHER_SYNTHESIZED = "teacher_synthesized"


class ExposureEvidence(StrEnum):
    """SPEC.md §4.3, §4.10. `absent` is the value that forces
    `negative_class=trivial` (§4.11) — without it, "chose not to act" and
    "never saw it" are indistinguishable."""

    READ_RECEIPT = "read_receipt"
    HISTORY = "history"
    INFERRED = "inferred"
    ABSENT = "absent"


class GateLevel(StrEnum):
    """SPEC.md §6.5, D29-D31. Shared by harness.report.EvalReport.gate_level
    (the fitness gate's record of where a twin stands) and agent.gate's own
    send-gate state — same vocabulary, two different consumers."""

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
