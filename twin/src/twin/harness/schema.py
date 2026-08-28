"""Eval-harness data types. EVAL.md §3-§11; the eval-harness skill's
directory/isolation contract. All script-only per that skill — nothing here
is ever delegated to an LLM's judgment about shape or content.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from twin.core.enums import GroundTruthSource
from twin.core.trajectory import Trajectory

Suite = Literal["s1", "s2", "s3", "s4"]


class HarnessError(Exception):
    """The harness refused to proceed — an admissibility rule was violated."""


class RawEvalSample(BaseModel):
    """Before shard-splitting strips the source label. eval-harness skill
    step 2: MUST be physically stripped in the splitting script, MUST NOT
    rely on a prompt telling the judge to ignore it."""

    model_config = ConfigDict(frozen=True)

    sample_id: str  # content-hash derived — never a sequence number
    source_label: Literal["principal", "twin"]
    content: str
    suite: Suite


class StrippedSample(BaseModel):
    """No `source_label` field at all — "physically stripped" is a type-level
    guarantee here, the same move `Fragment.split` being frozen makes for
    data-contract skill rule 3."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    content: str
    suite: Suite


class JudgedItem(BaseModel):
    """`verdict` is rubric-defined vocabulary, never a score or percentage —
    EVAL.md §6.2 point 4: scores are computed by a script, never given by
    the judge."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    verdict: str
    rationale: str


class S1Item(BaseModel):
    """EVAL.md §3.2's four situational-question categories, verbatim."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    item_type: Literal["value_tradeoff", "preference", "reaction_tendency", "recall"]
    prompt: str
    options: list[str] | None
    source_fragment_ids: list[str]


class S1Answer(BaseModel):
    """EVAL.md §3.2 Wave 1/Wave 2 (R1/R2): one recorded answer to a frozen
    `S1Item`. `answer` MUST be verbatim one of that item's own `options` —
    never free text — so Phase 6's `agreement(R1, R2)` is an exact-match
    comparison, not fuzzy string matching."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    wave: Literal[1, 2]
    answer: str
    answered_at: datetime


class VerifiedTaskResult(BaseModel):
    """EVAL.md §4.2: S2a completion MUST be programmatically verified, MUST
    NOT be judged by an LLM. This type is the structural boundary between
    that path and the judge path — they must never overlap."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    passed: bool
    detail: str


class TaskVerifier:
    """Protocol-shaped, but task-specific verifiers are inherently Phase-11
    work (they need real held-out tasks) — this class only documents the
    shape a verifier MUST have; there is no generic instance to construct."""

    def verify(self, task_id: str) -> VerifiedTaskResult:
        raise NotImplementedError("task-specific verifiers are built per held-out task in Phase 11")


class S1Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    normalized_accuracy: dict[Literal["T", "B0", "B1", "B2"], float]
    self_consistency: float


class S2Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    pass_1: float
    pass_4: float
    tool_top1: float
    call_count_mae: float
    giveup_delta: float
    plugin_transfer: float = Field(description="EVAL.md §4.4 style-consistency score, 0..1")


class S3Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision: float
    recall: float
    false_alarm: float
    f1: float
    silence_rate_delta: float


class S4Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    identification_rate: float
    reason_distribution: dict[str, int]
    code_switch_rate_delta: float


def reject_synthesized_for_eval(trajectories: Iterable[Trajectory]) -> None:
    """SPEC.md §4.10/D25: "ground_truth_source = teacher_synthesized 的軌跡
    MUST NOT 用於 EVAL.md 的任何 suite." A usage-context rule, not a
    single-record invariant — `Trajectory`'s own validator can't enforce this
    (train is allowed to use synthesized trajectories; only eval is
    forbidden), so it lives here, checked at eval-set assembly time."""
    offenders = [
        t.trajectory_id for t in trajectories if t.ground_truth_source == GroundTruthSource.TEACHER_SYNTHESIZED
    ]
    if offenders:
        raise HarnessError(
            f"{len(offenders)} teacher_synthesized trajectory(ies) found in an "
            f"eval-bound set (D25 violation): {offenders[:5]}"
            + (f" (+{len(offenders) - 5} more)" if len(offenders) > 5 else "")
        )


__all__ = [
    "HarnessError",
    "JudgedItem",
    "RawEvalSample",
    "S1Answer",
    "S1Item",
    "S1Metrics",
    "S2Metrics",
    "S3Metrics",
    "S4Metrics",
    "StrippedSample",
    "Suite",
    "TaskVerifier",
    "VerifiedTaskResult",
    "reject_synthesized_for_eval",
]
