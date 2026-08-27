"""Trajectory — a decision unit: observation -> action or no-action. SPEC.md
§4.10, the single normative schema, exactly as `core.fragment.Fragment` is for
fragments — same rules apply: this is the only constructor (data-contract
skill rule 1), and it is frozen so `split`/`negative_class`/`ground_truth_source`
cannot be silently rewritten downstream (rule 3).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split


class ToolCallStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: dict[str, Any]
    result_digest: str


class ReflectionStep(BaseModel):
    """SPEC.md §5.4 — a first-person, not-for-inference-output motive statement.
    MUST NOT be surfaced at inference time except on explicit introspection."""

    model_config = ConfigDict(frozen=True)

    type: Literal["reflection"] = "reflection"
    content: str


class ActionStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["action"] = "action"
    surface: str
    content: str


class NoActionStep(BaseModel):
    """SPEC.md §4.10: no_action MUST be an explicit step type, never data
    absence (D6) — the project's #1 named failure mode is training a twin
    that never learns to stay quiet, because "quiet" was never a sample."""

    model_config = ConfigDict(frozen=True)

    type: Literal["no_action"] = "no_action"
    reason: str


Step = Annotated[
    ToolCallStep | ReflectionStep | ActionStep | NoActionStep,
    Field(discriminator="type"),
]


class Exposure(BaseModel):
    """SPEC.md §4.3/§4.10. The precondition that makes a no_action classifiable
    as hard vs. trivial — see Trajectory's own validator below."""

    model_config = ConfigDict(frozen=True)

    occurred: bool
    stimulus: str
    evidence: ExposureEvidence


class Trajectory(BaseModel):
    """SPEC.md §4.10. `ground_truth_source == teacher_synthesized` MUST NOT be
    used in any EVAL.md suite (D25) — that is a usage-context rule, not a
    single-record invariant, so it is enforced in harness.schema at assembly
    time, not here."""

    model_config = ConfigDict(frozen=True)

    trajectory_id: str = Field(default_factory=lambda: uuid4().hex)
    principal_id: str
    context_time: datetime
    split: Split
    exposure: Exposure
    observation: str
    available_tools: list[str]
    steps: list[Step]
    negative_class: NegativeClass
    ground_truth_source: GroundTruthSource

    @model_validator(mode="after")
    def _negative_class_is_consistent_with_steps(self) -> Trajectory:
        """SPEC.md §2.3 defines both negative kinds *as* "曝光 → 不動作"
        trajectories — hard negative: "一筆 曝光 → 不動作 的軌跡..."; trivial is
        the same shape without real exposure evidence. That definition itself
        requires a `NoActionStep` to be present; a `negative_class` of `hard`
        or `trivial` on a trajectory with no no_action step at all would
        mislabel an action trajectory as a negative — inflating the §4.11
        "hard SHOULD be ≥ half of negatives" corpus statistic with a record
        that was never actually a negative example.

        Separately, §4.11/D6/D7/D20: exposure.evidence == absent on a
        no_action trajectory MUST be negative_class == trivial, never hard —
        without a real exposure record there is no way to tell "chose not to
        act" from "never saw it".
        """
        has_no_action = any(isinstance(step, NoActionStep) for step in self.steps)

        if self.negative_class in (NegativeClass.HARD, NegativeClass.TRIVIAL) and not has_no_action:
            raise ValueError(
                f"SPEC.md §2.3: negative_class={self.negative_class!r} MUST have a "
                f"NoActionStep present — both hard and trivial negatives are "
                f"*defined* as 曝光→不動作 trajectories, and this one has no "
                f"no_action step at all"
            )

        if (
            has_no_action
            and self.exposure.evidence == ExposureEvidence.ABSENT
            and self.negative_class != NegativeClass.TRIVIAL
        ):
            raise ValueError(
                "SPEC.md §4.11/D20: a no_action trajectory with "
                "exposure.evidence='absent' MUST have negative_class='trivial' "
                f"(got {self.negative_class!r}) — without exposure evidence this "
                "cannot be distinguished from 'never saw it', so it MUST NOT be "
                "counted as a hard negative"
            )
        return self
