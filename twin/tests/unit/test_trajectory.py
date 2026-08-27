"""Trajectory schema. SPEC.md §4.10, §4.11/D6/D7/D20."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split
from twin.core.trajectory import (
    ActionStep,
    Exposure,
    NoActionStep,
    ReflectionStep,
    ToolCallStep,
    Trajectory,
)


def _make_trajectory(**overrides: object) -> Trajectory:
    defaults: dict[str, object] = dict(
        principal_id="p1",
        context_time=datetime(2026, 3, 14, 9, 12, tzinfo=UTC),
        split=Split.HELDOUT,
        exposure=Exposure(occurred=True, stimulus="saw a news article", evidence=ExposureEvidence.HISTORY),
        observation="a news article about X appeared",
        available_tools=["recall", "web_search", "reply"],
        steps=[ActionStep(surface="line", content="reacted to it")],
        negative_class=NegativeClass.NONE,
        ground_truth_source=GroundTruthSource.OBSERVED,
    )
    defaults.update(overrides)
    return Trajectory(**defaults)  # type: ignore[arg-type]


def test_trajectory_id_is_generated_and_unique() -> None:
    a, b = _make_trajectory(), _make_trajectory()
    assert a.trajectory_id and b.trajectory_id
    assert a.trajectory_id != b.trajectory_id


def test_trajectory_is_frozen() -> None:
    trajectory = _make_trajectory()
    with pytest.raises(ValidationError):
        trajectory.split = Split.TRAIN  # type: ignore[misc]


@pytest.mark.parametrize(
    "step",
    [
        ToolCallStep(tool="recall", args={"query": "x"}, result_digest="abc"),
        ReflectionStep(content="thinking about whether to reply"),
        ActionStep(surface="line", content="ok!"),
        NoActionStep(reason="not interesting enough to respond to"),
    ],
)
def test_all_step_types_construct(step: object) -> None:
    trajectory = _make_trajectory(steps=[step])
    assert trajectory.steps == [step]


class TestAbsentEvidenceForcesTrivial:
    """SPEC.md §4.11/D20 — the project's named failure mode #1 if this breaks."""

    def test_absent_evidence_with_no_action_and_trivial_is_fine(self) -> None:
        trajectory = _make_trajectory(
            exposure=Exposure(occurred=False, stimulus="", evidence=ExposureEvidence.ABSENT),
            steps=[NoActionStep(reason="no record of exposure")],
            negative_class=NegativeClass.TRIVIAL,
        )
        assert trajectory.negative_class == NegativeClass.TRIVIAL

    def test_absent_evidence_with_no_action_and_hard_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="MUST have negative_class='trivial'"):
            _make_trajectory(
                exposure=Exposure(occurred=False, stimulus="", evidence=ExposureEvidence.ABSENT),
                steps=[NoActionStep(reason="no record of exposure")],
                negative_class=NegativeClass.HARD,
            )

    def test_present_evidence_with_no_action_and_hard_is_fine(self) -> None:
        trajectory = _make_trajectory(
            exposure=Exposure(occurred=True, stimulus="read the message", evidence=ExposureEvidence.READ_RECEIPT),
            steps=[NoActionStep(reason="chose not to respond")],
            negative_class=NegativeClass.HARD,
        )
        assert trajectory.negative_class == NegativeClass.HARD

    def test_absent_evidence_without_no_action_step_is_unaffected(self) -> None:
        trajectory = _make_trajectory(
            exposure=Exposure(occurred=False, stimulus="", evidence=ExposureEvidence.ABSENT),
            steps=[ActionStep(surface="line", content="replied anyway")],
            negative_class=NegativeClass.NONE,
        )
        assert trajectory.negative_class == NegativeClass.NONE


class TestNegativeClassRequiresANoActionStep:
    """SPEC.md §2.3: hard and trivial negatives are *defined* as 曝光→不動作
    trajectories — data-hygiene review of the interface-first pass flagged
    that this direction wasn't checked yet (only absent-evidence-forces-
    trivial was)."""

    @pytest.mark.parametrize("negative_class", [NegativeClass.HARD, NegativeClass.TRIVIAL])
    def test_rejects_hard_or_trivial_with_only_an_action_step(self, negative_class: NegativeClass) -> None:
        with pytest.raises(ValidationError, match="MUST have a NoActionStep"):
            _make_trajectory(
                steps=[ActionStep(surface="line", content="replied")],
                negative_class=negative_class,
            )

    def test_hard_with_a_no_action_step_is_fine(self) -> None:
        trajectory = _make_trajectory(
            exposure=Exposure(occurred=True, stimulus="x", evidence=ExposureEvidence.READ_RECEIPT),
            steps=[NoActionStep(reason="chose not to respond")],
            negative_class=NegativeClass.HARD,
        )
        assert trajectory.negative_class == NegativeClass.HARD

    def test_none_with_only_an_action_step_is_fine(self) -> None:
        trajectory = _make_trajectory(
            steps=[ActionStep(surface="line", content="replied")],
            negative_class=NegativeClass.NONE,
        )
        assert trajectory.negative_class == NegativeClass.NONE
