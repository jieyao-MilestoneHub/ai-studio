"""Harness data types. eval-harness skill; SPEC.md §4.10/D25."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split
from twin.core.trajectory import Exposure, NoActionStep, Trajectory
from twin.harness.schema import (
    HarnessError,
    RawEvalSample,
    S1Answer,
    StrippedSample,
    TaskVerifier,
    reject_synthesized_for_eval,
)


def test_task_verifier_is_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Phase 11"):
        TaskVerifier().verify("task-1")


def test_stripped_sample_has_no_source_label_field() -> None:
    """data-contract skill rule 3, applied via a type-level guarantee — same
    move as Fragment.split's frozen-ness. If this ever gains a source_label
    field, the "physically stripped" property becomes only conventional."""
    assert "source_label" not in StrippedSample.model_fields
    assert "source_label" in RawEvalSample.model_fields


def _trajectory(ground_truth_source: GroundTruthSource) -> Trajectory:
    return Trajectory(
        principal_id="p1",
        context_time=datetime(2026, 1, 1, tzinfo=UTC),
        split=Split.HELDOUT,
        exposure=Exposure(occurred=True, stimulus="x", evidence=ExposureEvidence.HISTORY),
        observation="x",
        available_tools=["recall"],
        steps=[NoActionStep(reason="test")],
        negative_class=NegativeClass.NONE,
        ground_truth_source=ground_truth_source,
    )


def test_reject_synthesized_for_eval_passes_with_only_observed_data() -> None:
    reject_synthesized_for_eval([_trajectory(GroundTruthSource.OBSERVED)])  # must not raise


def test_reject_synthesized_for_eval_raises_on_synthesized_data() -> None:
    with pytest.raises(HarnessError, match="D25"):
        reject_synthesized_for_eval(
            [_trajectory(GroundTruthSource.OBSERVED), _trajectory(GroundTruthSource.TEACHER_SYNTHESIZED)]
        )


def test_s1_answer_round_trips_through_json() -> None:
    answer = S1Answer(item_id="a" * 16, wave=1, answer="option text", answered_at=datetime(2026, 8, 28, tzinfo=UTC))
    assert S1Answer.model_validate_json(answer.model_dump_json()) == answer


def test_s1_answer_is_frozen() -> None:
    answer = S1Answer(item_id="a" * 16, wave=1, answer="x", answered_at=datetime(2026, 8, 28, tzinfo=UTC))
    with pytest.raises(ValidationError):
        answer.answer = "y"  # type: ignore[misc]


def test_s1_answer_wave_must_be_1_or_2() -> None:
    with pytest.raises(ValidationError):
        S1Answer(item_id="a" * 16, wave=3, answer="x", answered_at=datetime(2026, 8, 28, tzinfo=UTC))  # type: ignore[arg-type]
