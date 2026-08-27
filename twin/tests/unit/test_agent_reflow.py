"""Veto-to-hard-negative reflow. SPEC.md §6.5, data-contract skill rule 5."""

from __future__ import annotations

from datetime import UTC, datetime

from twin.agent.reflow import reflow_veto
from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split
from twin.core.trajectory import NoActionStep


def test_reflow_veto_produces_a_hard_negative_no_action_trajectory() -> None:
    trajectory = reflow_veto(
        principal_id="p1",
        context_time=datetime(2026, 6, 1, tzinfo=UTC),
        observation="Bob asked a question in the group chat",
        available_tools=["recall", "reply"],
        train_cutoff=datetime(2026, 1, 1, tzinfo=UTC),
        sealed_cutoff=datetime(2027, 1, 1, tzinfo=UTC),
    )

    assert trajectory.negative_class == NegativeClass.HARD
    assert trajectory.ground_truth_source == GroundTruthSource.OBSERVED
    assert trajectory.exposure.occurred is True
    assert trajectory.exposure.evidence == ExposureEvidence.READ_RECEIPT
    assert len(trajectory.steps) == 1
    assert isinstance(trajectory.steps[0], NoActionStep)


def test_reflow_veto_uses_the_normal_split_decision_path() -> None:
    """The split MUST be decided by ingest.split.decide_split, not by agent
    itself — this confirms context_time before train_cutoff really lands in
    train, exactly as any other ingested record would."""
    trajectory = reflow_veto(
        principal_id="p1",
        context_time=datetime(2025, 1, 1, tzinfo=UTC),
        observation="old event",
        available_tools=["reply"],
        train_cutoff=datetime(2026, 1, 1, tzinfo=UTC),
        sealed_cutoff=datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert trajectory.split == Split.TRAIN
