"""train.reflection is deliberately thin. SPEC.md §5.4 — pins the deferral
itself, not any real behavior."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split
from twin.core.trajectory import ActionStep, Exposure, Trajectory
from twin.train.reflection import maybe_insert_reflection


def test_maybe_insert_reflection_is_not_yet_implemented() -> None:
    trajectory = Trajectory(
        principal_id="p1",
        context_time=datetime(2026, 1, 1, tzinfo=UTC),
        split=Split.TRAIN,
        exposure=Exposure(occurred=True, stimulus="x", evidence=ExposureEvidence.HISTORY),
        observation="x",
        available_tools=["recall"],
        steps=[ActionStep(surface="line", content="hi")],
        negative_class=NegativeClass.NONE,
        ground_truth_source=GroundTruthSource.OBSERVED,
    )
    with pytest.raises(NotImplementedError, match=r"§5\.4"):
        maybe_insert_reflection(trajectory, ratio=0.2, teacher=object())  # type: ignore[arg-type]
