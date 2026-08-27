"""The split filter train reads through. SPEC.md §4.8: "訓練管線 MUST 於載入時
硬性過濾 split != train 的樣本，且此過濾 MUST 有測試覆蓋" — PLAN.md §3.5 names this
test file and its discipline explicitly: it MUST NOT be skipped or xfail'd,
because it is the one thing standing between a data reshuffle and silent,
undetectable time leakage. Uses real `Trajectory` constructors, never dict
literals, so it exercises the actual model, not a reimplementation of it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split
from twin.core.trajectory import Exposure, NoActionStep, Trajectory
from twin.ingest.store import write_trajectories_jsonl
from twin.train.data import load_training_examples


def _trajectory(split: Split, trajectory_id_hint: str) -> Trajectory:
    return Trajectory(
        principal_id="p1",
        context_time=datetime(2026, 1, 1, tzinfo=UTC),
        split=split,
        exposure=Exposure(occurred=True, stimulus=trajectory_id_hint, evidence=ExposureEvidence.HISTORY),
        observation=trajectory_id_hint,
        available_tools=["recall"],
        steps=[NoActionStep(reason="testing")],
        negative_class=NegativeClass.NONE,
        ground_truth_source=GroundTruthSource.OBSERVED,
    )


def test_load_training_examples_yields_exactly_the_train_subset(tmp_path: Path) -> None:
    train_a = _trajectory(Split.TRAIN, "train-a")
    train_b = _trajectory(Split.TRAIN, "train-b")
    heldout = _trajectory(Split.HELDOUT, "heldout")
    sealed = _trajectory(Split.SEALED, "sealed")
    all_trajectories = [train_a, heldout, train_b, sealed]

    uri = f"file://{tmp_path}/trajectories.jsonl"
    write_trajectories_jsonl(all_trajectories, uri)

    loaded = list(load_training_examples(uri))

    assert {t.trajectory_id for t in loaded} == {train_a.trajectory_id, train_b.trajectory_id}
    assert len(loaded) == 2  # exact count, not just "at least" — an extra leaked record must fail this too


def test_load_training_examples_yields_nothing_when_no_train_split_present(tmp_path: Path) -> None:
    uri = f"file://{tmp_path}/trajectories.jsonl"
    write_trajectories_jsonl([_trajectory(Split.HELDOUT, "h"), _trajectory(Split.SEALED, "s")], uri)
    assert list(load_training_examples(uri)) == []
