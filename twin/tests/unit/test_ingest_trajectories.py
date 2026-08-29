"""twin.ingest.trajectories — fictional names/content only (SPEC.md §8 guardrail 2)."""

from __future__ import annotations

from datetime import datetime, timedelta

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split
from twin.core.trajectory import ActionStep, NoActionStep, ReflectionStep
from twin.ingest.sources.line import LineMessage
from twin.ingest.trajectories import (
    V1_TOOLS,
    TrajectoryBuildParams,
    trajectories_from_line_messages,
)
from twin.train.formatting import trajectory_to_messages

P = "Alice Chen"
T0 = datetime(2026, 4, 1, 9, 0)
PARAMS = TrajectoryBuildParams(train_cutoff=datetime(2026, 3, 1), sealed_cutoff=datetime(2027, 1, 1))


def _m(minute: int, sender: str, content: str) -> LineMessage:
    return LineMessage(sent_at=T0 + timedelta(minutes=minute), sender=sender, content=content)


def _build(msgs: list[LineMessage], params: TrajectoryBuildParams = PARAMS):
    return list(trajectories_from_line_messages(msgs, principal_id="p", principal_display_name=P, params=params))


def test_reply_within_window_is_an_action_with_the_principals_burst() -> None:
    trajs = _build([_m(0, "Bob", "hi"), _m(1, "Bob", "free tonight?"), _m(10, P, "yes"), _m(11, P, "8pm?")])
    assert len(trajs) == 1
    t = trajs[0]
    assert t.steps == [ActionStep(surface="line", content="yes\n8pm?")]
    assert t.negative_class == NegativeClass.NONE
    assert t.exposure.evidence == ExposureEvidence.INFERRED
    assert t.exposure.stimulus == "Bob: hi\nBob: free tonight?"
    assert t.context_time == T0 + timedelta(minutes=1)
    assert t.split == Split.HELDOUT
    assert t.ground_truth_source == GroundTruthSource.OBSERVED
    assert t.available_tools == list(V1_TOOLS)


def test_no_reply_but_later_activity_is_a_hard_negative_with_inferred_exposure() -> None:
    trajs = _build([_m(0, "Bob", "hi"), _m(0 + 6 * 60, P, "sorry, was out")])
    assert len(trajs) == 1
    t = trajs[0]
    assert isinstance(t.steps[0], NoActionStep)
    assert t.exposure.evidence == ExposureEvidence.INFERRED
    assert t.negative_class == NegativeClass.HARD


def test_no_reply_and_no_later_activity_is_trivial_with_absent_exposure() -> None:
    trajs = _build([_m(0, P, "bye"), _m(5, "Bob", "wait")])
    (t,) = trajs
    assert isinstance(t.steps[0], NoActionStep)
    assert t.exposure.evidence == ExposureEvidence.ABSENT
    assert t.negative_class == NegativeClass.TRIVIAL


def test_reply_after_window_counts_as_no_action_for_that_stimulus() -> None:
    trajs = _build([_m(0, "Bob", "hi"), _m(200, P, "hi")])
    (t,) = trajs
    assert isinstance(t.steps[0], NoActionStep) and t.negative_class == NegativeClass.HARD


def test_observation_carries_bounded_prior_context_and_labels_principal_as_me() -> None:
    msgs = [_m(i * 10, "Bob" if i % 2 == 0 else P, f"m{i}") for i in range(12)]
    trajs = _build(msgs, TrajectoryBuildParams(train_cutoff=datetime(2026, 3, 1), sealed_cutoff=datetime(2027, 1, 1), context_messages=3))
    last = trajs[-1]
    lines = last.observation.splitlines()
    assert len(lines) == 4  # 3 prior + 1-message stimulus burst
    assert any(line.startswith("我: ") for line in lines) and P not in last.observation


def test_split_follows_the_cutoffs_and_never_emits_reflection() -> None:
    msgs = [_m(-60 * 24 * 40, "Bob", "old"), _m(-60 * 24 * 40 + 1, P, "ok"), _m(0, "Bob", "new"), _m(1, P, "ok")]
    trajs = _build(msgs)
    assert [t.split for t in trajs] == [Split.TRAIN, Split.HELDOUT]
    assert not any(isinstance(s, ReflectionStep) for t in trajs for s in t.steps)


def test_trajectories_format_into_sft_messages() -> None:
    trajs = _build([_m(0, "Bob", "hi"), _m(1, P, "yo"), _m(300, "Bob", "?"), _m(2000, P, "late")])
    for t in trajs:
        messages = trajectory_to_messages(t, seed=1)
        assert messages[0]["role"] == "system" and messages[1]["role"] == "user"


def test_activity_beyond_exposure_horizon_is_absent_and_trivial() -> None:
    trajs = _build([_m(0, "Bob", "hi"), _m(3 * 24 * 60, P, "sorry, only saw this now")])
    (t,) = trajs
    assert t.exposure.evidence == ExposureEvidence.ABSENT and t.negative_class == NegativeClass.TRIVIAL


def test_late_reply_can_be_skipped_instead_of_labelled() -> None:
    params = TrajectoryBuildParams(train_cutoff=datetime(2026, 3, 1), sealed_cutoff=datetime(2027, 1, 1), late_reply="skip")
    assert _build([_m(0, "Bob", "hi"), _m(200, P, "hi")], params) == []


def test_counterpart_double_text_within_window_is_one_stimulus() -> None:
    trajs = _build([_m(0, "Bob", "hi"), _m(40, "Bob", "you there?"), _m(50, P, "yes")])
    (t,) = trajs
    assert isinstance(t.steps[0], ActionStep)
    assert t.exposure.stimulus == "Bob: hi\nBob: you there?"


def test_counterpart_bursts_beyond_window_stay_separate() -> None:
    trajs = _build([_m(0, "Bob", "hi"), _m(300, "Bob", "hello?"), _m(305, P, "sorry")])
    assert [type(t.steps[0]) for t in trajs] == [NoActionStep, ActionStep]
