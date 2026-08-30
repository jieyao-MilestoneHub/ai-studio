"""Interview -> trajectories. SPEC.md §4.10, D19, D37, D39."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from twin.core.enums import ExposureEvidence, NegativeClass, Split
from twin.core.trajectory import ActionStep
from twin.ingest.interview_trajectories import trajectories_from_interview
from twin.ingest.interviewer import InterviewTranscript, Turn
from twin.train.formatting import trajectory_to_messages

T0 = datetime(2026, 8, 30, 10, tzinfo=UTC)


def _t(i: int, speaker: str, block: str, text: str, point: str | None = "A1") -> Turn:
    return Turn(speaker=speaker, block=block, point_id=point, text=text, at=T0 + timedelta(minutes=i))  # type: ignore[arg-type]


def _transcript() -> InterviewTranscript:
    return InterviewTranscript(
        principal_id="p1",
        mode="text",
        started_at=T0,
        ended_at=T0 + timedelta(minutes=10),
        turns=[
            _t(0, "interviewer", "A", "請跟我說說你的人生故事。", None),
            _t(1, "respondent", "A", "我在台北長大，大學念資工。", None),
            _t(2, "interviewer", "A", "有哪些轉折點？"),
            _t(3, "respondent", "A", "2019 年離職創業。"),
            _t(4, "interviewer", "A", "當時有哪些選項？"),
            _t(5, "respondent", "A", ""),  # skipped -> no trajectory
            _t(6, "interviewer", "B", "說三次沒回的事。", "B1"),
            _t(7, "respondent", "B", "老闆半夜傳訊息我隔天才回。", "B1"),
        ],
        notes=[],
    )


def test_one_trajectory_per_answered_question_skipping_empty_answers() -> None:
    trajectories = list(trajectories_from_interview(_transcript()))
    assert len(trajectories) == 3
    assert [t.steps[0].content for t in trajectories] == [  # type: ignore[union-attr]
        "我在台北長大，大學念資工。",
        "2019 年離職創業。",
        "老闆半夜傳訊息我隔天才回。",
    ]


def test_split_is_train_and_no_negatives_and_certain_exposure() -> None:
    for t in trajectories_from_interview(_transcript()):
        assert t.split == Split.TRAIN
        assert t.negative_class == NegativeClass.NONE
        assert t.exposure.evidence == ExposureEvidence.READ_RECEIPT
        assert isinstance(t.steps[0], ActionStep) and t.steps[0].surface == "interview"


def test_observation_keeps_same_block_context_only() -> None:
    trajectories = list(trajectories_from_interview(_transcript()))
    b1 = trajectories[2]
    assert "說三次沒回的事" in b1.observation
    assert "人生故事" not in b1.observation  # block A context does not bleed into block B
    a2 = trajectories[1]
    assert "我在台北長大" in a2.observation


def test_formats_through_the_same_sft_path_as_line_replies() -> None:
    trajectory = next(trajectories_from_interview(_transcript()))
    messages = trajectory_to_messages(trajectory, seed=0)
    assistant = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
    assert assistant, "the answer must become a (masked) reply tool call like a LINE reply"
    assert "我在台北長大" in assistant[0]["tool_calls"][0]["function"]["arguments"]
