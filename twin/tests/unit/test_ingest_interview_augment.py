"""Paraphrase augmentation. SPEC.md §4.10 ground_truth_source, D25, C1."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel

from twin.core.enums import GroundTruthSource
from twin.ingest.interview_augment import _VariantsPayload, augment_interview_trajectories
from twin.ingest.interview_trajectories import trajectories_from_interview
from twin.ingest.interviewer import InterviewTranscript, Turn

T = TypeVar("T", bound=BaseModel)
T0 = datetime(2026, 8, 30, 10, tzinfo=UTC)


class _FakeTeacher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, *, response_schema: type[T]) -> T:
        self.calls.append(prompt)
        assert response_schema is _VariantsPayload
        return response_schema(
            variants=[
                {"question": "轉折點有哪些？", "answer": "其實就是2019年離職創業那次。"},
                {"question": "", "answer": "2019 年離職創業。"},  # identical to original -> dropped
                {"question": "還有嗎", "answer": ""},  # empty -> dropped
                {"question": "說說轉折", "answer": "2019 年離職出來創業，算蛮大的转折哈哈。"},  # simplified slips -> normalised
            ]
        )


def _source():
    t = InterviewTranscript(
        principal_id="p1", mode="text", started_at=T0, ended_at=T0 + timedelta(minutes=5),
        turns=[
            Turn(speaker="interviewer", block="A", point_id="A1", text="有哪些轉折點？", at=T0),
            Turn(speaker="respondent", block="A", point_id="A1", text="2019 年離職創業。", at=T0),
        ],
        notes=[],
    )
    return list(trajectories_from_interview(t))


def test_variants_are_synthesized_keep_shape_and_drop_empty_or_identical() -> None:
    teacher = _FakeTeacher()
    out = list(augment_interview_trajectories(_source(), teacher=teacher, variants_per_trajectory=4, style_samples=["對呀"]))
    assert len(teacher.calls) == 1  # D9: one call per source, K variants per call
    assert len(out) == 2
    assert all(t.ground_truth_source == GroundTruthSource.TEACHER_SYNTHESIZED for t in out)
    assert all(t.split == _source()[0].split and t.steps[0].surface == "interview" for t in out)  # type: ignore[union-attr]
    assert out[0].observation == "訪談員：轉折點有哪些？"
    assert out[1].steps[0].content == "2019 年離職出來創業，算蠻大的轉折哈哈。"  # type: ignore[union-attr]


def test_prompt_forbids_new_facts_and_carries_style_samples() -> None:
    teacher = _FakeTeacher()
    list(augment_interview_trajectories(_source(), teacher=teacher, variants_per_trajectory=2, style_samples=["酷喔", "這啥"]))
    assert "MUST NOT 新增" in teacher.calls[0] and "酷喔" in teacher.calls[0]


def test_rejects_non_interview_trajectory() -> None:
    src0 = _source()[0]
    src = src0.model_copy(update={"steps": [src0.steps[0].model_copy(update={"surface": "line"})]})
    with pytest.raises(ValueError, match="not an interview trajectory"):
        list(augment_interview_trajectories([src], teacher=_FakeTeacher(), variants_per_trajectory=1, style_samples=[]))
