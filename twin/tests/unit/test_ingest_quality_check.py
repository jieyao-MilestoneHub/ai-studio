"""INTERVIEW.md §7's Q1-Q9 quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytest

from twin.ingest.quality_check import (
    CoverageCheckResult,
    InterviewQualityReport,
    check_coverage_and_instances,
    check_q3_session_duration,
    check_q4_transcript_length,
    check_q5_speaker_ratio,
    check_q6_time_expressions_preserved,
    check_q7_questionnaire_after_interview,
    check_q9_postprocessing_steps_ran,
)

STARTED_AT = datetime(2026, 8, 28, 10, 0)


@dataclass
class _FakeTeacher:
    response: object
    calls: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, response_schema: type[Any]) -> Any:
        self.calls.append(prompt)
        return self.response


def test_check_q3_session_duration_within_range() -> None:
    assert check_q3_session_duration(STARTED_AT, STARTED_AT + timedelta(minutes=110)) is True


def test_check_q3_session_duration_too_short() -> None:
    assert check_q3_session_duration(STARTED_AT, STARTED_AT + timedelta(minutes=90)) is False


def test_check_q4_transcript_length_passes_at_the_floor() -> None:
    assert check_q4_transcript_length("x" * 5_500) is True


def test_check_q4_transcript_length_fails_below_the_floor() -> None:
    assert check_q4_transcript_length("x" * 5_499) is False


def test_check_q5_speaker_ratio_passes_when_respondent_dominates() -> None:
    turns = [("interviewer", "問題"), ("respondent", "x" * 100)]
    assert check_q5_speaker_ratio(turns) is True


def test_check_q5_speaker_ratio_fails_when_interviewer_dominates() -> None:
    turns = [("interviewer", "x" * 100), ("respondent", "短")]
    assert check_q5_speaker_ratio(turns) is False


def test_check_q5_speaker_ratio_rejects_empty_turns() -> None:
    with pytest.raises(ValueError, match="no turns"):
        check_q5_speaker_ratio([])


def test_check_q7_questionnaire_after_interview_passes_when_ordered_correctly() -> None:
    ended = STARTED_AT + timedelta(hours=2)
    assert check_q7_questionnaire_after_interview(ended, ended + timedelta(minutes=1)) is True


def test_check_q7_questionnaire_after_interview_fails_when_questionnaire_comes_first() -> None:
    ended = STARTED_AT + timedelta(hours=2)
    assert check_q7_questionnaire_after_interview(ended, ended - timedelta(minutes=1)) is False


def test_check_q9_postprocessing_steps_ran_requires_both_steps() -> None:
    assert check_q9_postprocessing_steps_ran({"correction_glossary": True, "unclear_marking": True}) is True
    assert check_q9_postprocessing_steps_ran({"correction_glossary": True}) is False


def test_check_q6_flags_an_altered_time_expression() -> None:
    result = check_q6_time_expressions_preserved("大概去年夏天", "去年夏天")
    assert result.passed is False
    assert "大概" in result.detail


def test_check_q6_passes_when_time_expressions_are_untouched() -> None:
    result = check_q6_time_expressions_preserved("大概去年夏天", "大概去年夏天")
    assert result.passed is True


def test_check_coverage_and_instances_calls_teacher_exactly_once() -> None:
    response = CoverageCheckResult(covered={"A1": True}, instance_counts={"B1": 3})
    teacher = _FakeTeacher(response=response)

    result = check_coverage_and_instances("逐字稿內容", teacher=teacher)

    assert len(teacher.calls) == 1
    assert result.covered == {"A1": True}
    assert result.instance_counts == {"B1": 3}


def test_check_coverage_and_instances_prompt_contains_the_transcript() -> None:
    response = CoverageCheckResult(covered={}, instance_counts={})
    teacher = _FakeTeacher(response=response)

    check_coverage_and_instances("一段獨特的逐字稿內容標記", teacher=teacher)

    assert "一段獨特的逐字稿內容標記" in teacher.calls[0]


def test_interview_quality_report_marks_q8_as_structural_not_a_field() -> None:
    """Q8 (third_party_spans tagging) is INTERVIEW.md §7's one hard blocker
    and is enforced structurally in `ingest.sources.interview_transcript`
    (extraction runs unconditionally, with no code path to skip it) — not as
    a post-hoc report field here."""
    assert "q8" not in InterviewQualityReport.model_fields
