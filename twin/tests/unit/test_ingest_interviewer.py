"""Text interviewer session logic. INTERVIEW.md §2 (an interview follows up;
a questionnaire does not), §4 (schedule order, opening line), §6.1."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel

from twin.ingest.interview_schedule import BLOCK_OPENINGS, COVERAGE_POINTS
from twin.ingest.interviewer import FollowUpDecision, InterviewTranscript, TextInterviewer
from twin.teacher.base import TeacherRateExhausted

T = TypeVar("T", bound=BaseModel)


class _ScriptedTeacher:
    """Returns `not reached` + a follow-up the first `unreached_rounds`
    times per point, then `reached`."""

    def __init__(self, unreached_rounds: int) -> None:
        self.unreached_rounds = unreached_rounds
        self.calls: list[str] = []
        self._seen: dict[str, int] = {}

    def generate(self, prompt: str, *, response_schema: type[T]) -> T:
        assert response_schema is FollowUpDecision
        self.calls.append(prompt)
        point = prompt.split("目前的必達點是 ")[1].split("：")[0]
        n = self._seen.get(point, 0)
        self._seen[point] = n + 1
        if n < self.unreached_rounds:
            return response_schema(concrete_instances=n, reached=False, follow_up_question=f"追問{point}-{n}")
        return response_schema(concrete_instances=3, reached=True, follow_up_question=None)


class _ExhaustedTeacher:
    def generate(self, prompt: str, *, response_schema: type[T]) -> T:
        raise TeacherRateExhausted("quota")


def _ticking_clock(start: datetime):
    state = {"t": start}

    def clock() -> datetime:
        state["t"] += timedelta(minutes=1)
        return state["t"]

    return clock


def _run(teacher, answers_prefix: str = "答") -> tuple[InterviewTranscript, list[str]]:
    asked: list[str] = []

    def ask(question: str) -> str:
        asked.append(question)
        return f"{answers_prefix}{len(asked)}"

    interviewer = TextInterviewer(teacher=teacher, ask=ask, clock=_ticking_clock(datetime(2026, 8, 30, 10, tzinfo=UTC)))
    return interviewer.run(principal_id="p1"), asked


def test_opening_line_is_the_mandated_open_ended_one_and_blocks_run_in_order() -> None:
    transcript, asked = _run(_ScriptedTeacher(unreached_rounds=0))
    assert asked[0] == BLOCK_OPENINGS["A"]
    blocks_in_order = [t.block for t in transcript.turns]
    assert blocks_in_order == sorted(blocks_in_order, key="ABCD".index)


def test_every_coverage_point_is_asked() -> None:
    transcript, _ = _run(_ScriptedTeacher(unreached_rounds=0))
    asked_points = {t.point_id for t in transcript.turns if t.speaker == "interviewer"}
    assert {p.point_id for p in COVERAGE_POINTS} <= asked_points


def test_follows_up_until_reached_but_at_most_max_times() -> None:
    teacher = _ScriptedTeacher(unreached_rounds=5)  # never reaches on its own
    transcript, _ = _run(teacher)
    b1 = [t for t in transcript.turns if t.point_id == "B1" and t.speaker == "interviewer"]
    assert len(b1) == 1 + 2 + 1  # question + MAX_FOLLOW_UPS_PER_POINT + one block-end probe (INTERVIEW.md §6.1)
    assert b1[1].text == "追問B1-0"
    assert any("B1: not reached after block-end probe" in n for n in transcript.notes)


def test_points_with_no_instance_requirement_are_not_followed_up() -> None:
    teacher = _ScriptedTeacher(unreached_rounds=5)
    transcript, _ = _run(teacher)
    d2 = [t for t in transcript.turns if t.point_id == "D2" and t.speaker == "interviewer"]
    assert len(d2) == 1
    assert not any("必達點是 D2" in c for c in teacher.calls)


def test_teacher_exhaustion_is_recorded_not_fatal() -> None:
    transcript, _ = _run(_ExhaustedTeacher())
    teacher_notes = [n for n in transcript.notes if "follow-up skipped" in n]
    assert teacher_notes
    assert all("Teacher unavailable" in n for n in teacher_notes)
    assert not any("block-end probe" in n for n in transcript.notes)  # can't probe without a Teacher; not double-counted


def test_block_texts_keep_both_speakers_verbatim() -> None:
    transcript, _ = _run(_ScriptedTeacher(unreached_rounds=0), answers_prefix="嗯，大概去年夏天吧 ")
    blocks = transcript.block_texts()
    assert set(blocks) == {"A", "B", "C", "D"}
    assert "訪談員：" in blocks["A"] and "本人：嗯，大概去年夏天吧" in blocks["A"]
    assert transcript.ended_at > transcript.started_at


def test_respondent_turn_records_the_answer_as_given() -> None:
    transcript, _ = _run(_ScriptedTeacher(unreached_rounds=0))
    respondent = [t for t in transcript.turns if t.speaker == "respondent"]
    assert respondent[0].text == "答1"


@pytest.mark.parametrize("point_id", ["B1", "B2", "B6"])
def test_b1_b2_b6_require_three_instances(point_id: str) -> None:
    point = next(p for p in COVERAGE_POINTS if p.point_id == point_id)
    assert point.required_instances == 3  # INTERVIEW.md §7 Q2


def test_block_end_probe_happens_within_the_same_block() -> None:
    """INTERVIEW.md §6.1: unreached points are re-asked before leaving the
    block (the only remedy once re-interviews are excluded, D35)."""
    transcript, _ = _run(_ScriptedTeacher(unreached_rounds=5))
    b_turns = [t for t in transcript.turns if t.block == "B"]
    first_b8_question = min(i for i, t in enumerate(b_turns) if t.point_id == "B8" and t.speaker == "interviewer")
    probes_after = [t for t in b_turns[first_b8_question + 1 :] if t.speaker == "interviewer" and t.point_id == "B1"]
    assert probes_after, "expected a block-end probe for B1 after B8 was asked, still inside block B"
    assert all(t.block == "B" for t in probes_after)


def test_mid_block_speaker_share_is_self_checked() -> None:
    transcript, _ = _run(_ScriptedTeacher(unreached_rounds=0), answers_prefix="嗯")
    assert any("respondent share below 70%" in n for n in transcript.notes)
