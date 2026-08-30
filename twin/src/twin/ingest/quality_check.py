"""INTERVIEW.md §7's Q1-Q9 quality checks, split by mechanism. Q8
(third_party_spans tagging) has no function here — it's enforced
structurally in `ingest.sources.interview_transcript`, the ONE hard blocker
in §7's table ("MUST 阻擋：未標註前逐字稿 MUST NOT 進入記憶層"). Every other
check here only ever downgrades a suite's confidence — INTERVIEW.md §8 I-C
decided against re-interviewing, so a failed check is recorded, not acted on.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from twin.teacher.base import Teacher

MIN_SESSION_MINUTES = 102
MAX_SESSION_MINUTES = 120
MIN_TRANSCRIPT_CHARS = 5_500
MIN_RESPONDENT_SHARE = 0.70

# A representative, not exhaustive, set of common Traditional-Chinese fuzzy
# time expressions — a heuristic proxy for "verbatim retention" (Q6), not a
# guarantee. INTERVIEW.md §6.2 step 3/C2 requires these preserved verbatim.
_FUZZY_TIME_PATTERN = re.compile(r"大概|去年|前年|那陣子|那時候|上個月|之前")


def check_q3_session_duration(started_at: datetime, ended_at: datetime) -> bool:
    minutes = (ended_at - started_at).total_seconds() / 60
    return MIN_SESSION_MINUTES <= minutes <= MAX_SESSION_MINUTES


def check_q4_transcript_length(text: str) -> bool:
    return len(text) >= MIN_TRANSCRIPT_CHARS


def check_q5_speaker_ratio(turns: list[tuple[str, str]]) -> bool:
    """`turns` is `(speaker, text)` pairs, `speaker` one of "respondent" /
    "interviewer". Measured by character count, not turn count — a one-word
    interviewer prompt followed by a long respondent answer must not count
    as an even split."""
    if not turns:
        raise ValueError("no turns to measure a speaker ratio from")
    total_chars = sum(len(text) for _, text in turns)
    if total_chars == 0:
        raise ValueError("all turns are empty — nothing to measure a speaker ratio from")
    respondent_chars = sum(len(text) for speaker, text in turns if speaker == "respondent")
    return (respondent_chars / total_chars) >= MIN_RESPONDENT_SHARE


def check_q7_questionnaire_after_interview(interview_ended_at: datetime, questionnaire_started_at: datetime) -> bool:
    return questionnaire_started_at >= interview_ended_at


def check_q9_postprocessing_steps_ran(pipeline_log: dict[str, bool]) -> bool:
    """`pipeline_log` is a completion-flag record the pipeline's *caller*
    produces (kept separate from `ingest.postprocess.run_postprocessing_
    pipeline` itself, so that function stays a pure text-in/text-out
    transform with no side-channel bookkeeping)."""
    required = ("correction_glossary", "unclear_marking")
    return all(pipeline_log.get(step, False) for step in required)


class QualityCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    detail: str


def check_q6_time_expressions_preserved(raw_text: str, corrected_text: str) -> QualityCheckResult:
    """A heuristic proxy, not a guarantee: flags any common fuzzy-time
    expression present in `raw_text` but absent from `corrected_text` — i.e.
    altered or dropped by `ingest.postprocess.apply_correction_glossary`."""
    raw_matches = set(_FUZZY_TIME_PATTERN.findall(raw_text))
    corrected_matches = set(_FUZZY_TIME_PATTERN.findall(corrected_text))
    dropped = raw_matches - corrected_matches
    if dropped:
        return QualityCheckResult(
            passed=False, detail=f"fuzzy time expression(s) altered or dropped: {sorted(dropped)}"
        )
    return QualityCheckResult(passed=True, detail="all fuzzy time expressions preserved")


class CoverageCheckResult(BaseModel):
    """INTERVIEW.md §4's required coverage points (A1-A4/B1-B8/C1-C3/D1-D2).
    Q1 (全部必達點已涵蓋) and Q2 (B1/B2/B6 各至少三個具體事例) are the same
    underlying judgment at different granularities, merged into one Teacher
    call (D9: 少次、大批)."""

    model_config = ConfigDict(frozen=True)

    covered: dict[str, bool]  # e.g. {"A1": True, "B6": False, ...}
    instance_counts: dict[str, int]  # only meaningful for B1/B2/B6


class _PointCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    point_id: str
    count: int


class _CoveragePayload(BaseModel):
    """Gemini's structured-output schema: the Developer API rejects `dict`
    fields (`additionalProperties`, found on the first real call 2026-08-30),
    so the Teacher answers in lists and `check_coverage_and_instances`
    converts to the dict-shaped `CoverageCheckResult` the rest of the code
    reads."""

    model_config = ConfigDict(frozen=True)

    covered_points: list[str]
    uncovered_points: list[str]
    instance_counts: list[_PointCount]


_ALL_POINTS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "C1", "C2", "C3", "D1", "D2")


def check_coverage_and_instances(transcript: str, *, teacher: Teacher) -> CoverageCheckResult:
    """One batched `Teacher.generate()` call, never two — merges Q1 and Q2's
    judgments (twin.teacher.Teacher, not the eval-judge/EVAL.md judge path:
    this is an ingest-time QC classification, not an EVAL suite score)."""
    prompt = (
        "以下是一份訪談逐字稿。請判斷 INTERVIEW.md §4 規定的每個必達點是否已被涵蓋"
        "（A1 三個人生轉折點含時間/選項/理由；A2 兩次選錯的事例；A3 自己與他人的差異；A4 在乎與不在乎各兩項附事例；"
        "B1 三個想回沒回的事例；B2 三個不必回但回了的事例；B3 查/不查的分界各一例；B4 放棄點與一例；"
        "B5 什麼會主動發言及最近一次；B6 三個有感觸但不發的事例；B7 主動推薦的東西與方式；B8 對不同對象的回覆差異；"
        "C1 三個時期各一件事；C2 自然的時間表述；C3 表示記不清的情況；D1 未問到但必須知道的；D2 最不希望代理做的事），"
        "把已涵蓋的必達點代號放進 covered_points、未涵蓋的放進 uncovered_points（兩者合起來必須恰好是這 17 個），"
        "並針對 B1、B2、B6 各回報具體事例（有時間、有對象、有結果的真實事件）的數量到 instance_counts。"
        "\n\n逐字稿：\n" + transcript
    )
    payload = teacher.generate(prompt, response_schema=_CoveragePayload)
    covered = {p: False for p in _ALL_POINTS}
    for p in payload.covered_points:
        if p in covered:
            covered[p] = True
    return CoverageCheckResult(
        covered=covered,
        instance_counts={pc.point_id: pc.count for pc in payload.instance_counts if pc.point_id in ("B1", "B2", "B6")},
    )


class InterviewQualityReport(BaseModel):
    """Aggregates every check above. Deliberately has no `q8` field — Q8 is
    enforced structurally in `ingest.sources.interview_transcript`, not as a
    post-hoc flag here."""

    model_config = ConfigDict(frozen=True)

    q1_q2: CoverageCheckResult
    q3_session_duration: bool
    q4_transcript_length: bool
    q5_speaker_ratio: bool
    q6_time_expressions: QualityCheckResult
    q7_questionnaire_order: bool
    q9_postprocessing_ran: bool
