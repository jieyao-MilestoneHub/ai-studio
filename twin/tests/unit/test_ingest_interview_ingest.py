"""Transcript -> store merge. SPEC.md D26 (verbatim), §4.8 (split at ingest),
INTERVIEW.md §7 (checks mark, never block — except Q8, structural)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from twin.core.enums import Modality, Precision, SourceClass, Split
from twin.ingest.fragment import fragment_from_text_record
from twin.ingest.interview_ingest import ingest_interview_transcript
from twin.ingest.interviewer import InterviewTranscript, Turn
from twin.ingest.store import read_fragments_jsonl, write_fragments_jsonl

START = datetime(2026, 8, 30, 10, tzinfo=UTC)


def _transcript(minutes: int = 30) -> InterviewTranscript:
    turns = [
        Turn(speaker="interviewer", block="A", point_id=None, text="請跟我說說你的人生故事。", at=START),
        Turn(speaker="respondent", block="A", point_id=None, text="我大概去年夏天搬到台中，小美也一起過來，那時候工作剛換，先租了一間小套房。", at=START),
        Turn(speaker="interviewer", block="B", point_id="B1", text="說三次沒回的事。", at=START),
        Turn(speaker="respondent", block="B", point_id="B1", text="有一次小美傳訊息我沒回，因為在開會，開完會就忘了，到晚上才想起來。", at=START),
    ]
    return InterviewTranscript(
        principal_id="p1",
        mode="text",
        started_at=START,
        ended_at=START + timedelta(minutes=minutes),
        turns=turns,
        notes=[],
    )


def _store_with_one_line_fragment(tmp_path: Path) -> str:
    uri = f"file://{tmp_path}/fragments.jsonl"
    existing = fragment_from_text_record(
        principal_id="p1",
        content="舊訊息",
        event_time=datetime(2025, 1, 1),
        precision=Precision.MINUTE,
        confidence=1.0,
        source_class=SourceClass.BEHAVIOR,
        modality=Modality.TEXT,
        train_cutoff=datetime(2026, 3, 1),
        sealed_cutoff=datetime(2026, 7, 24),
    )
    write_fragments_jsonl([existing], uri)
    return uri


def test_merges_into_existing_store_with_backup_and_verbatim_content(tmp_path: Path) -> None:
    uri = _store_with_one_line_fragment(tmp_path)
    summary = ingest_interview_transcript(
        _transcript(), fragment_store_uri=uri, known_parties=["小美"], correction_glossary={}, teacher=None
    )
    assert summary.fragments_written == 2
    assert summary.store_total_after == 3
    assert summary.backup_uri is not None and Path(summary.backup_uri.removeprefix("file://")).exists()
    fragments = list(read_fragments_jsonl(uri))
    self_report = [f for f in fragments if f.source_class == SourceClass.SELF_REPORT]
    assert all(f.split == Split.TRAIN for f in self_report)
    assert any("本人：我大概去年夏天搬到台中，小美也一起過來" in f.content for f in self_report)
    assert all(f.third_party_spans for f in self_report)  # Q8: 小美 tagged in both blocks


def test_short_text_session_fails_q3_q4_but_still_ingests(tmp_path: Path) -> None:
    uri = _store_with_one_line_fragment(tmp_path)
    summary = ingest_interview_transcript(
        _transcript(minutes=30), fragment_store_uri=uri, known_parties=["小美"], correction_glossary={}, teacher=None
    )
    assert summary.quality.q3_session_duration is False
    assert summary.quality.q4_transcript_length is False
    assert summary.quality.q5_speaker_ratio is True
    assert summary.quality.q7_questionnaire_order is False
    assert any("questionnaire not administered" in n for n in summary.notes)
    assert "low confidence" in summary.render()


def test_refuses_ingesting_the_same_transcript_twice(tmp_path: Path) -> None:
    uri = _store_with_one_line_fragment(tmp_path)
    t = _transcript()
    ingest_interview_transcript(t, fragment_store_uri=uri, known_parties=["小美"], correction_glossary={}, teacher=None)
    with pytest.raises(ValueError, match="already in"):
        ingest_interview_transcript(t, fragment_store_uri=uri, known_parties=["小美"], correction_glossary={}, teacher=None)
    assert len(list(read_fragments_jsonl(uri))) == 3  # nothing written by the refused second call


def test_glossary_correction_is_applied_and_q6_reports_fuzzy_time_kept(tmp_path: Path) -> None:
    uri = _store_with_one_line_fragment(tmp_path)
    summary = ingest_interview_transcript(
        _transcript(), fragment_store_uri=uri, known_parties=["小美"], correction_glossary={"台中": "臺中"}, teacher=None
    )
    fragments = [f for f in read_fragments_jsonl(uri) if f.source_class == SourceClass.SELF_REPORT]
    assert any("臺中" in f.content for f in fragments)
    assert summary.quality.q6_time_expressions.passed
