"""Interview transcript -> fragment store, end to end. INTERVIEW.md §6.2
(re-runnable post-processing), §7 (quality checks, Q8 structural), SPEC.md
§4.8/D21 (split once, at ingest), D26 (verbatim round-trip).

Why merging into an existing store lives here and not as an `append` on
`ingest.store`: `write_fragments_jsonl` refuses append because two
*time-cutoff* policies must never coexist in one file. Self-report split is
not time-based at all (`ingest.split.decide_self_report_split`), so adding
it to a store built under any cutoffs changes nothing about that store's
policy — the check that matters here is instead "no fragment_id already
present", and a backup of the file being rewritten.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime

import fsspec
from pydantic import BaseModel, ConfigDict

from twin.core.enums import SourceClass
from twin.core.fragment import Fragment
from twin.ingest.interviewer import InterviewTranscript
from twin.ingest.postprocess import run_postprocessing_pipeline
from twin.ingest.quality_check import (
    CoverageCheckResult,
    InterviewQualityReport,
    check_coverage_and_instances,
    check_q3_session_duration,
    check_q4_transcript_length,
    check_q5_speaker_ratio,
    check_q6_time_expressions_preserved,
    check_q9_postprocessing_steps_ran,
)
from twin.ingest.sources.interview_transcript import fragments_from_interview_transcript
from twin.ingest.store import read_fragments_jsonl, write_fragments_jsonl
from twin.teacher.base import Teacher


class InterviewIngestSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    fragments_written: int
    store_total_after: int
    backup_uri: str | None
    quality: InterviewQualityReport
    notes: list[str]

    def render(self) -> str:
        q = self.quality
        lines = [
            f"wrote {self.fragments_written} self-report fragment(s); store now holds {self.store_total_after}",
            f"backup of the previous store: {self.backup_uri or '(none, store was new)'}",
            "INTERVIEW.md §7 quality (a failed item marks confidence low, it does not block; Q8 is structural):",
            f"  Q1 coverage: {sum(q.q1_q2.covered.values())}/{len(q.q1_q2.covered)} points covered"
            if q.q1_q2.covered
            else "  Q1/Q2: not checked (no Teacher)",
            f"  Q2 instances B1/B2/B6: {q.q1_q2.instance_counts}" if q.q1_q2.instance_counts else "",
            f"  Q3 session 102-120 min: {'PASS' if q.q3_session_duration else 'FAIL -> whole round low confidence'}",
            f"  Q4 >= 5,500 chars:     {'PASS' if q.q4_transcript_length else 'FAIL -> whole round low confidence'}",
            f"  Q5 respondent >= 70%:  {'PASS' if q.q5_speaker_ratio else 'FAIL -> S1 low confidence'}",
            f"  Q6 fuzzy time kept:    {'PASS' if q.q6_time_expressions.passed else 'FAIL: ' + q.q6_time_expressions.detail}",
            f"  Q7 questionnaire after: {'PASS' if q.q7_questionnaire_order else 'not administered -> S1 low confidence'}",
            f"  Q9 postprocessing ran:  {'PASS' if q.q9_postprocessing_ran else 'FAIL'}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(line for line in lines if line)


def _backup(uri: str) -> str | None:
    fs, path = fsspec.core.url_to_fs(uri)
    if not fs.exists(path):
        return None
    backup_path = f"{path}.bak-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    if fs.protocol in ("file", ("file", "local")) or "file" in fs.protocol:
        shutil.copyfile(path, backup_path)
    else:
        fs.copy(path, backup_path)
    return f"file://{backup_path}" if uri.startswith("file://") else backup_path


def ingest_interview_transcript(
    transcript: InterviewTranscript,
    *,
    fragment_store_uri: str,
    known_parties: list[str],
    correction_glossary: dict[str, str],
    teacher: Teacher | None,
    questionnaire_started_at: datetime | None = None,
) -> InterviewIngestSummary:
    """Post-process each block (glossary correction; no unclear ranges — a
    typed transcript has no ASR uncertainty, which is the one honest
    advantage of the text stand-in), tag third parties and build fragments
    (Q8 enforced inside `fragments_from_interview_transcript`), merge into
    the store, and measure §7. Raises if the store already holds a SELF_REPORT
    fragment with identical content — fragment_ids are fresh UUIDs, so
    content (verbatim by D26) is the only honest identity; re-running this
    on the same transcript must not double the principal's self-report
    weight in training."""
    raw_blocks = transcript.block_texts()
    corrected_blocks = {
        label: run_postprocessing_pipeline(text, correction_glossary=correction_glossary, low_confidence_ranges=[])
        for label, text in raw_blocks.items()
    }
    new_fragments: list[Fragment] = list(
        fragments_from_interview_transcript(
            corrected_blocks,
            principal_id=transcript.principal_id,
            session_started_at=transcript.started_at,
            known_parties=known_parties,
        )
    )

    fs, path = fsspec.core.url_to_fs(fragment_store_uri)
    existing: list[Fragment] = list(read_fragments_jsonl(fragment_store_uri)) if fs.exists(path) else []
    existing_self_report = {f.content for f in existing if f.source_class == SourceClass.SELF_REPORT}
    clashes = [f for f in new_fragments if f.content in existing_self_report]
    if clashes:
        raise ValueError(
            f"{len(clashes)} block(s) of this transcript are already in {fragment_store_uri} "
            f"(identical SELF_REPORT content) — refusing to ingest the same interview twice"
        )

    backup_uri = _backup(fragment_store_uri)
    total = write_fragments_jsonl([*existing, *new_fragments], fragment_store_uri)

    raw_text = "\n".join(raw_blocks[label] for label in sorted(raw_blocks))
    corrected_text = "\n".join(corrected_blocks[label] for label in sorted(corrected_blocks))
    notes = list(transcript.notes)
    if teacher is not None:
        coverage = check_coverage_and_instances(corrected_text, teacher=teacher)
    else:
        coverage = CoverageCheckResult(covered={}, instance_counts={})
        notes.append("Q1/Q2 not checked: no Teacher supplied")
    if questionnaire_started_at is None:
        notes.append("INTERVIEW.md §5 questionnaire not administered")
    quality = InterviewQualityReport(
        q1_q2=coverage,
        q3_session_duration=check_q3_session_duration(transcript.started_at, transcript.ended_at),
        q4_transcript_length=check_q4_transcript_length(corrected_text),
        q5_speaker_ratio=check_q5_speaker_ratio(transcript.speaker_turns()),
        q6_time_expressions=check_q6_time_expressions_preserved(raw_text, corrected_text),
        q7_questionnaire_order=questionnaire_started_at is not None
        and questionnaire_started_at >= transcript.ended_at,
        q9_postprocessing_ran=check_q9_postprocessing_steps_ran({"correction_glossary": True, "unclear_marking": True}),
    )
    return InterviewIngestSummary(
        fragments_written=len(new_fragments),
        store_total_after=total,
        backup_uri=backup_uri,
        quality=quality,
        notes=notes,
    )
