#!/usr/bin/env python3
"""Conduct the onboarding interview as a single continuous TEXT session in the
terminal — the 2026-08-30 minimal stand-in for INTERVIEW.md §6's voice
interviewer (twin/PLAN.md Phase 3-B). Follows the §4 schedule, follows up
via the Teacher (Gemini, ~1 call per follow-up; well inside the RPD), and
writes the verbatim two-speaker transcript under TWIN_TRANSCRIPT_STORE_URI
(file:// only — INTERVIEW.md §6.3). Nothing enters the fragment store here;
that is examples/ingest_interview_transcript.py's job, after you've read
the transcript back.

    uv run python examples/run_text_interview.py

Answer each question in one line (Enter to send). Type as much as you like;
concrete instances (when / who / what happened) are what the follow-ups are
hunting for. An empty answer moves on. Do not break the session: §3.1
"MUST NOT 分段" — the wall clock runs from the first question to the last.
"""

from __future__ import annotations

import sys

import fsspec

from twin.config.settings import get_settings
from twin.ingest.interviewer import TextInterviewer
from twin.teacher.gemini import GeminiTeacher


def _ask(question: str) -> str:
    print(f"\n訪談員：{question}")
    try:
        return input("本人：")
    except EOFError:
        return ""


def main() -> None:
    settings = get_settings()
    if not sys.stdin.isatty():
        sys.exit("This is an interactive session — run it in a real terminal.")
    teacher = GeminiTeacher.from_settings(settings)
    transcript = TextInterviewer(teacher=teacher, ask=_ask).run(principal_id=settings.principal_id)

    root = settings.transcript_store_uri.rstrip("/")
    stamp = f"{transcript.started_at:%Y%m%dT%H%M%SZ}"
    json_uri = f"{root}/interview-{stamp}.json"
    txt_uri = f"{root}/interview-{stamp}.txt"
    fs, path = fsspec.core.url_to_fs(json_uri)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fsspec.open(json_uri, "w", encoding="utf-8") as f:
        f.write(transcript.model_dump_json(indent=2))
    with fsspec.open(txt_uri, "w", encoding="utf-8") as f:
        f.write(transcript.full_text())

    minutes = (transcript.ended_at - transcript.started_at).total_seconds() / 60
    chars = sum(len(t.text) for t in transcript.turns if t.speaker == "respondent")
    print(f"\nSession {minutes:.0f} min, {chars} respondent chars, {len(transcript.turns)} turns.")
    print(f"Transcript: {json_uri}\n            {txt_uri}")
    for note in transcript.notes:
        print(f"note: {note}")
    print(f"\nNext: uv run python examples/ingest_interview_transcript.py --transcript {json_uri}")


if __name__ == "__main__":
    main()
