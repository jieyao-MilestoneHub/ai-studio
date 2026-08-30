#!/usr/bin/env python3
"""Ingest one interview transcript (examples/run_text_interview.py's JSON)
into the fragment store: glossary post-processing (INTERVIEW.md §6.2),
third-party tagging (§7 Q8, structural), split=train (SPEC.md §4.8 via
ingest.split.decide_self_report_split), merge with a backup, and the §7
quality report. Q1/Q2 cost one Teacher call; --no-teacher skips them.

    uv run python examples/ingest_interview_transcript.py \\
        --transcript file:///home/me/twin-data/transcripts/interview-....json \\
        --line-manifest ~/twin-data/raw/manifest.json --principal-display-name <本人在 LINE 的顯示名稱> \\
        [--known-party 媽 --known-party 老闆 ...] [--glossary glossary.json] [--no-teacher]

known parties = every non-principal sender in the LINE manifest + INTERVIEW.md
relationship terms + --known-party. The transcript file itself stays where it
is (file://, never uploaded).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fsspec

from twin.config.settings import get_settings
from twin.ingest.entities import build_glossary_from_contacts
from twin.ingest.interview_ingest import ingest_interview_transcript
from twin.ingest.interviewer import InterviewTranscript
from twin.teacher.gemini import GeminiTeacher

RELATIONSHIP_TERMS = ("媽", "爸", "媽媽", "爸爸", "老闆", "主管", "同事", "男友", "女友", "老婆", "老公", "哥", "姊", "弟", "妹")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", required=True, help="fsspec URI of the interview JSON")
    parser.add_argument("--line-manifest", type=Path, default=None, help='JSON list of {"path","senders"}')
    parser.add_argument("--principal-display-name", default=None, help="excluded from known parties")
    parser.add_argument("--known-party", action="append", dest="known_parties", default=[])
    parser.add_argument("--glossary", type=Path, default=None, help='JSON {"wrong": "right"} corrections')
    parser.add_argument("--no-teacher", action="store_true", help="skip Q1/Q2 (saves one Teacher call)")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.fragment_store_uri.startswith("file://"):
        sys.exit("fragment store must be file:// for self-report content (INTERVIEW.md §6.3)")

    with fsspec.open(args.transcript, "r", encoding="utf-8") as f:
        transcript = InterviewTranscript.model_validate_json(f.read())

    contacts: set[str] = set()
    if args.line_manifest is not None:
        for entry in json.loads(args.line_manifest.expanduser().read_text(encoding="utf-8")):
            contacts.update(entry["senders"])
    contacts.discard(args.principal_display_name or "")
    known_parties = build_glossary_from_contacts(contacts | set(args.known_parties), relationship_terms=RELATIONSHIP_TERMS)

    glossary = json.loads(args.glossary.read_text(encoding="utf-8")) if args.glossary else {}
    teacher = None if args.no_teacher else GeminiTeacher.from_settings(settings)

    summary = ingest_interview_transcript(
        transcript,
        fragment_store_uri=settings.fragment_store_uri,
        known_parties=known_parties,
        correction_glossary=glossary,
        teacher=teacher,
    )
    print(summary.render())
    print("\nNext: write a one-paragraph persona for B1 (EVAL.md §3.4) at the persona URI, then")
    print("      uv run modal run launch/modal_app.py::s1_candidates --labels B0,B1,B2")


if __name__ == "__main__":
    main()
