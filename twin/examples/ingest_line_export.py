#!/usr/bin/env python3
"""Ingest a real LINE chat export into `fragments.jsonl`. SPEC.md §4.4, §4.8/D21,
§4.9/D23; twin/PLAN.md Phase 1.

Reads the raw export text and the split boundary from the command line rather
than guessing — `principal_display_name` and `known_senders` are the
principal's real personal data and MUST come from the person running this,
not from this script reading through the export and inferring names itself
(twin/ingest/sources/line.py's docstring: a wrong guess here silently
corrupts every message from that point on). `train_cutoff` is SPEC.md
§4.8/D21's write-once split boundary, so this script first reports the
export's real first/last message timestamps and requires you to supply
`--train-cutoff` informed by that span, rather than accepting a default —
getting this wrong isn't fixable by re-running with better judgement later,
only by re-ingesting from scratch.

    uv run python examples/ingest_line_export.py \\
        --principal-display-name "Alice Chen" \\
        --known-senders "Alice Chen,Bob"

which reports the parsed span and exits; then, once a `--train-cutoff` is
chosen from that span:

    uv run python examples/ingest_line_export.py \\
        --principal-display-name "Alice Chen" \\
        --known-senders "Alice Chen,Bob" \\
        --train-cutoff 2026-06-01

Everything before `--train-cutoff` becomes `Split.TRAIN`; the rest is
`Split.HELDOUT`, with the most recent `--sealed-fraction` (default 0.2,
EVAL.md §9's 20%-reserved-split) of that becoming `Split.SEALED`
(`ingest.split.sealed_cutoff_for`, using the export's own last message as
"now" rather than wall-clock time, so the sealed slice is carved from real
data rather than from the empty time since the export was taken).

Writes to `TWIN_FRAGMENT_STORE_URI` (default `file://./data/fragments.jsonl`)
after an explicit confirmation — `write_fragments_jsonl` overwrites wholesale,
so re-running this later with different inputs replaces the file, not appends.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime

import fsspec

from twin.config.settings import get_settings
from twin.core.enums import Split
from twin.ingest.fragment import fragments_from_line_export
from twin.ingest.sources.line import parse_line_export
from twin.ingest.split import sealed_cutoff_for
from twin.ingest.store import write_fragments_jsonl

RAW_EXPORT_URI = "file://./data/line_export_raw.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--principal-display-name", required=True, help="exact display name in this export")
    parser.add_argument(
        "--known-senders", required=True, help="comma-separated exact display names, principal included"
    )
    parser.add_argument(
        "--train-cutoff",
        default=None,
        help="YYYY-MM-DD. Omit on a first run to see the export's real time span before choosing.",
    )
    parser.add_argument("--sealed-fraction", type=float, default=0.2)
    parser.add_argument("--raw-export-uri", default=RAW_EXPORT_URI)
    args = parser.parse_args()

    known_senders = [s.strip() for s in args.known_senders.split(",") if s.strip()]
    if args.principal_display_name not in known_senders:
        sys.exit(
            f"--principal-display-name {args.principal_display_name!r} must also appear in --known-senders"
        )

    fs, path = fsspec.core.url_to_fs(args.raw_export_uri)
    if not fs.exists(path):
        sys.exit(f"No raw export found at {args.raw_export_uri}")
    with fsspec.open(args.raw_export_uri, "r", encoding="utf-8") as f:
        text = f.read()

    messages = list(
        parse_line_export(text, known_senders=known_senders, principal_display_name=args.principal_display_name)
    )
    if not messages:
        sys.exit("Parsed 0 messages — check --known-senders/--principal-display-name against the real export.")

    first_at = min(m.sent_at for m in messages)
    last_at = max(m.sent_at for m in messages)
    print(f"Parsed {len(messages)} messages spanning {first_at.isoformat()} .. {last_at.isoformat()}.")

    if args.train_cutoff is None:
        sys.exit(
            "\nNo --train-cutoff given. Pick a date within the span above — everything before it "
            "becomes Split.TRAIN, the rest Split.HELDOUT (with the most recent "
            f"{args.sealed_fraction:.0%} of that becoming Split.SEALED). This is write-once "
            "(SPEC.md §4.8/D21) — re-run with --train-cutoff YYYY-MM-DD once decided."
        )
    train_cutoff = datetime.strptime(args.train_cutoff, "%Y-%m-%d")
    if not (first_at <= train_cutoff <= last_at):
        print(
            f"Warning: --train-cutoff {train_cutoff.date()} falls outside the parsed span "
            f"({first_at.date()} .. {last_at.date()}) — every fragment will land in a single split."
        )
    sealed_cutoff = sealed_cutoff_for(train_cutoff=train_cutoff, now=last_at, sealed_fraction=args.sealed_fraction)

    settings = get_settings()
    fragments = list(
        fragments_from_line_export(
            text,
            principal_id=settings.principal_id,
            principal_display_name=args.principal_display_name,
            known_senders=known_senders,
            train_cutoff=train_cutoff,
            sealed_cutoff=sealed_cutoff,
        )
    )

    breakdown = Counter(f.split for f in fragments)
    third_party_count = sum(1 for f in fragments if f.third_party_spans)
    print(f"\n{len(fragments)} fragments built from {len(messages)} messages:")
    for split in (Split.TRAIN, Split.HELDOUT, Split.SEALED):
        print(f"  {split.value}: {breakdown.get(split, 0)}")
    print(f"  third_party_spans present: {third_party_count}/{len(fragments)}")
    print(f"  split boundaries: train < {train_cutoff.isoformat()} <= heldout < {sealed_cutoff.isoformat()} <= sealed")
    if breakdown.get(Split.HELDOUT, 0) == 0:
        print("\nWarning: 0 held-out fragments — Phase 2's item bank draws only from Split.HELDOUT.")

    fragment_store_uri = settings.fragment_store_uri
    answer = (
        input(f"\nWrite {len(fragments)} fragments to {fragment_store_uri}? [y/N] ").strip().lower()
    )
    if answer != "y":
        print("Not written.")
        return

    count = write_fragments_jsonl(fragments, fragment_store_uri)
    print(f"\nWrote {count} fragments to {fragment_store_uri}.")
    print("Next: uv run python examples/build_s1_item_bank.py")


if __name__ == "__main__":
    main()
