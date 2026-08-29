#!/usr/bin/env python3
"""Run Phase 1's real LINE-export ingest (twin/PLAN.md Phase 1 "仍待辦 1").

Reads one LINE plain-text chat export, decides every fragment's `split`
from `--train-cutoff` plus EVAL.md §9's 20% sealed hold-back, tags every
non-principal message as third-party (SPEC.md §4.9), and writes the store
at `TWIN_FRAGMENT_STORE_URI`. Refuses to overwrite an existing store.

    uv run python examples/ingest_line_export.py \\
        --export ~/twin-data/raw/chat.txt \\
        --principal-display-name "本人顯示名稱" \\
        --known-sender "本人顯示名稱" --known-sender "對方顯示名稱" \\
        --train-cutoff 2026-01-01

The export's timestamps are naive local times; `--train-cutoff`/`--now` are
parsed the same way so `ingest.split.decide_split` compares like with like.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from twin.config.settings import get_settings
from twin.ingest.line_ingest import IngestRefused, ingest_line_export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="LINE plain-text export file")
    parser.add_argument("--principal-display-name", required=True)
    parser.add_argument(
        "--known-sender", action="append", required=True, dest="known_senders",
        help="every participant's exact display name in this export (repeat; include the principal)",
    )
    parser.add_argument("--train-cutoff", type=datetime.fromisoformat, required=True, help="YYYY-MM-DD[THH:MM]")
    parser.add_argument("--now", type=datetime.fromisoformat, default=None, help="default: now (naive local)")
    parser.add_argument("--sealed-fraction", type=float, default=0.2)
    parser.add_argument("--principal-id", default=None, help="default: TWIN_PRINCIPAL_ID")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing fragment store")
    args = parser.parse_args()

    settings = get_settings()
    text = args.export.read_text(encoding="utf-8")
    try:
        summary = ingest_line_export(
            text,
            uri=settings.fragment_store_uri,
            principal_id=args.principal_id or settings.principal_id,
            principal_display_name=args.principal_display_name,
            known_senders=args.known_senders,
            train_cutoff=args.train_cutoff,
            now=args.now or datetime.now(),
            sealed_fraction=args.sealed_fraction,
            overwrite=args.overwrite,
        )
    except IngestRefused as exc:
        sys.exit(f"refused: {exc}")
    print(summary.render())
    print("Next: uv run python examples/build_s1_item_bank.py")


if __name__ == "__main__":
    main()
