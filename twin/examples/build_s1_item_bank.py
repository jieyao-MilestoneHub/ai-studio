#!/usr/bin/env python3
"""Generate + freeze the S1 persona-fidelity item bank. EVAL.md §3.2,
twin/PLAN.md Phase 2. "Project day 0" begins when Wave 1 (R1) finishes
being collected against the bank this script freezes — NOT when this
script runs; see examples/collect_s1_answers.py for that step.

Reads every `Split.HELDOUT` fragment from `TWIN_FRAGMENT_STORE_URI`, makes
exactly one real Teacher call (D9: 少次、大批 — this burns 1 unit of the
day's Gemini quota), prints the returned item-type breakdown for a human to
eyeball against EVAL.md §3.2's 30/25/25/20 SHOULD ratio, then asks for
explicit confirmation before writing the bank as a write-once, frozen
artifact under `TWIN_S1_EVAL_ROOT_URI` (core.hashing.dataset_hash-backed — any
post-freeze edit is detectable, see harness.item_bank.read_and_verify_item_bank).

    uv run python examples/build_s1_item_bank.py

Requires TWIN_GEMINI_API_KEY/TWIN_GEMINI_MODEL (twin/PLAN.md Phase 0, done)
and a real twin/data/fragments.jsonl (twin/PLAN.md Phase 1's real
LINE-export ingest — not yet run in this checkout as of 2026-08-28; this
script fails loudly with an actionable message until that happens, rather
than a bare FileNotFoundError traceback).
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import UTC, datetime

import fsspec

from twin.config.settings import get_settings
from twin.core.enums import Split
from twin.core.hashing import dataset_hash
from twin.harness.item_bank import S1BankManifest, bank_hash, write_item_bank_once
from twin.harness.suites.s1 import build_item_bank, sample_held_out_windows
from twin.ingest.store import read_fragments_jsonl
from twin.teacher.gemini import GeminiTeacher

SAMPLE_SEED = 0  # deterministic: same store + seed -> same sampled subset -> same dataset hash

_ITEM_TYPES = ("value_tradeoff", "preference", "reaction_tendency", "recall")


def main() -> None:
    settings = get_settings()
    bank_uri = f"{settings.s1_eval_root_uri}/item_bank.jsonl"
    manifest_uri = f"{settings.s1_eval_root_uri}/manifest.json"

    fs, path = fsspec.core.url_to_fs(settings.fragment_store_uri)
    if not fs.exists(path):
        sys.exit(
            f"No fragment store at {settings.fragment_store_uri} — run Phase 1's real "
            f"LINE-export ingest first (twin/PLAN.md)."
        )

    held_out = [f for f in read_fragments_jsonl(settings.fragment_store_uri) if f.split == Split.HELDOUT]
    if not held_out:
        sys.exit(
            f"0 held-out fragments found in {settings.fragment_store_uri} — run "
            f"Phase 1's real LINE-export ingest first (twin/PLAN.md)."
        )
    sampled = sample_held_out_windows(held_out, seed=SAMPLE_SEED)
    held_out_ids = [f.fragment_id for f in sampled]
    print(
        f"{len(held_out)} held-out fragments found; {len(held_out_ids)} sampled in time-stratified "
        f"windows (seed={SAMPLE_SEED}) — making one Teacher call..."
    )

    teacher = GeminiTeacher.from_settings(settings)
    bank = build_item_bank(
        held_out_fragment_ids=held_out_ids,
        teacher=teacher,
        fragment_store_uri=settings.fragment_store_uri,
    )

    breakdown = Counter(item.item_type for item in bank)
    print(f"\nGenerated {len(bank)} items:")
    for item_type in _ITEM_TYPES:
        print(f"  {item_type}: {breakdown.get(item_type, 0)}")
    print("\nEVAL.md §3.2's 30/25/25/20 ratio is a SHOULD, not enforced — eyeball it before freezing.")

    answer = input("\nFreeze this bank? Write-once, irreversible in practice. [y/N] ").strip().lower()
    if answer != "y":
        print("Not written. Re-run to generate a fresh batch (costs another Teacher call).")
        return

    manifest = S1BankManifest(
        bank_hash=bank_hash(bank),
        item_count=len(bank),
        source_fragment_dataset_hash=dataset_hash(held_out_ids),
        teacher_model=teacher.model,
        created_at=datetime.now(UTC),
    )
    write_item_bank_once(bank, bank_uri=bank_uri, manifest=manifest, manifest_uri=manifest_uri)
    print(f"\nWritten to {bank_uri} and {manifest_uri}.")
    print("Next: uv run python examples/collect_s1_answers.py --wave 1")


if __name__ == "__main__":
    main()
