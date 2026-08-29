#!/usr/bin/env python3
"""Collect Wave 1 or Wave 2 answers against the frozen S1 item bank.
EVAL.md §3.2/§3.3, twin/PLAN.md Phase 2 (wave 1) / Phase 6 (wave 2).

Wave 1's completion timestamp IS "project day 0" — the moment it's
recorded, EVAL.md §1.2's 14-day uncompressible wait to Wave 2 begins for
real. Re-run with --wave 2 exactly 14 days later, against the SAME frozen
bank (its hash is verified before a single item is shown) — never generate
a new bank for Wave 2.

Resumable: already-answered items for the given wave are skipped on
restart, so Ctrl-C or a crash mid-session loses nothing. Answers are
recorded by option number, never free text, so Phase 6's R1-vs-R2
comparison can be an exact match.

    uv run python examples/collect_s1_answers.py --wave 1
    uv run python examples/collect_s1_answers.py --wave 2
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import Literal

import fsspec

from twin.config.settings import get_settings
from twin.harness.item_bank import (
    S1WaveManifest,
    bank_hash,
    read_and_verify_item_bank,
    write_wave_manifest_once,
)
from twin.harness.schema import S1Answer


def _answers_uri(root_uri: str, wave: Literal[1, 2]) -> str:
    return f"{root_uri}/answers/r{wave}.jsonl"


def _wave_manifest_uri(root_uri: str, wave: Literal[1, 2]) -> str:
    return f"{root_uri}/answers/r{wave}_manifest.json"


def _read_existing_answers(uri: str) -> dict[str, S1Answer]:
    fs, path = fsspec.core.url_to_fs(uri)
    if not fs.exists(path):
        return {}
    answers: dict[str, S1Answer] = {}
    with fsspec.open(uri, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                answer = S1Answer.model_validate_json(stripped)
                answers[answer.item_id] = answer
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    wave: Literal[1, 2] = 1 if args.wave == 1 else 2

    root_uri = get_settings().s1_eval_root_uri
    items, _bank_manifest = read_and_verify_item_bank(
        bank_uri=f"{root_uri}/item_bank.jsonl", manifest_uri=f"{root_uri}/manifest.json"
    )

    answers_uri = _answers_uri(root_uri, wave)
    already_answered = _read_existing_answers(answers_uri)
    remaining = [item for item in items if item.item_id not in already_answered]

    if not remaining:
        print(f"All {len(items)} items already have a wave {wave} answer.")
    else:
        fs, path = fsspec.core.url_to_fs(answers_uri)
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            fs.makedirs(parent, exist_ok=True)

        for i, item in enumerate(remaining, start=1):
            options = item.options or []
            print(f"\n[{i}/{len(remaining)}] ({item.item_type}) {item.prompt}")
            for idx, option in enumerate(options, start=1):
                print(f"  {idx}. {option}")
            while True:
                raw = input("Answer #: ").strip()
                if raw.isdigit() and 1 <= int(raw) <= len(options):
                    chosen = options[int(raw) - 1]
                    break
                print("Invalid — enter the option number shown.")
            answer = S1Answer(item_id=item.item_id, wave=wave, answer=chosen, answered_at=datetime.now(UTC))
            with fsspec.open(answers_uri, "a", encoding="utf-8") as f:
                f.write(answer.model_dump_json())
                f.write("\n")

    total_answered = len(already_answered) + len(remaining)
    if total_answered != len(items):
        return

    manifest_uri = _wave_manifest_uri(root_uri, wave)
    fs, path = fsspec.core.url_to_fs(manifest_uri)
    if fs.exists(path):
        print(f"\nWave {wave} already had a completed manifest at {manifest_uri}.")
        return

    write_wave_manifest_once(
        S1WaveManifest(wave=wave, bank_hash=bank_hash(items), item_count=len(items), completed_at=datetime.now(UTC)),
        manifest_uri,
    )
    print(f"\nWave {wave} complete — manifest written to {manifest_uri}.")
    if wave == 1:
        print("This timestamp is project day 0. Wave 2 opens in 14 days (EVAL.md §1.2) — not before.")


if __name__ == "__main__":
    main()
