#!/usr/bin/env python3
"""Prepare one S1 baseline eval round: EVAL.md §3.4's B0/B1/B2 answers against
the frozen item bank + real R2 (Wave 2) answers, stripped and sharded ready
for judging. `eval-harness` skill steps 0-2 — script-only. Step 3 (dispatch
one `eval-judge` subagent per shard) can only be done by the orchestrating
Claude Code session, not a script: this script ends by printing the shard and
rubric paths for that dispatch. After judging, run
examples/score_s1_eval_round.py.

    uv run python examples/prepare_s1_eval_round.py

Requires (all real, none stubbed):
- A frozen item bank + completed Wave 2 (R2) answers (twin/PLAN.md Phase 2/6).
- A real S1 backend (examples/run_baseline_inference.py's HFBaselineBackend,
  GPU-only) and, for B1/B2, a persona paragraph (`file://./data/persona.txt`)
  and self-report transcript fragments (twin/PLAN.md Phase 3-A). None of
  these exist yet in this checkout as of 2026-08-28 — B1/B2 are skipped with
  an actionable message rather than a bare traceback until they do; B0 alone
  can still run once the item bank + R2 answers exist.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from uuid import uuid4

import fsspec

# examples/ is not a package (see twin/PLAN.md §3.1) — this relies on Python
# putting the running script's own directory on sys.path, true for every
# `uv run python examples/<script>.py` invocation this repo's examples/
# docstrings document, but NOT a general-purpose import path.
from run_baseline_inference import HFBaselineBackend

from twin.config.settings import get_settings
from twin.core.enums import SourceClass, Split
from twin.harness.baseline import BaselineId, generate_baseline_samples
from twin.harness.eval_io import compute_rubric_hash, write_shards
from twin.harness.item_bank import bank_hash, read_and_verify_item_bank
from twin.harness.manifest import RunManifest, find_existing_run, write_manifest_once
from twin.harness.s1_run import BaselineKey, build_s1_raw_samples, write_sample_index
from twin.harness.schema import S1Answer
from twin.harness.shard import split_into_shards, strip_source_label
from twin.ingest.store import read_fragments_jsonl
from twin.train.model import DEFAULT_MODEL_SPEC

RUBRIC_URI = "file://./eval/rubric/s1.md"
PERSONA_URI = "file://./data/persona.txt"
EVAL_ROOT_URI = "file://./eval"
RUNS_ROOT_URI = "file://./eval/runs"

SHARD_SIZE = 20
EVAL_SET_VERSION = "s1-v1"  # bump when S1's methodology (not the item bank content) changes
BASELINES: tuple[BaselineId, ...] = ("B0", "B1", "B2")


def _read_r2_answers(uri: str) -> dict[str, S1Answer]:
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


def _read_text(uri: str) -> str | None:
    fs, path = fsspec.core.url_to_fs(uri)
    if not fs.exists(path):
        return None
    with fsspec.open(uri, "r", encoding="utf-8") as f:
        return f.read()


def _load_self_report_transcript(fragment_store_uri: str) -> str | None:
    """Concatenates every `Split.HELDOUT` `SourceClass.SELF_REPORT`
    fragment's content, in `event_time.value` order — this is B2's
    transcript-injection context. Deliberately excludes `Split.SEALED`:
    this script prepares a routine eval round, not the one-time final
    acceptance run, and EVAL.md §9 reserves the sealed slice for that
    ("MUST only unseal at final acceptance") — B0/B1's item-bank/R2 paths
    are already hardcoded to `split="heldout"` for the same reason, this
    just applies it here too, which the first version of this function
    missed (caught by `spec-auditor`). Real content only exists once
    twin/PLAN.md Phase 3-A's interview ingest has actually run against a
    real transcript."""
    fs, path = fsspec.core.url_to_fs(fragment_store_uri)
    if not fs.exists(path):
        return None
    fragments = [
        f
        for f in read_fragments_jsonl(fragment_store_uri)
        if f.source_class == SourceClass.SELF_REPORT and f.split == Split.HELDOUT
    ]
    if not fragments:
        return None
    fragments.sort(key=lambda f: f.event_time.value)
    return "\n\n".join(f.content for f in fragments)


def main() -> None:
    settings = get_settings()
    s1_root = get_settings().s1_eval_root_uri
    items, _bank_manifest = read_and_verify_item_bank(
        bank_uri=f"{s1_root}/item_bank.jsonl", manifest_uri=f"{s1_root}/manifest.json"
    )
    r2_answers_uri = f"{s1_root}/answers/r2.jsonl"

    r2_answers = _read_r2_answers(r2_answers_uri)
    missing_r2 = [item.item_id for item in items if item.item_id not in r2_answers]
    if missing_r2:
        sys.exit(
            f"{len(missing_r2)}/{len(items)} items have no R2 (Wave 2) answer at {r2_answers_uri} — "
            f"run examples/collect_s1_answers.py --wave 2 to completion first (twin/PLAN.md Phase 6)."
        )

    dataset_hash_value = bank_hash(items)
    rubric_hash_value = compute_rubric_hash(RUBRIC_URI, shard_size=SHARD_SIZE)
    existing = find_existing_run(
        dataset_hash=dataset_hash_value, rubric_hash=rubric_hash_value, split="heldout", runs_root=RUNS_ROOT_URI
    )
    if existing is not None:
        sys.exit(
            f"A run with this exact (dataset_hash, rubric_hash, split) already exists: "
            f"{existing.run_id} (EVAL.md §12 anti-pattern 10 — re-running because a score "
            f"wasn't welcome is not a valid reason). If the rubric genuinely changed, that "
            f"changes rubric_hash and this check would not have fired — use the existing run."
        )

    persona_text = _read_text(PERSONA_URI)
    transcript_text = _load_self_report_transcript(settings.fragment_store_uri)

    backend = HFBaselineBackend()

    candidate_answers: dict[BaselineKey, dict[str, str]] = {}
    for baseline in BASELINES:
        if baseline == "B1" and persona_text is None:
            print(f"Skipping B1 — no persona paragraph at {PERSONA_URI}.")
            continue
        if baseline == "B2" and transcript_text is None:
            print("Skipping B2 — no self-report transcript fragments found yet (twin/PLAN.md Phase 3-A).")
            continue
        samples = generate_baseline_samples(
            items=items,
            baseline=baseline,
            backend=backend,
            persona_text=persona_text,
            transcript_text=transcript_text,
        )
        candidate_answers[baseline] = dict(
            zip((item.item_id for item in items), (sample.content for sample in samples), strict=True)
        )

    if not candidate_answers:
        sys.exit("No baseline produced any answers — nothing to prepare.")

    raw_samples, index = build_s1_raw_samples(items=items, r2_answers=r2_answers, candidate_answers=candidate_answers)
    stripped = [strip_source_label(sample) for sample in raw_samples]
    shards = split_into_shards(stripped, shard_size=SHARD_SIZE)

    run_id = f"s1-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    shard_uris = write_shards(shards, run_id=run_id, root_uri=EVAL_ROOT_URI)
    write_sample_index(index, f"{RUNS_ROOT_URI}/{run_id}_index.jsonl")

    manifest = RunManifest(
        run_id=run_id,
        base_model=f"{DEFAULT_MODEL_SPEC.base_model_id}@{DEFAULT_MODEL_SPEC.base_model_revision}",
        adapter_hash="none",  # baseline-only round — no LoRA adapter involved (B0/B1/B2, never T)
        dataset_hash=dataset_hash_value,
        eval_set_version=EVAL_SET_VERSION,
        rubric_hash=rubric_hash_value,
        date=datetime.now(UTC),
        split="heldout",
        shard_ids=shard_uris,
    )
    write_manifest_once(manifest, f"{RUNS_ROOT_URI}/{run_id}.json")

    print(f"\nPrepared run {run_id}: {len(stripped)} samples across {len(shard_uris)} shard(s).")
    print(f"Baselines included: {', '.join(sorted(candidate_answers))}.")
    print("\nDispatch one eval-judge subagent per shard, giving it exactly these three paths:")
    print(f"  rubric: {RUBRIC_URI}")
    for shard_uri in shard_uris:
        out_uri = shard_uri.replace("/in/", "/out/")
        print(f"  shard:  {shard_uri}\n  out:    {out_uri}")
    print(f"\nThen run: uv run python examples/score_s1_eval_round.py --run-id {run_id}")


if __name__ == "__main__":
    main()
