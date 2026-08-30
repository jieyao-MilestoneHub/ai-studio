#!/usr/bin/env python3
"""Prepare one S1 eval round: the candidate answers already generated on a GPU
(examples/generate_s1_candidates.py via `modal run launch/modal_app.py::
s1_candidates`, files under <s1_root>/candidates/) paired with the frozen
item bank + real R2 (Wave 2) answers, stripped and sharded ready for
judging. CPU only. `eval-harness` skill steps 0-2 — script-only. Step 3 (dispatch
one `eval-judge` subagent per shard) can only be done by the orchestrating
Claude Code session, not a script: this script ends by printing the shard and
rubric paths for that dispatch. After judging, run
examples/score_s1_eval_round.py.

    uv run python examples/prepare_s1_eval_round.py

Requires (all real, none stubbed):
- A frozen item bank + completed Wave 2 (R2) answers (twin/PLAN.md Phase 2/6).
- At least one candidates file (B0/B1/B2/T) under <s1_root>/candidates/.
Every path derives from TWIN_S1_EVAL_ROOT_URI (eval root = its parent), never
from the checkout.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from uuid import uuid4

import fsspec

from twin.config.settings import get_settings
from twin.harness.eval_io import compute_rubric_hash, rubric_uri, write_shards
from twin.harness.item_bank import bank_hash, read_and_verify_item_bank
from twin.harness.manifest import RunManifest, find_existing_run, write_manifest_once
from twin.harness.s1_run import (
    BaselineKey,
    build_s1_raw_samples,
    read_candidates,
    write_sample_index,
)
from twin.harness.schema import S1Answer
from twin.harness.shard import split_into_shards, strip_source_label
from twin.train.model import DEFAULT_MODEL_SPEC

SHARD_SIZE = 20
EVAL_SET_VERSION = "s1-v1"  # bump when S1's methodology (not the item bank content) changes
LABELS: tuple[BaselineKey, ...] = ("B0", "B1", "B2", "T")


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


def main() -> None:
    s1_root = get_settings().s1_eval_root_uri.rstrip("/")
    eval_root = s1_root.rsplit("/", 1)[0]
    runs_root = f"{eval_root}/runs"
    rubric = rubric_uri("s1")
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
    rubric_hash_value = compute_rubric_hash(rubric, shard_size=SHARD_SIZE)
    existing = find_existing_run(
        dataset_hash=dataset_hash_value, rubric_hash=rubric_hash_value, split="heldout", runs_root=runs_root
    )
    if existing is not None:
        sys.exit(
            f"A run with this exact (dataset_hash, rubric_hash, split) already exists: "
            f"{existing.run_id} (EVAL.md §12 anti-pattern 10 — re-running because a score "
            f"wasn't welcome is not a valid reason). If the rubric genuinely changed, that "
            f"changes rubric_hash and this check would not have fired — use the existing run."
        )

    candidate_answers: dict[BaselineKey, dict[str, str]] = {}
    adapter_hash = "none"
    for label in LABELS:
        uri = f"{s1_root}/candidates/{label}.jsonl"
        fs, path = fsspec.core.url_to_fs(uri)
        if not fs.exists(path):
            print(f"No {label} candidates at {uri} — skipping {label}.")
            continue
        candidates = read_candidates(uri)
        if label == "T" and any("(no recall)" in c.model for c in candidates):
            # EVAL.md §3.4: T = LoRA + memory store. An adapter answering without
            # recall() is not T by definition (twin/PLAN.md Phase 5 (b)) — refuse
            # rather than let a mislabelled system reach the kill switch.
            sys.exit(f"{uri} was generated without recall() — not T under EVAL.md §3.4; not preparing a round with it.")
        by_item = {c.item_id: c.content for c in candidates}
        missing = [item.item_id for item in items if item.item_id not in by_item]
        if missing:
            sys.exit(f"{label} candidates cover {len(by_item)}/{len(items)} items — regenerate {label} before preparing a round.")
        candidate_answers[label] = by_item
        if label == "T":
            adapter_hash = candidates[0].adapter_hash  # a real content hash (SPEC.md §7.5), not the URI

    if not candidate_answers:
        sys.exit("No baseline produced any answers — nothing to prepare.")
    if "T" in candidate_answers and "B2" not in candidate_answers:
        # EVAL.md §3.4 / §12 anti-pattern 4: T's only meaningful opponent is B2.
        sys.exit("T candidates present but no B2 — a T round without B2 is anti-pattern #4 (EVAL.md §12); generate B2 first.")

    raw_samples, index = build_s1_raw_samples(items=items, r2_answers=r2_answers, candidate_answers=candidate_answers)
    stripped = [strip_source_label(sample) for sample in raw_samples]
    shards = split_into_shards(stripped, shard_size=SHARD_SIZE)

    run_id = f"s1-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    shard_uris = write_shards(shards, run_id=run_id, root_uri=eval_root)
    write_sample_index(index, f"{runs_root}/{run_id}_index.jsonl")

    manifest = RunManifest(
        run_id=run_id,
        base_model=f"{DEFAULT_MODEL_SPEC.base_model_id}@{DEFAULT_MODEL_SPEC.base_model_revision}",
        adapter_hash=adapter_hash,  # "none" for a baseline-only round
        dataset_hash=dataset_hash_value,
        eval_set_version=EVAL_SET_VERSION,
        rubric_hash=rubric_hash_value,
        date=datetime.now(UTC),
        split="heldout",
        shard_ids=shard_uris,
    )
    write_manifest_once(manifest, f"{runs_root}/{run_id}.json")

    print(f"\nPrepared run {run_id}: {len(stripped)} samples across {len(shard_uris)} shard(s).")
    print(f"Baselines included: {', '.join(sorted(candidate_answers))}.")
    print("\nDispatch one eval-judge subagent per shard, giving it exactly these three paths:")
    print(f"  rubric: {rubric}")
    for shard_uri in shard_uris:
        out_uri = shard_uri.replace("/in/", "/out/")
        print(f"  shard:  {shard_uri}\n  out:    {out_uri}")
    print(f"\nThen run: uv run python examples/score_s1_eval_round.py --run-id {run_id}")


if __name__ == "__main__":
    main()
