#!/usr/bin/env python3
"""Score a prepared S1 baseline eval round after judging completes.
`eval-harness` skill steps 4-6 — script-only, run after every
`eval/out/<run_id>/shard-*.jsonl` file exists (one `eval-judge` subagent
dispatch per shard, against the paths examples/prepare_s1_eval_round.py
printed).

    uv run python examples/score_s1_eval_round.py --run-id <run_id> \\
        --judge-agreement <float> --confidence {high,low}

`--judge-agreement` and `--confidence` are NOT computed by this script —
they MUST come from a real, already-completed EVAL.md §6.3 alignment run
(30 samples, the principal's own annotation vs. the judge's, twin/PLAN.md
Phase 6, gated on the real 14-day wait — not run yet in this checkout).
This script refuses to write anything if `--judge-agreement` is below
`core.gate_metrics.JUDGE_AGREEMENT_FLOOR` (`harness.gate_check.
check_judge_agreement_floor`, unmodified — reused, not reimplemented), and
writes only through `harness.eval_io.write_report`, the one output channel
the `eval-harness` skill sanctions for the main session to read
(`eval/report/<run_id>.md`).

Earlier draft note (fixed after `spec-auditor` review): a first version of
this script computed and wrote/printed `S1Metrics` straight from real judge
verdicts before any alignment check existed, bypassing `check_judge_
agreement_floor` and writing to a parallel `_s1_metrics.json` file instead
of the sanctioned report path. That is exactly the `eval-harness` skill's
own named anti-pattern ("附警語的報告一定會被當成報告用") — a caveat gets
read once, the numbers get quoted ten times — just via a different
filename. Gating construction of the report behind a real external
`judge_agreement` input, and behind the existing floor check, closes that.
`s2`/`s3`/`s4` are left `None` on the written report (not fabricated) —
this script never calls `harness.gate_check.check_t1`/`check_t2`, which
would themselves refuse on a report missing any suite; it only asserts the
judge's own conclusions are trustworthy enough to record, not that the
twin is fit for anything.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import fsspec

from twin.config.settings import get_settings
from twin.core.enums import GateLevel
from twin.harness.aggregate import aggregate_s1
from twin.harness.eval_io import read_judged_shard, write_report
from twin.harness.gate_check import check_judge_agreement_floor
from twin.harness.manifest import RunManifest
from twin.harness.report import EvalReport
from twin.harness.s1_run import (
    compute_self_consistency,
    read_sample_index,
    regroup_judged_items_by_baseline,
)
from twin.harness.schema import JudgedItem, S1Answer

UNJUDGEABLE_VERDICT = "unjudgeable"


def _read_answers(uri: str) -> list[S1Answer]:
    with fsspec.open(uri, "r", encoding="utf-8") as f:
        return [S1Answer.model_validate_json(line) for line in f if line.strip()]


def _read_manifest(runs_root: str, run_id: str) -> RunManifest:
    with fsspec.open(f"{runs_root}/{run_id}.json", "r", encoding="utf-8") as f:
        return RunManifest.model_validate_json(f.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--judge-agreement",
        type=float,
        required=True,
        help="Real result of EVAL.md §6.3's 30-sample alignment run against this "
        "rubric — computed elsewhere (Phase 6), never by this script.",
    )
    parser.add_argument("--confidence", choices=["high", "low"], required=True)
    args = parser.parse_args()
    run_id: str = args.run_id

    s1_root = get_settings().s1_eval_root_uri.rstrip("/")
    eval_root = s1_root.rsplit("/", 1)[0]  # same derivation as prepare_s1_eval_round.py
    runs_root = f"{eval_root}/runs"
    manifest = _read_manifest(runs_root, run_id)
    index = read_sample_index(f"{runs_root}/{run_id}_index.jsonl")

    judged: list[JudgedItem] = []
    unjudgeable_total = 0
    low_confidence_total = 0
    flag_counts: dict[str, int] = {}
    for shard_uri in manifest.shard_ids:
        out_uri = shard_uri.replace("/in/", "/out/")
        fs, path = fsspec.core.url_to_fs(out_uri)
        if not fs.exists(path):
            sys.exit(
                f"Missing judged shard {out_uri} — dispatch an eval-judge subagent "
                f"against every shard in {manifest.shard_ids} first."
            )
        shard_judged, quality = read_judged_shard(out_uri, unjudgeable_verdict=UNJUDGEABLE_VERDICT)
        judged.extend(shard_judged)
        unjudgeable_total += quality.unjudgeable
        low_confidence_total += quality.low_confidence
        for flag, count in quality.flag_counts.items():
            flag_counts[flag] = flag_counts.get(flag, 0) + count

    # Pre-flight diagnostics about the judge run's own mechanical health —
    # NOT a persona-fidelity conclusion about the twin, so printing this
    # ahead of the alignment gate below is not the thing spec-auditor
    # flagged.
    print(f"Judged: {len(judged)} total, {unjudgeable_total} unjudgeable, {low_confidence_total} low-confidence.")
    if flag_counts:
        print(f"Flags: {flag_counts}")

    judged_by_baseline = regroup_judged_items_by_baseline(judged, index)

    r1 = _read_answers(f"{s1_root}/answers/r1.jsonl")
    r2 = _read_answers(f"{s1_root}/answers/r2.jsonl")
    self_consistency = compute_self_consistency(r1, r2)

    s1_metrics = aggregate_s1(judged_by_baseline=judged_by_baseline, self_consistency=self_consistency)

    report = EvalReport(
        run_id=run_id,
        base_model=manifest.base_model,
        adapter_hash=manifest.adapter_hash,
        dataset_hash=manifest.dataset_hash,
        eval_set_version=manifest.eval_set_version,
        date=datetime.now(UTC),
        s1=s1_metrics,
        judge_agreement=args.judge_agreement,
        judge_rubric_hash=manifest.rubric_hash,
        confidence=args.confidence,
        gate_level=GateLevel.L0,  # this script never raises the gate; only agent.gate does, on separate criteria
    )

    # Raises and refuses to write anything below EVAL.md §6.3's 0.80 floor —
    # this is the actual gate, not a comment promising one.
    check_judge_agreement_floor(report)

    out_uri = f"{eval_root}/report/{run_id}.md"
    write_report(report, out_uri)
    print(f"\nWritten to {out_uri}.")


if __name__ == "__main__":
    main()
