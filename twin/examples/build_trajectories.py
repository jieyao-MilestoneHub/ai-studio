#!/usr/bin/env python3
"""Build Phase 4's slim trajectory set from the LINE exports (SPEC.md §4.10,
twin/PLAN.md Phase 4 "仍待辦 2") and write it to TWIN_TRAJECTORY_STORE_URI.

Uses the SAME manifest and the SAME cutoffs as examples/ingest_line_export.py
(pass them explicitly; they are written next to the store as
`<store>.manifest.json` so a mismatch with the fragment store is visible).
Prints corpus statistics only — never message content.

    uv run python examples/build_trajectories.py \\
        --manifest ~/twin-data/raw/manifest.json \\
        --principal-display-name "本人顯示名稱" \\
        --train-cutoff 2026-03-01 --sealed-cutoff 2026-07-24T14:23:12 \\
        --reply-window-min 120
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import fsspec

from twin.config.settings import get_settings
from twin.core.enums import NegativeClass, Split
from twin.core.trajectory import NoActionStep
from twin.ingest.sources.line import parse_line_export
from twin.ingest.store import write_trajectories_jsonl
from twin.ingest.trajectories import TrajectoryBuildParams, trajectories_from_line_messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--principal-display-name", required=True)
    parser.add_argument("--train-cutoff", type=datetime.fromisoformat, required=True)
    parser.add_argument("--sealed-cutoff", type=datetime.fromisoformat, required=True)
    parser.add_argument("--reply-window-min", type=int, default=120)
    parser.add_argument("--burst-gap-min", type=int, default=5)
    parser.add_argument("--context-messages", type=int, default=8)
    parser.add_argument("--exposure-horizon-h", type=int, default=24)
    parser.add_argument("--late-reply", choices=["no_action", "skip"], default="no_action")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    uri = settings.trajectory_store_uri
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path) and not args.overwrite:
        sys.exit(f"refused: trajectory store already exists at {uri} (pass --overwrite to replace it)")

    params = TrajectoryBuildParams(
        train_cutoff=args.train_cutoff,
        sealed_cutoff=args.sealed_cutoff,
        burst_gap=timedelta(minutes=args.burst_gap_min),
        reply_window=timedelta(minutes=args.reply_window_min),
        context_messages=args.context_messages,
        exposure_horizon=timedelta(hours=args.exposure_horizon_h),
        late_reply=args.late_reply,
    )
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    trajectories = []
    for e in entries:
        messages = parse_line_export(
            Path(e["path"]).expanduser().read_text(encoding="utf-8"),
            known_senders=list(e["senders"]),
            principal_display_name=args.principal_display_name,
        )
        trajectories.extend(
            trajectories_from_line_messages(
                list(messages),
                principal_id=settings.principal_id,
                principal_display_name=args.principal_display_name,
                params=params,
            )
        )
    trajectories.sort(key=lambda t: t.context_time)

    by_split = Counter(t.split for t in trajectories)
    no_action = [t for t in trajectories if any(isinstance(s, NoActionStep) for s in t.steps)]
    neg = Counter(t.negative_class for t in no_action)
    train_no_action = [t for t in no_action if t.split == Split.TRAIN]
    train_neg = Counter(t.negative_class for t in train_no_action)
    train_total = by_split[Split.TRAIN]
    if train_no_action and train_neg[NegativeClass.HARD] / len(train_no_action) < 0.5:
        sys.exit(
            "refused: SPEC.md §4.11 — trivial negatives would be the majority of train negatives "
            f"(hard {train_neg[NegativeClass.HARD]} / {len(train_no_action)}); nothing written. "
            "Exposure capture, not a label tweak, is the fix."
        )
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    written = write_trajectories_jsonl(trajectories, uri)

    stats = {
        "written": written,
        "by_split": {s.value: by_split.get(s, 0) for s in Split},
        "no_action_total": len(no_action),
        "no_action_by_class": {c.value: neg.get(c, 0) for c in (NegativeClass.HARD, NegativeClass.TRIVIAL)},
        "train_silence_rate": round(len(train_no_action) / train_total, 4) if train_total else None,
        "train_hard_share_of_negatives": round(train_neg[NegativeClass.HARD] / len(train_no_action), 4) if train_no_action else None,
        "params": {
            "train_cutoff": args.train_cutoff.isoformat(),
            "sealed_cutoff": args.sealed_cutoff.isoformat(),
            "reply_window_min": args.reply_window_min,
            "burst_gap_min": args.burst_gap_min,
            "context_messages": args.context_messages,
            "exposure_horizon_h": args.exposure_horizon_h,
            "late_reply": args.late_reply,
        },
        "exposure_note": "LINE export has no read receipt (SPEC.md §11 item H); hard negatives here rest on "
        "inferred exposure at the lower confidence SPEC.md §4.3 grants historical data. "
        "MUST NOT be used as an S3 evaluation source.",
        "built_at": datetime.now(UTC).isoformat(),
    }
    with fsspec.open(f"{uri}.manifest.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
