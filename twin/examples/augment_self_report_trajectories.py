#!/usr/bin/env python3
"""Paraphrase every observed interview trajectory K ways in the principal's
own register (twin.ingest.interview_augment; SPEC.md §5.2/D9, D25) and merge
the variants into the trajectory store as `teacher_synthesized`. User
decision 2026-08-30: rewrite, never repeat — repetition teaches recitation.

    uv run python examples/augment_self_report_trajectories.py --variants 14 \\
        [--upload-to s3://twin-checkpoints/data/trajectories.jsonl] [--dry-run]

Costs one Teacher call per interview trajectory (~30). --dry-run prints the
first few variants and writes nothing — read them before merging: the prompt
forbids new facts, the Teacher may still invent; C1's failure symptom is the
twin quoting names/dates it was never told.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime

import fsspec
from dotenv import load_dotenv

from twin.config.settings import get_settings
from twin.core.enums import GroundTruthSource
from twin.core.trajectory import ActionStep
from twin.ingest.interview_augment import augment_interview_trajectories
from twin.ingest.interview_trajectories import INTERVIEW_SURFACE
from twin.ingest.store import read_trajectories_jsonl, write_trajectories_jsonl
from twin.teacher.gemini import GeminiTeacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", type=int, default=14, help="paraphrases per interview trajectory")
    parser.add_argument("--upload-to", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    settings = get_settings()
    uri = settings.trajectory_store_uri
    existing = list(read_trajectories_jsonl(uri))
    sources = [
        t
        for t in existing
        if t.ground_truth_source == GroundTruthSource.OBSERVED
        and isinstance(t.steps[0], ActionStep)
        and t.steps[0].surface == INTERVIEW_SURFACE
    ]
    if not sources:
        sys.exit("no observed interview trajectories in the store — run build_self_report_trajectories.py first")
    already = {t.steps[0].content for t in existing if isinstance(t.steps[0], ActionStep) and t.steps[0].surface == INTERVIEW_SURFACE}

    # Register samples: the principal's own LINE replies (short, casual) — the
    # voice the variants must keep. Interview answers themselves are the source
    # text, so they are not reused as style samples.
    line_replies = [
        t.steps[0].content
        for t in existing
        if isinstance(t.steps[0], ActionStep) and t.steps[0].surface == "line" and 12 <= len(t.steps[0].content) <= 60
    ]
    style_samples = line_replies[:: max(1, len(line_replies) // 6)][:6]

    teacher = GeminiTeacher.from_settings(settings)
    print(f"{len(sources)} interview trajectories -> up to {args.variants} variants each ({len(sources)} Teacher calls)")
    variants = [
        v
        for v in augment_interview_trajectories(
            sources, teacher=teacher, variants_per_trajectory=args.variants, style_samples=style_samples
        )
        if v.steps[0].content not in already  # type: ignore[union-attr]
    ]
    print(f"{len(variants)} variants produced")
    if args.dry_run:
        for v in variants[:8]:
            print(f"\nQ: {v.exposure.stimulus}\nA: {v.steps[0].content}")  # type: ignore[union-attr]
        return

    fs, path = fsspec.core.url_to_fs(uri)
    backup = f"{path}.bak-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    shutil.copyfile(path, backup) if uri.startswith("file://") else fs.copy(path, backup)
    total = write_trajectories_jsonl([*existing, *variants], uri)
    manifest_uri = f"{uri}.manifest.json"
    with fsspec.open(manifest_uri, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["written"] = total
    manifest["by_split"]["train"] = manifest["by_split"].get("train", 0) + len(variants)
    manifest.setdefault("self_report_augmented", []).append(
        {"sources": len(sources), "variants": len(variants), "variants_per": args.variants, "teacher_model": teacher.model,
         "ground_truth_source": "teacher_synthesized", "added_at": datetime.now(UTC).isoformat()}
    )
    with fsspec.open(manifest_uri, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"backup: {backup}\nstore now {total}")
    if args.upload_to:
        for src, dst in ((uri, args.upload_to), (manifest_uri, f"{args.upload_to}.manifest.json")):
            with fsspec.open(src, "rb") as fin, fsspec.open(dst, "wb") as fout:
                fout.write(fin.read())
            print(f"uploaded {dst}")


if __name__ == "__main__":
    main()
