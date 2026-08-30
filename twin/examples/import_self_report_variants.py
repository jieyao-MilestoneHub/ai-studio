#!/usr/bin/env python3
"""Merge Claude-Code-written paraphrases of the interview trajectories into the
trajectory store as `teacher_synthesized` (SPEC.md §4.10/D25; twin/PLAN.md
Phase 4 待辦 4). Input: one or more JSONL files of
`{"trajectory_id", "question", "answer"}` whose trajectory_id is an observed
interview trajectory already in the store.

    uv run python examples/import_self_report_variants.py \\
        ~/twin-data/transcripts/variants-*.jsonl [--upload-to s3://twin-checkpoints/data/trajectories.jsonl]

Refuses a variant whose answer is empty, identical to the original, or already
in the store; prints a per-source count so a human can see the balance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import fsspec
from dotenv import load_dotenv

from twin.config.settings import get_settings
from twin.core.enums import GroundTruthSource
from twin.core.trajectory import ActionStep
from twin.ingest.interview_augment import variant_trajectory
from twin.ingest.interview_trajectories import INTERVIEW_SURFACE
from twin.ingest.store import read_trajectories_jsonl, write_trajectories_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--upload-to", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    uri = get_settings().trajectory_store_uri
    existing = list(read_trajectories_jsonl(uri))
    interview = [t for t in existing if isinstance(t.steps[0], ActionStep) and t.steps[0].surface == INTERVIEW_SURFACE]
    sources = {t.trajectory_id: t for t in interview if t.ground_truth_source == GroundTruthSource.OBSERVED}
    seen = {t.steps[0].content for t in interview}  # type: ignore[union-attr]

    built = []
    per_source: Counter[str] = Counter()
    rejected = 0
    for file in args.files:
        for line in file.expanduser().read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source = sources.get(row["trajectory_id"])
            if source is None:
                sys.exit(f"{file}: unknown trajectory_id {row['trajectory_id']}")
            try:
                t = variant_trajectory(source, question=row["question"], answer=row["answer"])
            except ValueError:
                rejected += 1
                continue
            content = t.steps[0].content  # type: ignore[union-attr]
            if content in seen:
                rejected += 1
                continue
            seen.add(content)
            built.append(t)
            per_source[source.trajectory_id[:8]] += 1
    print(f"{len(built)} variants accepted, {rejected} rejected (empty/identical/duplicate)")
    print("per source:", dict(per_source))
    if args.dry_run or not built:
        return

    fs, path = fsspec.core.url_to_fs(uri)
    backup = f"{path}.bak-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    shutil.copyfile(path, backup) if uri.startswith("file://") else fs.copy(path, backup)
    total = write_trajectories_jsonl([*existing, *built], uri)
    manifest_uri = f"{uri}.manifest.json"
    with fsspec.open(manifest_uri, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["written"] = total
    manifest["by_split"]["train"] = manifest["by_split"].get("train", 0) + len(built)
    manifest.setdefault("self_report_augmented", []).append(
        {
            "sources": len(per_source),
            "variants": len(built),
            "teacher_model": "claude-code (interviewer session, file-based)",
            "ground_truth_source": "teacher_synthesized",
            "files": [str(f) for f in args.files],
            "added_at": datetime.now(UTC).isoformat(),
        }
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
