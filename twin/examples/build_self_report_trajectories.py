#!/usr/bin/env python3
"""Turn an interview transcript into training trajectories and merge them into
the trajectory store (SPEC.md D19/D37/D39, twin/PLAN.md Phase 7 前置 "T v2").
Refuses to merge the same transcript twice (matching observation+answer),
backs up the store, and rewrites `<store>.manifest.json` with the added
self-report counts so the next training run's dataset_hash change is
explained.

    uv run python examples/build_self_report_trajectories.py \\
        --transcript file:///home/me/twin-data/transcripts/interview-....json \\
        [--upload-to s3://twin-checkpoints/data/trajectories.jsonl]

--upload-to copies the merged store (and manifest) to the URI the Modal
training secret's TWIN_TRAJECTORY_STORE_URI points at — D39: effect first,
plaintext like the LINE trajectories already there. Then retrain:
    uv run modal run --detach launch/modal_app.py::train_entrypoint -- --config launch/configs/qwen3-8b-t4-r64-v1.json
(a new dataset_hash => a new run_id; nothing resumes from T v1).
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
from twin.ingest.interview_trajectories import trajectories_from_interview
from twin.ingest.interviewer import InterviewTranscript
from twin.ingest.store import read_trajectories_jsonl, write_trajectories_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--upload-to", default=None, help="fsspec URI to copy the merged store + manifest to")
    parser.add_argument("--upload-only", action="store_true", help="skip the merge (already done); just copy to --upload-to")
    args = parser.parse_args()

    load_dotenv()  # AWS_* for R2 reach the process env only this way (Settings never writes os.environ) — same as train.py
    settings = get_settings()
    uri = settings.trajectory_store_uri
    manifest_uri = f"{uri}.manifest.json"
    if args.upload_only:
        if not args.upload_to:
            sys.exit("--upload-only needs --upload-to")
        _upload(uri, manifest_uri, args.upload_to)
        return
    with fsspec.open(args.transcript, "r", encoding="utf-8") as f:
        transcript = InterviewTranscript.model_validate_json(f.read())
    new = list(trajectories_from_interview(transcript))
    if not new:
        sys.exit("transcript produced no answered questions — nothing to add")

    fs, path = fsspec.core.url_to_fs(uri)
    existing = list(read_trajectories_jsonl(uri)) if fs.exists(path) else []
    seen = {(t.observation, t.steps[0].model_dump_json()) for t in existing}
    dupes = [t for t in new if (t.observation, t.steps[0].model_dump_json()) in seen]
    if dupes:
        sys.exit(f"{len(dupes)}/{len(new)} of these trajectories are already in {uri} — refusing to add the same interview twice")

    if fs.exists(path):
        backup = f"{path}.bak-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        shutil.copyfile(path, backup) if uri.startswith("file://") else fs.copy(path, backup)
        print(f"backup: {backup}")
    total = write_trajectories_jsonl([*existing, *new], uri)

    manifest: dict = {}
    mfs, mpath = fsspec.core.url_to_fs(manifest_uri)
    if mfs.exists(mpath):
        with fsspec.open(manifest_uri, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest["written"] = total
    manifest.setdefault("by_split", {})["train"] = manifest.get("by_split", {}).get("train", 0) + len(new)
    manifest.setdefault("self_report", []).append(
        {"transcript": args.transcript, "trajectories": len(new), "added_at": datetime.now(UTC).isoformat(), "decision": "SPEC.md D37/D39"}
    )
    with fsspec.open(manifest_uri, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"added {len(new)} self-report trajectories; store now {total} (manifest updated)")

    if args.upload_to:
        _upload(uri, manifest_uri, args.upload_to)


def _upload(uri: str, manifest_uri: str, target: str) -> None:
    for src, dst in ((uri, target), (manifest_uri, f"{target}.manifest.json")):
        with fsspec.open(src, "rb") as fin, fsspec.open(dst, "wb") as fout:
            fout.write(fin.read())
        print(f"uploaded {dst}")
    print("Next: uv run modal run --detach launch/modal_app.py::train_entrypoint -- --config launch/configs/qwen3-8b-t4-r64-v1.json")


if __name__ == "__main__":
    main()
