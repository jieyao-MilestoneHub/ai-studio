#!/usr/bin/env python3
"""Thin CLI shell. SPEC.md §7.1/§7.4 name this exact invocation:
`python train.py --resume auto`. Deliberately a bare argparse script, not a
`twin.cli` typer subcommand (twin/PLAN.md §3.4): a remote, freshly `uv sync`'d
training box needs one unambiguous entrypoint, and this script's own import
footprint should stay minimal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from twin.config.settings import get_settings
from twin.train.run import TrainingConfig
from twin.train.run import main as run_training


def main() -> None:
    # twin.config.settings.Settings reads .env for its own TWIN_* fields, but
    # that parsing is internal to pydantic-settings — it never touches
    # os.environ. checkpoint.py's fsspec/boto3 calls (R2) read raw
    # AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_ENDPOINT_URL_S3 directly
    # from the real process environment, so .env MUST also be exported here
    # or those credentials are silently invisible to everything except
    # get_settings() itself (verified against a live R2 bucket 2026-08-28:
    # this is the actual failure mode without this line).
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    parser.add_argument("--config", type=Path, required=True, help="Path to a TrainingConfig JSON file")
    parser.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=600.0,
        help="SPEC.md §7.4 SHOULD 10-15 min (600-900s). Tests use a small value.",
    )
    parser.add_argument(
        "--allow-no-self-report",
        action="store_true",
        help="skip train/preflight.py's D19 gate (self-report >= 1%% of samples). For LINE-only experiments and the toy CI run.",
    )
    args = parser.parse_args()

    config = TrainingConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    settings = get_settings()
    if settings.checkpoint_store_uri is None:
        sys.exit(
            "TWIN_CHECKPOINT_STORE_URI is not set. SPEC.md §7.2/§7.4: checkpoints "
            "MUST land at an explicit fsspec URI, never a guessed default — set it "
            "before running train.py for real."
        )
    if settings.adapter_encryption_key is None:
        sys.exit(
            "TWIN_ADAPTER_ENCRYPTION_KEY is not set. SPEC.md §8: adapters are "
            "personal data and MUST be stored encrypted — generate a key with "
            "examples/generate_adapter_encryption_key.py and set it before "
            "running train.py for real."
        )

    manifest = run_training(
        config,
        require_self_report=not args.allow_no_self_report,
        principal_id=settings.principal_id,
        trajectories_uri=settings.trajectory_store_uri,
        checkpoint_store_uri=settings.checkpoint_store_uri,
        encryption_key=settings.adapter_encryption_key.get_secret_value().encode("utf-8"),
        resume=(args.resume == "auto"),
        checkpoint_interval_seconds=args.checkpoint_interval_seconds,
    )
    print(manifest.model_dump_json())


if __name__ == "__main__":
    main()
