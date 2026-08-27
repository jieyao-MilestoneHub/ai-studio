"""Kaggle kernel entrypoint. SPEC.md §7.1/D12: vendor-specific glue confined
to launch/*.sh (and its companion files, same reasoning as launch/modal_app.py)
and teacher.py — this file never imports anything from `twin.train`, it only
shells out to the same `train.py --resume auto` every other launch path uses.

OPEN ITEM, not resolved here (verify at implementation time): getting the
`twin` package source itself onto the kernel. A Kaggle script kernel uploads
only `code_file` (this file) — it does NOT bundle the rest of the repo the
way `launch/modal_app.py`'s `add_local_dir` does for Modal. Two real options,
neither committed to yet:
  (a) attach the repo as a Kaggle Dataset and read it from
      /kaggle/input/<dataset-slug>/, refreshed before each kernel run;
  (b) `git clone` a real remote URL, read from a Kaggle Secret
      (`kaggle_secrets.UserSecretsClient`) rather than hardcoded here — no
      remote URL is guessed or hardcoded in this file for that reason.
This script assumes (b) via the `TWIN_GIT_REMOTE` Kaggle Secret; set up (a)
instead and rewrite the block below if that turns out more reliable.
"""

from __future__ import annotations

import os
import subprocess
import sys

TWIN_DIR = "/kaggle/working/twin"

DEPENDENCIES = [
    "torch>=2.4",
    "transformers>=5.16",
    "trl>=1.12",
    "peft>=0.20",
    "accelerate>=1.14",
    "bitsandbytes>=0.50",
    "datasets>=3.0",
    "fsspec>=2024.6.1",
    "pydantic>=2.13.4",
    "pydantic-settings>=2.15.0",
    "google-genai>=2.20.0",
    "typer>=0.12",
    "rich>=13.7",
]


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPENDENCIES], check=True)

    if not os.path.isdir(TWIN_DIR):
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        remote = UserSecretsClient().get_secret("TWIN_GIT_REMOTE")
        subprocess.run(["git", "clone", "--depth", "1", remote, TWIN_DIR], check=True)

    subprocess.run(
        [sys.executable, "train.py", "--resume", "auto", "--config", "config.json"],
        cwd=TWIN_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
