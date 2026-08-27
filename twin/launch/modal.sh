#!/usr/bin/env bash
# SPEC.md §7.1/D12: vendor SDK coupling is confined to two places, launch/*.sh
# and teacher.py. Modal's SDK is Python-first (GPU/volume/secret config is
# expressed via @app.function decorators, not CLI flags — confirmed live
# against Modal's current docs, 2026-08-27: `modal run`/`modal shell` still
# resolve to a decorated Python app object, there is no way to define a
# custom GPU job from pure CLI). This script is therefore deliberately thin:
# it does the actual "coupling" via launch/modal_app.py (which lives outside
# src/twin/, so import-linter's root_package="twin" scan never sees it), and
# this .sh file's only job is `uv sync` + invoking `modal run`.
#
# Deviates from twin/PLAN.md §3.1's tree comment ("launch/ 只放 shell") —
# see twin/PLAN.md's Phase 4 notes for why, and update that comment when this
# lands for real rather than let the tree drift silently out of sync.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync

# Modal's provisioned CUDA driver version for T4 instances needs a live check
# against Modal's current base images before this index URL is trusted —
# placeholder, verify at implementation time.
# uv pip install torch --index-url https://download.pytorch.org/whl/cu124

modal run launch/modal_app.py::train_entrypoint -- "$@"
