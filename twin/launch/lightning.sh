#!/usr/bin/env bash
# SPEC.md §7.3: Lightning AI is the explicitly named backup compute tier (lowest
# priority of the three — Modal is primary, Kaggle is for long runs). Unlike
# Modal/Kaggle, a Lightning AI Studio looks closer to "provision a devbox, SSH
# or exec in, run a command" than a serverless-function-with-decorators model,
# so this one may genuinely be pure-shell-drivable via the `lightning` CLI —
# but the exact 2026-08 CLI verb set was NOT verified against live docs before
# writing this (lowest priority of the three, deliberately left for last).
# VERIFY the `lightning` CLI's actual current subcommands before relying on
# this script for a real run — treat everything below as a placeholder shape,
# not a tested command.
#
# SPEC.md §7.3 "MUST 優先使用 spot/preemptible": whether Lightning AI Studios
# offer a spot/preemptible option at all was NOT verified before writing this
# (lowest-priority of the three platforms, deliberately left for last — see
# above). Flag this explicitly alongside the CLI-verb uncertainty rather than
# silently assuming either way.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync

# Placeholder — confirm the actual current `lightning` CLI verb for "run a
# command on a Studio with a GPU attached" before using this for real.
lightning run studio --name twin-train --gpu T4 -- \
  bash -c "cd $(pwd) && uv sync && python train.py --resume auto \"\$@\""
