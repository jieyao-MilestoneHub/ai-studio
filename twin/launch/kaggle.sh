#!/usr/bin/env bash
# SPEC.md §7.1/D12: vendor SDK coupling confined to launch/*.sh and teacher.py.
# `kaggle kernels push` needs an actual code artifact + kernel-metadata.json,
# not pure CLI flags — same shape of constraint as Modal (see launch/modal.sh's
# comment). This script's job is narrow: turn twin's own uv.lock into a
# requirements.txt Kaggle's kernel environment can actually install from (no
# evidence, checked 2026-08-27, that Kaggle's kernel runner reads
# pyproject.toml/uv.lock directly — VERIFY at implementation time, this is the
# safe/portable fallback regardless), then push.
#
# Deviates from twin/PLAN.md §3.1's tree comment ("launch/ 只放 shell") for the
# same reason launch/modal.sh does — see that file and twin/PLAN.md's Phase 4
# notes.
#
# SPEC.md §7.3 "MUST 優先使用 spot/preemptible" does not apply here: Kaggle's
# free-tier GPU sessions are a weekly time quota (30h/week), not a spot-
# pricing/preemption model at all — checked live 2026-08-27, no such knob
# exists on this platform to set. This is a documented inapplicability, not a
# gap: `train.py --resume auto` still matters here for Kaggle's own hard
# per-session wall-clock cutoff, which is a different reason to need resume,
# not the same one spot preemption gives Modal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv export --format requirements.txt --no-hashes > launch/requirements.txt

kaggle kernels push -p launch/
