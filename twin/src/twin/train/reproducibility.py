"""SPEC.md §7.5: a `run_id` MUST bind `seed`, `dataset_hash`, `config_hash`.
Both hashes are already computed by `core.hashing` — this module's only job
is the binding itself and RNG seeding, not re-deriving either hash.
"""

from __future__ import annotations

import hashlib

from transformers import set_seed as _hf_set_seed


def derive_run_id(*, seed: int, dataset_hash: str, config_hash: str) -> str:
    """Deterministic, not time-ordered (unlike `ai_studio.core.ids.new_run_id`
    — reimplemented here per twin/PLAN.md §3.3's "沿用做法，重新實作" rule,
    not shared code). The same `(seed, dataset_hash, config_hash)` always
    produces the same `run_id`, which is what lets `--resume auto` rediscover
    the right checkpoint path on a brand-new machine without persisting
    `run_id` anywhere else first, and makes "same `run_id`" and SPEC.md
    §7.5's "MUST NOT compare runs without matching hashes" the same fact by
    construction rather than two things that could drift apart.

    Open item, not silently decided: two genuinely independent training
    attempts launched with identical seed/dataset_hash/config_hash collide on
    `run_id` and write to the same checkpoint path — by this design that IS
    the same run (a restart), not two runs. A deliberate second replicate
    requires varying at least the seed. SPEC.md §7.5 doesn't decide either
    way whether that's acceptable; confirm before this is relied on for
    anything beyond Phase 4's single-run kill switch comparison.
    """
    payload = f"{seed}\x1f{dataset_hash}\x1f{config_hash}".encode()
    return "run_" + hashlib.sha256(payload).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    """Thin wrapper over `transformers.set_seed` (python `random`, numpy,
    torch CPU+CUDA). Covers only the *initial* seed — mid-run RNG state
    capture/restore across a kill/resume cycle is Trainer/Accelerate's own
    checkpoint mechanism (`twin.train.checkpoint`), not this function's job.
    """
    _hf_set_seed(seed)
