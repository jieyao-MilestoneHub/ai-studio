"""Stable, sortable identifiers.

Run ids are time-ordered so `ls runs/` reads chronologically. Child ids are
*derived* rather than random: the same plan produces the same scene and shot
ids, which is what lets `resume` match a re-planned run against clips already
paid for in `clips.json`.
"""

from __future__ import annotations

import hashlib
import os
import time

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _b36(value: int, width: int) -> str:
    out: list[str] = []
    while value:
        value, rem = divmod(value, 36)
        out.append(_ALPHABET[rem])
    return "".join(reversed(out)).rjust(width, "0")[-width:]


def new_run_id(prefix: str = "run") -> str:
    """A fresh, time-ordered run id, e.g. ``run_1m3k7q2p_4f9a``.

    The timestamp component sorts lexicographically in creation order; the
    random suffix keeps two runs started in the same second distinct.
    """
    stamp = _b36(int(time.time()), 8)
    salt = _b36(int.from_bytes(os.urandom(4), "big"), 4)
    return f"{prefix}_{stamp}_{salt}"


def derive_id(kind: str, *parts: object) -> str:
    """A deterministic id for a plan element.

    Same inputs always produce the same id. This is what makes a re-plan
    comparable to a previous run rather than a fresh set of strangers, so
    `resume` can reuse clips instead of regenerating them.

    >>> derive_id("scene", "run_x", 0) == derive_id("scene", "run_x", 0)
    True
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=6).hexdigest()
    return f"{kind}_{digest}"


def scene_id(run_id: str, index: int) -> str:
    return derive_id("sc", run_id, index)


def shot_id(scene: str, index: int) -> str:
    return derive_id("sh", scene, index)


def segment_id(shot: str, subcut_index: int) -> str:
    return derive_id("sg", shot, subcut_index)


def cue_id(segment: str, index: int) -> str:
    return derive_id("cue", segment, index)
