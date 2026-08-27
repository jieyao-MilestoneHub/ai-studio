"""Deterministic hashes for reproducibility. SPEC.md §7.5: a `run_id` MUST
bind seed, `dataset_hash`, and `config_hash`. data-contract skill rule 8: the
computation MUST be fixed and centralized here, and MUST NOT take a
timestamp, file mtime, or dict-iteration-order as input — any of those would
make "the same data" hash differently between runs, silently invalidating
every historical comparison SPEC.md §7.5 depends on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

import fsspec


def dataset_hash(record_ids: Iterable[str]) -> str:
    """Sorted before hashing — a dataset built from the same records in a
    different order MUST hash identically."""
    joined = "\n".join(sorted(record_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Canonical JSON (sorted keys) — dict insertion order MUST NOT affect the hash."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adapter_hash(adapter_uri: str) -> str:
    """Hashes the adapter artifact's actual bytes via fsspec, so re-uploading
    identical weights hashes identically regardless of storage backend."""
    hasher = hashlib.sha256()
    with fsspec.open(adapter_uri, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
