"""Shard splitting. eval-harness skill step 2:

- source labels MUST be physically stripped in this script, not left to a
  prompt telling the judge to ignore them.
- sample ids MUST be content-hash derived, never a sequence number — a
  sequence number can't survive a reorder, and EVAL.md §6.4's cross-session
  item-by-item comparison depends on stable ids.
- shard size MUST come from config, never from remaining context budget — it
  feeds `rubric_hash`, and a size that drifts makes that hash meaningless.
"""

from __future__ import annotations

import hashlib

from twin.harness.schema import RawEvalSample, StrippedSample


def sample_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def strip_source_label(sample: RawEvalSample) -> StrippedSample:
    return StrippedSample(sample_id=sample.sample_id, content=sample.content, suite=sample.suite)


def split_into_shards(samples: list[StrippedSample], *, shard_size: int) -> list[list[StrippedSample]]:
    if shard_size <= 0:
        raise ValueError(f"shard_size MUST be positive, got {shard_size}")
    return [samples[i : i + shard_size] for i in range(0, len(samples), shard_size)]
