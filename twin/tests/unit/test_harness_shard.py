"""Shard splitting. eval-harness skill step 2."""

from __future__ import annotations

import pytest

from twin.harness.schema import RawEvalSample
from twin.harness.shard import sample_id, split_into_shards, strip_source_label


def test_sample_id_is_deterministic_and_content_derived() -> None:
    assert sample_id("hello") == sample_id("hello")
    assert sample_id("hello") != sample_id("world")


def test_strip_source_label_removes_the_label() -> None:
    raw = RawEvalSample(sample_id="abc", source_label="twin", content="hi", suite="s1")
    stripped = strip_source_label(raw)
    assert stripped.content == "hi"
    assert not hasattr(stripped, "source_label")


def test_split_into_shards_respects_shard_size() -> None:
    samples = [strip_source_label(RawEvalSample(sample_id=str(i), source_label="twin", content="x", suite="s1")) for i in range(5)]
    shards = split_into_shards(samples, shard_size=2)
    assert [len(s) for s in shards] == [2, 2, 1]


def test_split_into_shards_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        split_into_shards([], shard_size=0)
