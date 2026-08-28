"""S1 run assembly: self-consistency, bundled judge samples, and reattaching
judge verdicts to their originating baseline after blinding strips it.
EVAL.md §3.2-§3.4, §6.1.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from twin.harness.s1_run import (
    S1SampleIndexEntry,
    build_s1_raw_samples,
    compute_self_consistency,
    read_sample_index,
    regroup_judged_items_by_baseline,
    write_sample_index,
)
from twin.harness.schema import HarnessError, JudgedItem, S1Answer, S1Item


def _item(item_id: str) -> S1Item:
    return S1Item(
        item_id=item_id,
        item_type="preference",
        prompt=f"prompt-{item_id}",
        options=["a", "b"],
        source_fragment_ids=["frag-1"],
    )


def _answer(item_id: str, answer: str, wave: int) -> S1Answer:
    return S1Answer(item_id=item_id, wave=wave, answer=answer, answered_at=datetime(2026, 8, 28, tzinfo=UTC))  # type: ignore[arg-type]


class TestComputeSelfConsistency:
    def test_is_pure_exact_match(self) -> None:
        r1 = [_answer("i1", "a", 1), _answer("i2", "b", 1)]
        r2 = [_answer("i1", "a", 2), _answer("i2", "a", 2)]
        assert compute_self_consistency(r1, r2) == 0.5

    def test_ignores_items_missing_from_either_wave(self) -> None:
        r1 = [_answer("i1", "a", 1), _answer("i2", "b", 1)]
        r2 = [_answer("i1", "a", 2)]  # i2 never answered in wave 2
        assert compute_self_consistency(r1, r2) == 1.0

    def test_rejects_no_shared_items(self) -> None:
        with pytest.raises(HarnessError, match="share no item_ids"):
            compute_self_consistency([_answer("i1", "a", 1)], [_answer("i2", "a", 2)])


class TestBuildS1RawSamples:
    def test_produces_one_sample_per_item_per_baseline(self) -> None:
        items = [_item("i1"), _item("i2")]
        r2_answers = {"i1": _answer("i1", "a", 2), "i2": _answer("i2", "b", 2)}
        candidate_answers = {"B0": {"i1": "x", "i2": "y"}, "B1": {"i1": "x2", "i2": "y2"}}

        samples, index = build_s1_raw_samples(items=items, r2_answers=r2_answers, candidate_answers=candidate_answers)

        assert len(samples) == 4
        assert len(index) == 4
        assert {(entry.item_id, entry.baseline) for entry in index} == {
            ("i1", "B0"),
            ("i2", "B0"),
            ("i1", "B1"),
            ("i2", "B1"),
        }

    def test_rejects_missing_r2_answer(self) -> None:
        items = [_item("i1")]
        with pytest.raises(HarnessError, match="no R2 answer"):
            build_s1_raw_samples(items=items, r2_answers={}, candidate_answers={"B0": {"i1": "x"}})

    def test_rejects_missing_candidate_answer(self) -> None:
        items = [_item("i1")]
        r2_answers = {"i1": _answer("i1", "a", 2)}
        with pytest.raises(HarnessError, match="no B0 candidate answer"):
            build_s1_raw_samples(items=items, r2_answers=r2_answers, candidate_answers={"B0": {}})


class TestRegroupJudgedItemsByBaseline:
    def test_round_trips_from_build_s1_raw_samples(self) -> None:
        items = [_item("i1"), _item("i2")]
        r2_answers = {"i1": _answer("i1", "a", 2), "i2": _answer("i2", "b", 2)}
        candidate_answers = {"B0": {"i1": "x", "i2": "y"}, "B1": {"i1": "x2", "i2": "y2"}}
        samples, index = build_s1_raw_samples(items=items, r2_answers=r2_answers, candidate_answers=candidate_answers)

        judged = [JudgedItem(sample_id=sample.sample_id, verdict="match", rationale="") for sample in samples]
        grouped = regroup_judged_items_by_baseline(judged, index)

        assert set(grouped) == {"B0", "B1"}
        assert len(grouped["B0"]) == 2
        assert len(grouped["B1"]) == 2

    def test_raises_on_a_judged_sample_id_missing_from_the_index(self) -> None:
        index = [S1SampleIndexEntry(sample_id="a", item_id="i1", baseline="B0")]
        judged = [JudgedItem(sample_id="a", verdict="match", rationale=""), JudgedItem(sample_id="not-in-index", verdict="match", rationale="")]
        with pytest.raises(HarnessError, match="index"):
            regroup_judged_items_by_baseline(judged, index)

    def test_raises_when_an_index_entry_was_never_judged(self) -> None:
        index = [
            S1SampleIndexEntry(sample_id="a", item_id="i1", baseline="B0"),
            S1SampleIndexEntry(sample_id="b", item_id="i1", baseline="B1"),
        ]
        judged = [JudgedItem(sample_id="a", verdict="match", rationale="")]
        with pytest.raises(HarnessError, match="never judged"):
            regroup_judged_items_by_baseline(judged, index)


def test_sample_index_round_trips_through_disk(tmp_path: Path) -> None:
    index = [
        S1SampleIndexEntry(sample_id="a", item_id="i1", baseline="B0"),
        S1SampleIndexEntry(sample_id="b", item_id="i1", baseline="T"),
    ]
    uri = f"file://{tmp_path}/index.jsonl"
    write_sample_index(index, uri)
    assert read_sample_index(uri) == index
