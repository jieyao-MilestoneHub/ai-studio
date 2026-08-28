"""Interview transcript post-processing. INTERVIEW.md §6.2's four ordered
steps.
"""

from __future__ import annotations

import pytest

from twin.ingest.postprocess import (
    apply_correction_glossary,
    mark_unclear_spans,
    run_postprocessing_pipeline,
)


def test_apply_correction_glossary_replaces_known_mis_transcriptions() -> None:
    text = "我昨天去找小梅吃飯"
    corrected = apply_correction_glossary(text, glossary={"小梅": "小美"})
    assert corrected == "我昨天去找小美吃飯"


def test_apply_correction_glossary_leaves_fillers_and_repetition_untouched() -> None:
    text = "然後然後我就，呃，去了"
    corrected = apply_correction_glossary(text, glossary={"小梅": "小美"})
    assert corrected == text


def test_mark_unclear_spans_inserts_markers_at_given_ranges() -> None:
    text = "我昨天去找XXX吃飯"
    marked = mark_unclear_spans(text, low_confidence_ranges=[(5, 8)])
    assert marked == "我昨天去找[unclear]吃飯"


def test_mark_unclear_spans_returns_text_unchanged_when_no_ranges_given() -> None:
    text = "完全清楚的一句話"
    assert mark_unclear_spans(text, low_confidence_ranges=[]) == text


def test_mark_unclear_spans_rejects_an_out_of_bounds_range() -> None:
    with pytest.raises(ValueError, match="invalid low_confidence_range"):
        mark_unclear_spans("short", low_confidence_ranges=[(0, 100)])


def test_run_postprocessing_pipeline_is_idempotent_given_the_same_glossary() -> None:
    raw = "我昨天去找小梅吃飯"
    glossary = {"小梅": "小美"}
    first = run_postprocessing_pipeline(raw, correction_glossary=glossary, low_confidence_ranges=[])
    second = run_postprocessing_pipeline(raw, correction_glossary=glossary, low_confidence_ranges=[])
    assert first == second == "我昨天去找小美吃飯"


def test_run_postprocessing_pipeline_reruns_cleanly_after_a_glossary_update() -> None:
    raw = "我昨天去找小梅吃飯"
    before = run_postprocessing_pipeline(raw, correction_glossary={}, low_confidence_ranges=[])
    after = run_postprocessing_pipeline(raw, correction_glossary={"小梅": "小美"}, low_confidence_ranges=[])
    assert before == "我昨天去找小梅吃飯"
    assert after == "我昨天去找小美吃飯"
    assert before != after
