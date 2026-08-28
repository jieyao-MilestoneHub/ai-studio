"""INTERVIEW.md §5: the questionnaire item bank MUST be disjoint from S1's."""

from __future__ import annotations

import pytest

from twin.harness.questionnaire_guard import assert_disjoint_from_s1_item_bank
from twin.harness.schema import HarnessError, S1Item
from twin.ingest.sources.questionnaire import QuestionnaireItem


def _s1_item(prompt: str) -> S1Item:
    return S1Item(item_id=prompt, item_type="preference", prompt=prompt, options=["a", "b"], source_fragment_ids=["f1"])


def _questionnaire_item(prompt: str) -> QuestionnaireItem:
    return QuestionnaireItem(item_id=prompt, prompt=prompt, scale_labels=["低", "高"])


def test_assert_disjoint_passes_when_no_overlap() -> None:
    assert_disjoint_from_s1_item_bank([_questionnaire_item("q-only")], [_s1_item("s1-only")])


def test_assert_disjoint_raises_on_a_shared_prompt() -> None:
    with pytest.raises(HarnessError, match="disjoint"):
        assert_disjoint_from_s1_item_bank([_questionnaire_item("shared prompt")], [_s1_item("shared prompt")])
