"""Structured questionnaire ingest. INTERVIEW.md §5."""

from __future__ import annotations

from datetime import datetime

import pytest

from twin.core.enums import Precision, SourceClass
from twin.ingest.sources.questionnaire import (
    QuestionnaireItem,
    QuestionnaireResponse,
    fragments_from_questionnaire,
)


def _item(item_id: str) -> QuestionnaireItem:
    return QuestionnaireItem(item_id=item_id, prompt=f"prompt-{item_id}", scale_labels=["低", "中", "高"])


def _response(item_id: str, answer: str = "中") -> QuestionnaireResponse:
    return QuestionnaireResponse(item_id=item_id, answer=answer, answered_at=datetime(2026, 8, 28, 12, 0))


def test_fragments_from_questionnaire_one_fragment_per_response() -> None:
    responses = [_response("q1"), _response("q2", answer="高")]
    items = {"q1": _item("q1"), "q2": _item("q2")}

    fragments = list(
        fragments_from_questionnaire(responses, items, principal_id="p1")
    )

    assert len(fragments) == 2
    assert fragments[0].content == "prompt-q1: 中"
    assert fragments[1].content == "prompt-q2: 高"


def test_fragments_are_self_report_precision_minute() -> None:
    responses = [_response("q1")]
    items = {"q1": _item("q1")}

    fragments = list(
        fragments_from_questionnaire(responses, items, principal_id="p1")
    )

    assert fragments[0].source_class == SourceClass.SELF_REPORT
    assert fragments[0].event_time.precision == Precision.MINUTE


def test_fragments_from_questionnaire_rejects_a_response_with_no_matching_item() -> None:
    responses = [_response("q-missing")]

    with pytest.raises(ValueError, match="item_id"):
        list(
            fragments_from_questionnaire(
                responses, {}, principal_id="p1"
            )
        )


def test_fragments_from_questionnaire_rejects_an_answer_outside_the_scale_labels() -> None:
    responses = [_response("q1", answer="不在量表上的自由文字")]
    items = {"q1": _item("q1")}

    with pytest.raises(ValueError, match="scale_labels"):
        list(
            fragments_from_questionnaire(
                responses, items, principal_id="p1"
            )
        )
