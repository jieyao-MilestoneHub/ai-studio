"""Structured questionnaire ingest. INTERVIEW.md §5 — administered after the
interview; item bank MUST be disjoint from S1's (enforced in
`harness.questionnaire_guard`, not here — see that module's docstring for
why the check can't live in this layer).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from twin.core.enums import Modality, Precision, SourceClass
from twin.core.fragment import Fragment
from twin.ingest.fragment import fragment_from_text_record


class QuestionnaireItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    prompt: str
    scale_labels: list[str]


class QuestionnaireResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    answer: str
    answered_at: datetime


def fragments_from_questionnaire(
    responses: list[QuestionnaireResponse],
    items: dict[str, QuestionnaireItem],
    *,
    principal_id: str,
    train_cutoff: datetime,
    sealed_cutoff: datetime,
) -> Iterator[Fragment]:
    """One `SourceClass.SELF_REPORT`, `Precision.MINUTE` fragment per
    response — reuses `ingest.fragment.fragment_from_text_record`, never a
    separate constructor. Big-Five/GSS-style items become ordinary fragments
    this way, also usable as a future B1 persona-paragraph source.

    Does NOT call `ingest.entities.extract_third_party_spans` — every
    fragment gets `third_party_spans=[]` (the `fragment_from_text_record`
    default). INTERVIEW.md §7 Q8's tagging requirement is scoped to
    "逐字稿" (the interview transcript), not the questionnaire, and this
    function enforces (not just assumes) that a response is verbatim one of
    its item's own `scale_labels` — there is no free text here for a third
    party to appear in. If a future item type accepts open-ended answers
    (e.g. INTERVIEW.md §5's self-estimated response-rate items, if ever made
    free-text), this exemption MUST be revisited."""
    for response in responses:
        item = items.get(response.item_id)
        if item is None:
            raise ValueError(
                f"questionnaire response references unknown item_id {response.item_id!r} "
                f"— not present in the supplied item bank"
            )
        if response.answer not in item.scale_labels:
            raise ValueError(
                f"questionnaire response for item_id {response.item_id!r} has answer "
                f"{response.answer!r}, not one of that item's scale_labels {item.scale_labels} — "
                f"answers MUST be verbatim one of the item's own scale labels (same discipline "
                f"as harness.schema.S1Answer.answer)"
            )
        content = f"{item.prompt}: {response.answer}"
        yield fragment_from_text_record(
            principal_id=principal_id,
            content=content,
            event_time=response.answered_at,
            precision=Precision.MINUTE,
            confidence=1.0,
            source_class=SourceClass.SELF_REPORT,
            modality=Modality.TEXT,
            train_cutoff=train_cutoff,
            sealed_cutoff=sealed_cutoff,
        )
