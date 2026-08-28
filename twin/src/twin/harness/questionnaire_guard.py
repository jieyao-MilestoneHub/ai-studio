"""INTERVIEW.md §5: the questionnaire item bank MUST be disjoint from S1's.

Lives in `twin.harness`, not `twin.ingest`, even though its subject matter is
an ingest-time concern: a disjointness check needs to see both
`ingest.sources.questionnaire.QuestionnaireItem` and `harness.schema.S1Item`,
but `twin.ingest` sits *below* `twin.harness` in the import-linter layer
spine, and "Eval harness stays a leaf" explicitly forbids `twin.ingest`
importing `twin.harness`. `twin.harness` *is* permitted to import
`twin.ingest` (harness sits above ingest in the spine), so the check belongs
here — the same direction-of-dependency reasoning as `core.adapter.
AdapterManifest`/`core.gate_metrics.GateMetrics`, just resolved by picking
the higher layer instead of a `core` projection, since both types involved
are already harness/ingest-specific rather than cross-cutting.
"""

from __future__ import annotations

from twin.harness.schema import HarnessError, S1Item
from twin.ingest.sources.questionnaire import QuestionnaireItem


def assert_disjoint_from_s1_item_bank(questionnaire_items: list[QuestionnaireItem], s1_bank: list[S1Item]) -> None:
    s1_prompts = {item.prompt for item in s1_bank}
    questionnaire_prompts = {item.prompt for item in questionnaire_items}
    overlap = sorted(questionnaire_prompts & s1_prompts)
    if overlap:
        raise HarnessError(
            f"INTERVIEW.md §5: the questionnaire item bank MUST be disjoint from the S1 "
            f"item bank — {len(overlap)} shared prompt(s): {overlap[:5]}"
        )
