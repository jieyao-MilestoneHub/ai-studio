"""Score aggregation. EVAL.md §3.3/§4.3/§5.3/§8.2's formulas, literal."""

from __future__ import annotations

import pytest

from twin.harness.aggregate import aggregate_s1, aggregate_s2, aggregate_s3, aggregate_s4, pass_at_k
from twin.harness.schema import JudgedItem


def _judged(*verdicts: str) -> list[JudgedItem]:
    return [JudgedItem(sample_id=str(i), verdict=v, rationale="") for i, v in enumerate(verdicts)]


class TestAggregateS1:
    def test_normalizes_by_self_consistency(self) -> None:
        metrics = aggregate_s1(
            judged_by_baseline={"T": _judged("match", "match", "no_match", "match")},
            self_consistency=0.8,
        )
        assert metrics.normalized_accuracy["T"] == pytest.approx(0.75 / 0.8)
        assert metrics.self_consistency == 0.8

    def test_rejects_zero_self_consistency(self) -> None:
        with pytest.raises(ValueError, match="self_consistency is 0"):
            aggregate_s1(judged_by_baseline={"T": _judged("match")}, self_consistency=0.0)


class TestPassAtK:
    def test_pass_at_1_is_simple_success_rate(self) -> None:
        assert pass_at_k([[True], [False], [True], [True]]) == 0.75

    def test_pass_at_k_requires_all_runs_to_succeed(self) -> None:
        assert pass_at_k([[True, True, True, True], [True, True, True, False]]) == 0.5


class TestAggregateS3:
    def test_matches_the_literal_confusion_matrix_formulas(self) -> None:
        metrics = aggregate_s3(
            true_positive=8, false_positive=2, false_negative=2, true_negative=18, silence_rate_delta=0.05
        )
        assert metrics.precision == pytest.approx(8 / 10)
        assert metrics.recall == pytest.approx(8 / 10)
        assert metrics.false_alarm == pytest.approx(2 / 20)
        assert metrics.f1 == pytest.approx(2 * 0.8 * 0.8 / (0.8 + 0.8))
        assert metrics.silence_rate_delta == 0.05

    def test_handles_zero_denominators_without_dividing_by_zero(self) -> None:
        metrics = aggregate_s3(true_positive=0, false_positive=0, false_negative=0, true_negative=0, silence_rate_delta=0.0)
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.false_alarm == 0.0
        assert metrics.f1 == 0.0


class TestAggregateS4:
    def test_identification_rate_and_reason_distribution(self) -> None:
        metrics = aggregate_s4(
            correct_identifications=[True, True, False, True],
            reasons=["too clean", "too clean", "vague", "too clean"],
            code_switch_rate_delta=0.03,
        )
        assert metrics.identification_rate == 0.75
        assert metrics.reason_distribution == {"too clean": 3, "vague": 1}
        assert metrics.code_switch_rate_delta == 0.03


def test_aggregate_s2_combines_all_sub_metrics() -> None:
    metrics = aggregate_s2(
        task_runs_k1=[[True], [False]],
        task_runs_k4=[[True, True, True, True], [True, False, True, True]],
        tool_top1_matches=[True, True, False],
        call_count_errors=[1.0, -1.0, 2.0],
        giveup_point_errors=[0.0, 1.0],
        plugin_transfer_score=0.6,
    )
    assert metrics.pass_1 == 0.5
    assert metrics.pass_4 == 0.5
    assert metrics.tool_top1 == pytest.approx(2 / 3)
    assert metrics.call_count_mae == pytest.approx((1 + 1 + 2) / 3)
    assert metrics.giveup_delta == pytest.approx(0.5)
    assert metrics.plugin_transfer == 0.6
