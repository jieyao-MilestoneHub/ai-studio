"""The fitness gate. EVAL.md §7.1's T1/T2 tables, literal; §3.4's kill switch."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from twin.core.enums import GateLevel
from twin.harness.gate_check import check_judge_agreement_floor, check_t1, check_t2
from twin.harness.report import EvalReport
from twin.harness.schema import HarnessError, S1Metrics, S2Metrics, S3Metrics, S4Metrics


def _report(**overrides: object) -> EvalReport:
    defaults: dict[str, object] = dict(
        run_id="run-001",
        base_model="m",
        adapter_hash="a" * 64,
        dataset_hash="b" * 64,
        eval_set_version="v1",
        date=datetime(2026, 8, 27, tzinfo=UTC),
        s1=S1Metrics(normalized_accuracy={"T": 0.72, "B0": 0.4, "B1": 0.5, "B2": 0.6}, self_consistency=0.8),
        s2=S2Metrics(pass_1=0.61, pass_4=0.36, tool_top1=0.61, call_count_mae=0.5, giveup_delta=0.3, plugin_transfer=0.7),
        s3=S3Metrics(precision=0.7, recall=0.7, false_alarm=0.25, f1=0.56, silence_rate_delta=0.14),
        s4=S4Metrics(identification_rate=0.7, reason_distribution={}, code_switch_rate_delta=0.14),
        judge_agreement=0.82,
        judge_rubric_hash="c" * 64,
        confidence="high",
        gate_level=GateLevel.L0,
    )
    defaults.update(overrides)
    return EvalReport(**defaults)  # type: ignore[arg-type]


class TestCheckT1:
    def test_passes_when_every_threshold_clears(self) -> None:
        result = check_t1(_report())
        assert result.passed is True
        assert result.failures == []

    def test_fails_when_s1_below_absolute_threshold_even_if_above_b2(self) -> None:
        report = _report(s1=S1Metrics(normalized_accuracy={"T": 0.65, "B0": 0.3, "B1": 0.4, "B2": 0.5}, self_consistency=0.8))
        result = check_t1(report)
        assert result.passed is False
        assert any("normalized_accuracy" in f for f in result.failures)

    def test_fails_the_kill_switch_when_s1_does_not_exceed_b2_even_if_above_absolute_threshold(self) -> None:
        """EVAL.md §3.4: T >= 0.70 alone is not enough — it MUST also beat B2."""
        report = _report(s1=S1Metrics(normalized_accuracy={"T": 0.75, "B0": 0.4, "B1": 0.5, "B2": 0.80}, self_consistency=0.8))
        result = check_t1(report)
        assert result.passed is False
        assert any("kill switch" in f for f in result.failures)

    def test_fails_when_s3_false_alarm_too_high(self) -> None:
        report = _report(s3=S3Metrics(precision=0.5, recall=0.5, false_alarm=0.5, f1=0.5, silence_rate_delta=0.1))
        assert check_t1(report).passed is False

    def test_raises_on_a_partial_report(self) -> None:
        report = _report(s3=None)
        with pytest.raises(HarnessError, match="partial report"):
            check_t1(report)


class TestCheckT2:
    def test_t2_is_strictly_harder_than_t1(self) -> None:
        """A report that clears T1 by a small margin should not automatically clear T2."""
        report = _report()  # tuned to just clear T1 above
        assert check_t1(report).passed is True
        assert check_t2(report).passed is False


def test_check_judge_agreement_floor_raises_below_080() -> None:
    with pytest.raises(HarnessError, match=r"0\.8"):
        check_judge_agreement_floor(_report(judge_agreement=0.79))


def test_check_judge_agreement_floor_passes_at_exactly_080() -> None:
    check_judge_agreement_floor(_report(judge_agreement=0.80))  # must not raise
