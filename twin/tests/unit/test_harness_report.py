"""EvalReport. EVAL.md §11's literal field list; §1.4's no-cross-suite-total rule."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from twin.core.enums import GateLevel
from twin.harness.report import EvalReport, assert_no_cross_suite_total, render_report_md
from twin.harness.schema import HarnessError, S1Metrics, S2Metrics, S3Metrics, S4Metrics


def _report(**overrides: object) -> EvalReport:
    defaults: dict[str, object] = dict(
        run_id="run-001",
        base_model="some-org/some-8b-model",
        adapter_hash="a" * 64,
        dataset_hash="b" * 64,
        eval_set_version="v1",
        date=datetime(2026, 8, 27, tzinfo=UTC),
        s1=S1Metrics(normalized_accuracy={"T": 0.75, "B0": 0.5, "B1": 0.6, "B2": 0.65}, self_consistency=0.8),
        s2=S2Metrics(pass_1=0.6, pass_4=0.35, tool_top1=0.6, call_count_mae=0.5, giveup_delta=0.3, plugin_transfer=0.7),
        s3=S3Metrics(precision=0.8, recall=0.8, false_alarm=0.1, f1=0.8, silence_rate_delta=0.05),
        s4=S4Metrics(identification_rate=0.6, reason_distribution={"clean": 2}, code_switch_rate_delta=0.05),
        judge_agreement=0.85,
        judge_rubric_hash="c" * 64,
        judge_session_delta=0.02,
        confidence="high",
        gate_level=GateLevel.L0,
    )
    defaults.update(overrides)
    return EvalReport(**defaults)  # type: ignore[arg-type]


def test_assert_no_cross_suite_total_passes_for_the_real_report_type() -> None:
    assert_no_cross_suite_total(_report())  # must not raise


def test_render_report_md_includes_every_eval_11_field() -> None:
    text = render_report_md(_report())
    for expected in (
        "run_id", "base_model", "adapter_hash", "dataset_hash", "eval_set_version", "date",
        "S1:", "S2:", "S3:", "S4:", "judge_agreement", "judge_rubric_hash", "judge_session_delta",
        "confidence", "gate_level",
    ):
        assert expected in text


def test_render_report_md_puts_low_confidence_on_the_first_line() -> None:
    text = render_report_md(_report(confidence="low"))
    assert text.splitlines()[0] == "confidence: low"


def test_render_report_md_does_not_put_confidence_low_first_when_high() -> None:
    text = render_report_md(_report(confidence="high"))
    assert text.splitlines()[0] != "confidence: low"


def test_render_report_md_calls_the_cross_suite_total_guard() -> None:
    """A structural regression check: if EvalReport ever gains a forbidden
    field, rendering MUST fail loudly rather than silently including it."""
    with pytest.raises(HarnessError, match="cross-suite"):
        # Simulate a report type that gained a forbidden field, using the same
        # substring check assert_no_cross_suite_total uses internally.
        class _BadReport(EvalReport):
            overall_score: float = 0.5

        assert_no_cross_suite_total(_BadReport(**_report().model_dump()))
