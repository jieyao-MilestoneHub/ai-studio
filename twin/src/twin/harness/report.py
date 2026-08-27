"""The eval report. EVAL.md §11's field list, transcribed literally — MUST
NOT add or rename fields. EVAL.md §1.4: MUST NOT produce a cross-suite
weighted total; `assert_no_cross_suite_total` makes that a script-level
assertion (eval-harness skill step 5), not a comment reminding someone not to
add one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from twin.core.enums import GateLevel
from twin.harness.schema import HarnessError, S1Metrics, S2Metrics, S3Metrics, S4Metrics

_FORBIDDEN_FIELD_SUBSTRINGS = ("overall", "total_score", "weighted", "combined_score", "aggregate_score")


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    base_model: str
    adapter_hash: str
    dataset_hash: str
    eval_set_version: str
    date: datetime
    s1: S1Metrics | None = None
    s2: S2Metrics | None = None
    s3: S3Metrics | None = None
    s4: S4Metrics | None = None
    judge_agreement: float
    judge_rubric_hash: str
    judge_session_delta: float | None = None
    confidence: Literal["high", "low"]
    gate_level: GateLevel


def assert_no_cross_suite_total(report: EvalReport) -> None:
    """EVAL.md §1.4/§12 anti-pattern 9. Checked against the model's actual
    declared fields, not just against this report instance — the failure
    mode this guards is someone adding such a field to `EvalReport` later,
    not this particular value carrying one today."""
    forbidden = [
        name for name in type(report).model_fields if any(s in name.lower() for s in _FORBIDDEN_FIELD_SUBSTRINGS)
    ]
    if forbidden:
        raise HarnessError(
            f"EvalReport MUST NOT carry a cross-suite total field (EVAL.md "
            f"§1.4): found {forbidden}. S1-S4 measure abilities that trade "
            f"off by design; a weighted total makes attribution impossible."
        )


def render_report_md(report: EvalReport) -> str:
    """Field-for-field transcription of EVAL.md §11 — MUST NOT add or rename
    fields relative to that list."""
    assert_no_cross_suite_total(report)

    lines: list[str] = []
    if report.confidence == "low":
        # eval-harness skill step 6: MUST be the first line.
        lines.append("confidence: low")

    lines.append(
        f"run_id: {report.run_id}, base_model: {report.base_model}, "
        f"adapter_hash: {report.adapter_hash}, dataset_hash: {report.dataset_hash}, "
        f"eval_set_version: {report.eval_set_version}, date: {report.date.isoformat()}"
    )
    if report.s1 is not None:
        lines.append(f"S1: normalized_accuracy={report.s1.normalized_accuracy}, self_consistency={report.s1.self_consistency}")
    if report.s2 is not None:
        lines.append(
            f"S2: pass^1={report.s2.pass_1}, pass^4={report.s2.pass_4}, tool_top1={report.s2.tool_top1}, "
            f"call_count_mae={report.s2.call_count_mae}, giveup_delta={report.s2.giveup_delta}, "
            f"plugin_transfer={report.s2.plugin_transfer}"
        )
    if report.s3 is not None:
        lines.append(
            f"S3: precision={report.s3.precision}, recall={report.s3.recall}, "
            f"false_alarm={report.s3.false_alarm}, f1={report.s3.f1}, "
            f"silence_rate_delta={report.s3.silence_rate_delta}"
        )
    if report.s4 is not None:
        lines.append(
            f"S4: 辨識率={report.s4.identification_rate}, 理由分布={report.s4.reason_distribution}, "
            f"code_switch_rate_delta={report.s4.code_switch_rate_delta}"
        )
    lines.append(
        f"judge_agreement: {report.judge_agreement}, judge_rubric_hash: {report.judge_rubric_hash}, "
        f"judge_session_delta: {report.judge_session_delta}, confidence: {report.confidence}"
    )
    lines.append(f"gate_level: {report.gate_level.value}")
    return "\n".join(lines)
