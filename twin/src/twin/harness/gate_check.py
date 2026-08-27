"""The fitness gate: "is the twin usable at all?" EVAL.md §7.1's T1/T2
tables, literal — distinct from the *send* gate in `agent.gate` ("should the
send-gate level change?", EVAL.md §7.2). They consume the same shape of
numbers but answer different questions; do not merge them.

Every number here is already a literal in EVAL.md — there is no undetermined
design choice left in the comparison logic, only in the inputs (a real eval
round, which doesn't exist yet).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from twin.core.gate_metrics import JUDGE_AGREEMENT_FLOOR
from twin.harness.report import EvalReport
from twin.harness.schema import HarnessError


@dataclass(frozen=True)
class FitnessThresholds:
    s1_min: float
    s1_must_exceed_b2: bool
    s2a_pass1_min: float
    s2a_pass4_min: float
    s2b_top1_min: float
    s3_false_alarm_max: float
    s3_f1_min: float
    s3_silence_rate_delta_max: float
    s4_identification_rate_max: float
    s4_code_switch_rate_delta_max: float


T1 = FitnessThresholds(
    s1_min=0.70,
    s1_must_exceed_b2=True,  # EVAL.md §3.4: the kill switch — T MUST beat B2, not just clear an absolute bar
    s2a_pass1_min=0.60,
    s2a_pass4_min=0.35,
    s2b_top1_min=0.60,
    s3_false_alarm_max=0.30,
    s3_f1_min=0.55,
    s3_silence_rate_delta_max=0.15,
    s4_identification_rate_max=0.75,
    s4_code_switch_rate_delta_max=0.15,
)
T2 = FitnessThresholds(
    s1_min=0.78,
    s1_must_exceed_b2=False,  # EVAL.md §7.1's T2 row does not repeat the ">B2" condition — not invented here
    s2a_pass1_min=0.70,
    s2a_pass4_min=0.50,
    s2b_top1_min=0.72,
    s3_false_alarm_max=0.20,
    s3_f1_min=0.65,
    s3_silence_rate_delta_max=0.08,
    s4_identification_rate_max=0.65,
    s4_code_switch_rate_delta_max=0.08,
)


class FitnessCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    failures: list[str]


def check_judge_agreement_floor(report: EvalReport) -> None:
    """EVAL.md §6.3: below 0.80, judge conclusions MUST NOT be trusted at
    all. eval-harness skill step 6: harness MUST abort and refuse to emit a
    report, MUST NOT emit one with a caveat attached — a caveat gets read
    once, the numbers get quoted ten times."""
    if report.judge_agreement < JUDGE_AGREEMENT_FLOOR:
        raise HarnessError(
            f"judge_agreement {report.judge_agreement} is below the {JUDGE_AGREEMENT_FLOOR} "
            f"floor (EVAL.md §6.3) — this round's conclusions MUST NOT be trusted; "
            f"rewrite the rubric and re-run before any gate decision is made"
        )


def _check(report: EvalReport, thresholds: FitnessThresholds) -> FitnessCheckResult:
    check_judge_agreement_floor(report)

    if report.s1 is None or report.s2 is None or report.s3 is None or report.s4 is None:
        raise HarnessError(
            "EVAL.md §10.1: MUST NOT judge fitness (T1/T2) on a partial report — "
            "S1-S4 MUST all be present, exactly to prevent 'S1 improved, ship it' "
            "(anti-pattern 8) without looking at S3."
        )

    failures: list[str] = []
    s1 = report.s1.normalized_accuracy
    if "T" not in s1 or "B2" not in s1:
        raise HarnessError("S1Metrics.normalized_accuracy is missing the 'T' or 'B2' baseline")

    if s1["T"] < thresholds.s1_min:
        failures.append(f"S1 normalized_accuracy[T]={s1['T']} < {thresholds.s1_min}")
    if thresholds.s1_must_exceed_b2 and not (s1["T"] > s1["B2"]):
        failures.append(
            f"S1 T ({s1['T']}) does not exceed B2 baseline ({s1['B2']}) — "
            f"EVAL.md §3.4's kill switch: if T doesn't beat B2, the LoRA shouldn't exist"
        )
    if report.s2.pass_1 < thresholds.s2a_pass1_min:
        failures.append(f"S2a pass^1={report.s2.pass_1} < {thresholds.s2a_pass1_min}")
    if report.s2.pass_4 < thresholds.s2a_pass4_min:
        failures.append(f"S2a pass^4={report.s2.pass_4} < {thresholds.s2a_pass4_min}")
    if report.s2.tool_top1 < thresholds.s2b_top1_min:
        failures.append(f"S2b tool_top1={report.s2.tool_top1} < {thresholds.s2b_top1_min}")
    if report.s3.false_alarm > thresholds.s3_false_alarm_max:
        failures.append(f"S3 false_alarm={report.s3.false_alarm} > {thresholds.s3_false_alarm_max}")
    if report.s3.f1 < thresholds.s3_f1_min:
        failures.append(f"S3 f1={report.s3.f1} < {thresholds.s3_f1_min}")
    if report.s3.silence_rate_delta > thresholds.s3_silence_rate_delta_max:
        failures.append(f"S3 silence_rate_delta={report.s3.silence_rate_delta} > {thresholds.s3_silence_rate_delta_max}")
    if report.s4.identification_rate > thresholds.s4_identification_rate_max:
        failures.append(f"S4 identification_rate={report.s4.identification_rate} > {thresholds.s4_identification_rate_max}")
    if report.s4.code_switch_rate_delta > thresholds.s4_code_switch_rate_delta_max:
        failures.append(
            f"S4 code_switch_rate_delta={report.s4.code_switch_rate_delta} > {thresholds.s4_code_switch_rate_delta_max}"
        )

    return FitnessCheckResult(passed=not failures, failures=failures)


def check_t1(report: EvalReport) -> FitnessCheckResult:
    return _check(report, T1)


def check_t2(report: EvalReport) -> FitnessCheckResult:
    return _check(report, T2)
