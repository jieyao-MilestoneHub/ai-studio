"""Score aggregation. EVAL.md §3.3/§4.3/§5.3/§8.2's formulas, transcribed
literally — scores MUST be computed by a script, never spoken or estimated
by the judge (EVAL.md §6.2 point 4). Each function here takes already-judged
or already-measured per-item values; producing those values (running the
suite, calling the judge) is `harness.suites.*`'s job, deferred to each
suite's own phase.
"""

from __future__ import annotations

from typing import Literal

from twin.harness.schema import JudgedItem, S1Metrics, S2Metrics, S3Metrics, S4Metrics


def _agreement_rate(judged: list[JudgedItem], *, agree_verdicts: frozenset[str]) -> float:
    """Fraction of judged items whose verdict counts as agreement. Which
    verdict strings count is rubric-defined (the rubric doesn't exist yet),
    hence the parameter rather than a hardcoded string."""
    if not judged:
        raise ValueError("no judged items to aggregate")
    matches = sum(1 for item in judged if item.verdict in agree_verdicts)
    return matches / len(judged)


def aggregate_s1(
    *,
    judged_by_baseline: dict[Literal["T", "B0", "B1", "B2"], list[JudgedItem]],
    self_consistency: float,
    agree_verdicts: frozenset[str] = frozenset({"match"}),
) -> S1Metrics:
    """EVAL.md §3.3: normalized_accuracy = raw_accuracy / self_consistency,
    per baseline. `self_consistency = agreement(R1, R2)` is computed
    upstream (EVAL.md §3.2/§3.3) — it is the S1 denominator this project's
    single hardest calendar constraint (the 14-day wait) exists to produce."""
    if self_consistency == 0:
        raise ValueError("self_consistency is 0 — normalized_accuracy is undefined (division by zero)")
    normalized = {
        baseline: _agreement_rate(items, agree_verdicts=agree_verdicts) / self_consistency
        for baseline, items in judged_by_baseline.items()
    }
    return S1Metrics(normalized_accuracy=normalized, self_consistency=self_consistency)


def pass_at_k(task_run_results: list[list[bool]]) -> float:
    """EVAL.md §4.3: "pass^k = 同一任務獨立跑 k 次全部成功的比例." Each inner list
    is one task's k independent run outcomes."""
    if not task_run_results:
        raise ValueError("no task results to aggregate")
    all_passed = sum(1 for runs in task_run_results if runs and all(runs))
    return all_passed / len(task_run_results)


def _rate(matches: list[bool]) -> float:
    if not matches:
        raise ValueError("no values to aggregate")
    return sum(matches) / len(matches)


def _mean_absolute_error(errors: list[float]) -> float:
    if not errors:
        raise ValueError("no values to aggregate")
    return sum(abs(e) for e in errors) / len(errors)


def aggregate_s2(
    *,
    task_runs_k1: list[list[bool]],
    task_runs_k4: list[list[bool]],
    tool_top1_matches: list[bool],
    call_count_errors: list[float],
    giveup_point_errors: list[float],
    plugin_transfer_score: float,
) -> S2Metrics:
    """EVAL.md §4.3/§4.4. `plugin_transfer_score` is passed through as-is:
    EVAL.md §4.4 states a SHOULD-consistency judgment, not a formula, so
    there is nothing here to aggregate from raw parts yet."""
    return S2Metrics(
        pass_1=pass_at_k(task_runs_k1),
        pass_4=pass_at_k(task_runs_k4),
        tool_top1=_rate(tool_top1_matches),
        call_count_mae=_mean_absolute_error(call_count_errors),
        giveup_delta=_mean_absolute_error(giveup_point_errors),
        plugin_transfer=plugin_transfer_score,
    )


def aggregate_s3(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
    silence_rate_delta: float,
) -> S3Metrics:
    """EVAL.md §5.3's confusion-matrix formulas, literal."""
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    false_alarm = false_positive / (false_positive + true_negative) if (false_positive + true_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return S3Metrics(
        precision=precision, recall=recall, false_alarm=false_alarm, f1=f1, silence_rate_delta=silence_rate_delta
    )


def aggregate_s4(
    *, correct_identifications: list[bool], reasons: list[str], code_switch_rate_delta: float
) -> S4Metrics:
    """EVAL.md §8.2: "辨識率 = 正確判斷的比例", plus the reason distribution
    (§8.2: "MUST 同時記錄辨識者的判斷理由。理由的分布比數字有用")."""
    distribution: dict[str, int] = {}
    for reason in reasons:
        distribution[reason] = distribution.get(reason, 0) + 1
    return S4Metrics(
        identification_rate=_rate(correct_identifications),
        reason_distribution=distribution,
        code_switch_rate_delta=code_switch_rate_delta,
    )
