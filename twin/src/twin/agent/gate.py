"""The send gate. SPEC.md §6.5, D29-D31 — L0 draft / L1 whitelist / L2 fully
automatic, intercepted at the runtime layer so upgrading or downgrading never
touches a tool definition or needs a retrain.

This is the *send* gate ("given we're already at some level, should it
change?"), using EVAL.md §7.2's numbers — distinct from the *fitness* gate in
`harness.gate_check` ("is the twin usable at all?", EVAL.md §7.1's T1/T2).
They consume the same `core.gate_metrics.GateMetrics` shape but different
threshold tables answering different questions; do not merge them.

Every number here is already a literal in EVAL.md §7.2 — there is no
undetermined design choice left in the *comparison* logic, only in the
*inputs* (a real eval round, which doesn't exist yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from twin.core.enums import GateLevel
from twin.core.gate_metrics import JUDGE_AGREEMENT_FLOOR, GateMetrics


@dataclass(frozen=True)
class GateThresholds:
    s3_false_alarm_max: float
    s3_silence_rate_delta_max: float | None = None
    s1_normalized_accuracy_min: float | None = None


L1_THRESHOLDS = GateThresholds(
    s3_false_alarm_max=0.15, s3_silence_rate_delta_max=0.10, s1_normalized_accuracy_min=0.78
)
L2_THRESHOLDS = GateThresholds(s3_false_alarm_max=0.10)
"""EVAL.md §7.2's other two L2 conditions (30 real days of L1 runtime; a
principal veto rate ≤ 0.10 over that period) need a real deployment-log
tracker that doesn't exist yet (Phase 14) — they are not representable in
`GateMetrics` and are deliberately NOT checked here. `is_upgrade_eligible`'s
docstring says so explicitly rather than silently only checking the metric
it can."""


class GateState(BaseModel):
    model_config = ConfigDict(frozen=True)

    principal_id: str
    level: GateLevel
    updated_at: datetime
    reason: str
    evidence_run_ids: list[str]


def _passes(metrics: GateMetrics, thresholds: GateThresholds) -> bool:
    if metrics.judge_agreement < JUDGE_AGREEMENT_FLOOR:
        return False
    if metrics.s3_false_alarm > thresholds.s3_false_alarm_max:
        return False
    if (
        thresholds.s3_silence_rate_delta_max is not None
        and metrics.s3_silence_rate_delta > thresholds.s3_silence_rate_delta_max
    ):
        return False
    return not (
        thresholds.s1_normalized_accuracy_min is not None
        and metrics.s1_normalized_accuracy < thresholds.s1_normalized_accuracy_min
    )


def should_downgrade(current: GateLevel, metrics: GateMetrics) -> GateLevel | None:
    """D31: downgrade MUST be automatic — any round failing the current
    level's thresholds drops one level immediately, no human required.
    Returns the new level, or None if no downgrade is warranted."""
    if current == GateLevel.L0:
        return None
    thresholds = L1_THRESHOLDS if current == GateLevel.L1 else L2_THRESHOLDS
    if _passes(metrics, thresholds):
        return None
    return GateLevel.L0 if current == GateLevel.L1 else GateLevel.L1


def is_upgrade_eligible(target: GateLevel, two_consecutive: tuple[GateMetrics, GateMetrics]) -> bool:
    """D31: upgrade MUST be manual-triggered, with two consecutive passing
    rounds as evidence — this only checks eligibility, it never itself
    upgrades. A round marked low-confidence (eval-harness skill, cross-session
    non-determinism > 5%) MUST NOT be used for an upgrade decision at all."""
    if target == GateLevel.L0:
        raise ValueError("L0 is the default; there is no 'upgrade to L0'")
    if any(m.confidence == "low" for m in two_consecutive):
        return False
    thresholds = L1_THRESHOLDS if target == GateLevel.L1 else L2_THRESHOLDS
    return all(_passes(m, thresholds) for m in two_consecutive)
