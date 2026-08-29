"""Pacing checks over a list of segment durations (editing grammar section 2).

Pure arithmetic on seconds; it does not know what a shot, a clip or a job is.
The numbers are the caller's (`PacingPolicy`), because the right band for a
one-minute drama is not the right band for a Shorts explainer. Upstream's
Shorts profile is `[reported]`: warn above 4.0 s, fail above 6.5 s, >= 2
consecutive slow shots fail, and a duration coefficient of variation under
0.11 is "metronomic" -- six equal shots in a row is what an audience reads
as a slideshow, whatever the content.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ai_studio.core.enums import Severity
from ai_studio.core.models import GateFinding

SOURCE_URL = "https://github.com/Hao0321/video-autopilot-kit"

CV_FLOOR = 0.11
"""[reported] upstream pace_gate D-E: below this the cut rhythm is a metronome."""


@dataclass(frozen=True)
class PacingPolicy:
    """The band one kind of piece must sit in. All seconds."""

    min_s: float
    warn_s: float
    fail_s: float
    cv_floor: float = CV_FLOOR
    max_consecutive_slow: int = 1
    total_band_s: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not 0 < self.min_s <= self.warn_s <= self.fail_s:
            raise ValueError(f"pacing band must be 0 < min <= warn <= fail: {self}")
        if self.total_band_s is not None and self.total_band_s[0] > self.total_band_s[1]:
            raise ValueError(f"total band inverted: {self.total_band_s}")


def coefficient_of_variation(durations: Sequence[float]) -> float:
    """Population standard deviation over the mean; 0.0 for a single value."""
    if len(durations) < 2:
        return 0.0
    mean = sum(durations) / len(durations)
    if mean <= 0:
        raise ValueError("durations must be positive")
    var = sum((d - mean) ** 2 for d in durations) / len(durations)
    return math.sqrt(var) / mean


def check(durations: Sequence[float], policy: PacingPolicy) -> list[GateFinding]:
    """Every pacing rule as a finding. An empty list is a pass."""
    if not durations:
        return [GateFinding(rule_id="R-EMPTY", severity=Severity.FAIL, message="no segments to pace")]
    if any(d <= 0 for d in durations):
        raise ValueError(f"durations must be positive: {list(durations)}")
    findings: list[GateFinding] = []

    def add(rule: str, severity: Severity, message: str, *, where: str | None = None,
            observed: object = None, expected: object = None) -> None:
        findings.append(GateFinding(
            rule_id=rule, severity=severity, message=message, where=where,
            observed=None if observed is None else str(observed),
            expected=None if expected is None else str(expected), source_url=SOURCE_URL,
        ))

    slow_run = 0
    for i, d in enumerate(durations):
        where = f"segment {i}"
        if d < policy.min_s:
            add("R-BAND-MIN", Severity.FAIL, f"{where} is {d:.2f}s, under the {policy.min_s}s floor",
                where=where, observed=f"{d:.3f}", expected=f">= {policy.min_s}")
        if d > policy.fail_s:
            add("R-BAND-FAIL", Severity.FAIL, f"{where} is {d:.2f}s, over the {policy.fail_s}s ceiling",
                where=where, observed=f"{d:.3f}", expected=f"<= {policy.fail_s}")
        elif d > policy.warn_s:
            add("R-BAND-WARN", Severity.WARN, f"{where} is {d:.2f}s, a slow shot (> {policy.warn_s}s)",
                where=where, observed=f"{d:.3f}", expected=f"<= {policy.warn_s}")
        slow_run = slow_run + 1 if d > policy.warn_s else 0
        if slow_run == policy.max_consecutive_slow + 1:
            add("R-CONSEC-SLOW", Severity.FAIL,
                f"{slow_run} consecutive slow segments ending at {where}; the rhythm stalls",
                where=where, observed=slow_run, expected=f"<= {policy.max_consecutive_slow}")

    cv = coefficient_of_variation(durations)
    if len(durations) >= 3 and cv < policy.cv_floor:
        add("R-CV", Severity.FAIL,
            f"segment durations vary by CV {cv:.3f}; under {policy.cv_floor} reads as a metronome",
            observed=f"{cv:.3f}", expected=f">= {policy.cv_floor}")

    if policy.total_band_s is not None:
        lo, hi = policy.total_band_s
        total = sum(durations)
        if not lo <= total <= hi:
            add("R-TOTAL", Severity.FAIL, f"total {total:.1f}s is outside {lo}-{hi}s",
                observed=f"{total:.2f}", expected=f"{lo}..{hi}")
    return findings
