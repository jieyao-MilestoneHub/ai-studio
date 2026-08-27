"""Reflection-step insertion. SPEC.md §5.4 — deliberately thin.

The 15-30% insertion ratio is explicitly a SHOULD, "以 ablation 驗證" — an
experimental parameter, not a decided value — and generating real reflection
*content* needs a real Teacher call against real behavioral trajectories
(Phase 10). Nothing is lost by deferring: this signature is stable regardless
of what the eventual ratio/generation strategy turns out to be.
"""

from __future__ import annotations

from twin.core.trajectory import Trajectory
from twin.teacher.base import Teacher


def maybe_insert_reflection(trajectory: Trajectory, *, ratio: float, teacher: Teacher) -> Trajectory:
    raise NotImplementedError(
        "SPEC.md §5.4: the reflection insertion ratio (15-30%, SHOULD, "
        "'以 ablation 驗證') is an experimental parameter pending real ablation "
        "against real trajectories — deferred to Phase 10, not yet decided."
    )
