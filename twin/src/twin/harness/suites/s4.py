"""S4 — blind test. EVAL.md §8. Mixing real principal output with real twin
output needs a trained, running twin to actually produce comparable output
(twin/PLAN.md Phase 12) — deferred; the real machinery already exists
(`harness.aggregate.aggregate_s4`, `harness.report.EvalReport`)."""

from __future__ import annotations

from twin.harness.schema import RawEvalSample


def build_blind_test_set(*, principal_samples: list[str], twin_samples: list[str]) -> list[RawEvalSample]:
    raise NotImplementedError(
        "EVAL.md §8.1: S4 needs real twin-produced output to mix with real "
        "principal output (twin/PLAN.md Phase 12) — not yet built."
    )
