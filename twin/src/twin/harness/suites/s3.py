"""S3 — proactivity. EVAL.md §5. A real time-series feed with hard-negative
labeling needs a real tick loop actually running (twin/PLAN.md Phase 11) —
deferred; the real machinery already exists (`harness.aggregate.aggregate_s3`,
`harness.report.EvalReport`)."""

from __future__ import annotations

from twin.harness.schema import JudgedItem


def build_time_series_feed(*, principal_id: str, window_start: str, window_end: str) -> list[JudgedItem]:
    raise NotImplementedError(
        "EVAL.md §5.2: S3's time-series feed and hard-negative labeling need "
        "a real running tick loop (twin/PLAN.md Phase 11) — not yet built."
    )
