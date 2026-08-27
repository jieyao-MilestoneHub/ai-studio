"""S2 — tool use. EVAL.md §4. Real tasks (with programmatically-verifiable
end states) and their verifiers need real held-out tool-use situations
(twin/PLAN.md Phase 11) — deferred; the real machinery already exists
(`harness.schema.VerifiedTaskResult`/`TaskVerifier`, `harness.aggregate.
aggregate_s2`)."""

from __future__ import annotations

from twin.harness.schema import VerifiedTaskResult


def run_held_out_tasks(*, task_ids: list[str], decider: object) -> list[VerifiedTaskResult]:
    raise NotImplementedError(
        "EVAL.md §4.2: S2 tasks and their programmatic verifiers need real "
        "held-out tool-use situations (twin/PLAN.md Phase 11) — not yet built."
    )
