"""The projection of an eval report that agent.gate is allowed to see.

`agent.gate` (SPEC.md §6.5, D31) needs this round's numbers to decide
send-gate transitions, but the full `harness.report.EvalReport` lives behind
the "Eval harness stays a leaf" import-linter contract — production MUST NOT
depend on eval plumbing. `harness.report` emits both the full report and this
small frozen projection; `agent.gate` depends only on this module, never on
`twin.harness`, exactly mirroring how `core.adapter.AdapterManifest` lets
agent read train's output without importing train internals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

JUDGE_AGREEMENT_FLOOR = 0.80
"""EVAL.md §6.3: MUST NOT be lowered under any stage — the calibration value
of the measuring instrument, not a product target. Defined once here (not
duplicated in both `agent.gate` and `harness.gate_check`, which both need it
and cannot import each other) so there is exactly one place a future edit
could get this wrong."""


class GateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    s3_false_alarm: float
    s3_silence_rate_delta: float
    s1_normalized_accuracy: float
    judge_agreement: float
    confidence: Literal["high", "low"]
