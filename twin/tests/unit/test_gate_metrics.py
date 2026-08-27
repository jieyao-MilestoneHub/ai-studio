"""GateMetrics — the core-level projection that lets agent.gate see eval
numbers without importing twin.harness. SPEC.md §6.5, D31."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from twin.core.gate_metrics import GateMetrics


def test_gate_metrics_constructs() -> None:
    metrics = GateMetrics(
        run_id="run-001",
        s3_false_alarm=0.12,
        s3_silence_rate_delta=0.05,
        s1_normalized_accuracy=0.8,
        judge_agreement=0.85,
        confidence="high",
    )
    assert metrics.confidence == "high"


def test_confidence_is_restricted_to_high_or_low() -> None:
    with pytest.raises(ValidationError):
        GateMetrics(
            run_id="run-001",
            s3_false_alarm=0.1,
            s3_silence_rate_delta=0.1,
            s1_normalized_accuracy=0.8,
            judge_agreement=0.85,
            confidence="medium",  # type: ignore[arg-type]
        )
