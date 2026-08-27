"""The send gate. SPEC.md §6.5, D29-D31; EVAL.md §7.2's literal numbers."""

from __future__ import annotations

from twin.agent.gate import is_upgrade_eligible, should_downgrade
from twin.core.enums import GateLevel
from twin.core.gate_metrics import GateMetrics


def _metrics(**overrides: object) -> GateMetrics:
    defaults: dict[str, object] = dict(
        run_id="run-1",
        s3_false_alarm=0.05,
        s3_silence_rate_delta=0.05,
        s1_normalized_accuracy=0.85,
        judge_agreement=0.85,
        confidence="high",
    )
    defaults.update(overrides)
    return GateMetrics(**defaults)  # type: ignore[arg-type]


class TestShouldDowngrade:
    def test_l0_never_downgrades_further(self) -> None:
        assert should_downgrade(GateLevel.L0, _metrics(s3_false_alarm=0.99)) is None

    def test_l1_stays_when_all_thresholds_pass(self) -> None:
        assert should_downgrade(GateLevel.L1, _metrics()) is None

    def test_l1_downgrades_to_l0_when_false_alarm_too_high(self) -> None:
        assert should_downgrade(GateLevel.L1, _metrics(s3_false_alarm=0.20)) == GateLevel.L0

    def test_l1_downgrades_to_l0_when_s1_too_low(self) -> None:
        assert should_downgrade(GateLevel.L1, _metrics(s1_normalized_accuracy=0.5)) == GateLevel.L0

    def test_l2_downgrades_to_l1_when_false_alarm_too_high(self) -> None:
        assert should_downgrade(GateLevel.L2, _metrics(s3_false_alarm=0.20)) == GateLevel.L1

    def test_downgrade_triggers_below_the_judge_agreement_floor_even_if_metrics_pass(self) -> None:
        assert should_downgrade(GateLevel.L1, _metrics(judge_agreement=0.5)) == GateLevel.L0


class TestIsUpgradeEligible:
    def test_eligible_when_both_rounds_pass(self) -> None:
        assert is_upgrade_eligible(GateLevel.L1, (_metrics(), _metrics())) is True

    def test_not_eligible_if_either_round_fails(self) -> None:
        assert is_upgrade_eligible(GateLevel.L1, (_metrics(), _metrics(s3_false_alarm=0.9))) is False

    def test_not_eligible_if_either_round_is_low_confidence(self) -> None:
        low = _metrics(confidence="low")
        assert is_upgrade_eligible(GateLevel.L1, (_metrics(), low)) is False

    def test_l0_is_not_a_valid_upgrade_target(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="default"):
            is_upgrade_eligible(GateLevel.L0, (_metrics(), _metrics()))

    def test_l2_eligibility_only_reflects_the_checkable_metric(self) -> None:
        """EVAL §7.2's other two L2 conditions (30-day L1 runtime, veto rate)
        aren't representable in GateMetrics — True here means only "the
        false-alarm metric passed twice," not "L2 upgrade is fully justified"."""
        assert is_upgrade_eligible(GateLevel.L2, (_metrics(), _metrics())) is True
