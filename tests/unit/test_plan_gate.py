"""`gates.plan_gate`: a pure function of `plan.json`, proven by fixtures
that each break exactly one rule (the `selftest` discipline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_studio.core.errors import GateFailure
from ai_studio.gates import plan_gate
from ai_studio.gates.core import GateContext, enforce, selftest, write_report

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gates" / "plan_gate"


def test_the_good_plan_passes_with_one_designed_warning() -> None:
    report = plan_gate.run(GateContext(FIXTURES / "good"))
    assert report.passed
    assert [f.rule_id for f in report.warnings] == ["R-BAND-WARN"]  # the held conflict shot
    assert report.counters["segments"] == 10 and report.counters["dissolves"] == 1
    assert plan_gate.describe(report) == "passed with 1 warning(s): R-BAND-WARN"


@pytest.mark.parametrize(
    ("fixture", "rule"),
    [
        ("metronome", "R-CV"),
        ("slow_pair", "R-CONSEC-SLOW"),
        ("caption_too_fast", "C-READ-WARN"),
        ("emoji", "C-EMOJI"),
        ("off_grid", "P-QUANTUM"),
        ("dangling_cue", "P-SEGMENT-REF"),
    ],
)
def test_each_fixture_trips_its_rule(fixture: str, rule: str) -> None:
    if rule.endswith("WARN"):
        report = plan_gate.run(GateContext(FIXTURES / fixture))
        assert rule in {f.rule_id for f in report.warnings}
    else:
        selftest(plan_gate.run, FIXTURES / fixture, expect_fail=rule)


def test_enforce_raises_and_the_report_is_written(tmp_path: Path) -> None:
    report = plan_gate.run(GateContext(FIXTURES / "metronome"))
    assert plan_gate.describe(report).startswith("failed: ") and "[R-CV]" in plan_gate.describe(report)
    path = write_report(tmp_path, report)
    assert path == tmp_path / "gates" / "plan_gate.json"
    assert json.loads(path.read_text(encoding="utf-8"))["gate"] == "plan_gate"
    with pytest.raises(GateFailure, match="R-CV"):
        enforce(report)


def test_a_missing_plan_is_a_gate_failure_not_a_pass(tmp_path: Path) -> None:
    with pytest.raises(GateFailure, match="missing"):
        plan_gate.run(GateContext(tmp_path))
