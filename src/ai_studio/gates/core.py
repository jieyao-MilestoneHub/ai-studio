"""Shared gate shell.

Two structural rules, both inherited from video-autopilot-kit:

1. **Every gate returns the same shape.** One `GateReport`, so the runner, the
   CLI, and CI all read gates the same way regardless of what they check.

2. **Rule bodies live in their own files, not here.** Upstream's phrasing is
   不集中才不會互相污染 — keeping them apart is what stops one gate's helpers
   quietly becoming another gate's assumptions. `import-linter` additionally
   forbids gate-to-gate imports.

And the invariant that makes the whole layer testable:

> **A gate is a pure function of JSON artifacts on disk.** Gates never import
> providers, render, or runtime. `delivery_gate` gets its media facts from
> `probe.json`, not by shelling out to ffprobe.

That buys four things at once: gates run against fixture directories with no
GPU and no ffmpeg; a contributor can add a rule without understanding RunPod;
any stage can be re-checked in isolation; and a failed run stays fully
diagnosable from `runs/<id>/` after the fact.

## PRE and POST are not cosmetic

An H3 clip is 2-6 minutes of GPU time. A gate that runs after generation is a
receipt, not a check. So gates split into PRE (plan, format, prompt — pure
functions of `plan.json`, run before a single GPU-second is spent) and POST
(pace, caption, audio, grammar, delivery).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_studio.core.enums import Severity
from ai_studio.core.errors import GateFailure
from ai_studio.core.models import GateFinding, GateReport


class GateContext:
    """Reads a run directory. The only thing a gate is allowed to touch."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def artifact(self, name: str) -> dict[str, Any]:
        """Load `runs/<id>/<name>`, cached. Raises if it is missing."""
        if name not in self._cache:
            path = self.run_dir / name
            if not path.is_file():
                raise GateFailure(
                    f"{path} is missing — the stage that produces it has not run"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise GateFailure(f"{path} is not a JSON object: {type(data).__name__}")
            self._cache[name] = data
        return self._cache[name]

    def has(self, name: str) -> bool:
        return (self.run_dir / name).is_file()


class GateRun:
    """Accumulates findings for one gate.

    `assert_that(condition, ...)` reads as a positive statement — the message
    describes the *violation*, so a reader sees what went wrong rather than
    having to invert a rule.
    """

    def __init__(self, gate: str) -> None:
        self.gate = gate
        self._findings: list[GateFinding] = []
        self._counters: dict[str, float] = {}
        self._started = time.perf_counter()

    def assert_that(
        self,
        condition: bool,
        rule_id: str,
        message: str,
        *,
        severity: Severity = Severity.FAIL,
        where: str | None = None,
        observed: object = None,
        expected: object = None,
        source_url: str | None = None,
    ) -> bool:
        if not condition:
            self._findings.append(
                GateFinding(
                    rule_id=rule_id,
                    severity=severity,
                    message=message,
                    where=where,
                    observed=None if observed is None else str(observed),
                    expected=None if expected is None else str(expected),
                    source_url=source_url,
                )
            )
        return condition

    def warn_unless(self, condition: bool, rule_id: str, message: str, **kw: Any) -> bool:
        return self.assert_that(condition, rule_id, message, severity=Severity.WARN, **kw)

    def count(self, key: str, value: float) -> None:
        self._counters[key] = value

    def report(self) -> GateReport:
        return GateReport(
            gate=self.gate,
            findings=tuple(self._findings),
            counters=dict(self._counters),
            duration_ms=(time.perf_counter() - self._started) * 1000,
        )


GateFn = Callable[[GateContext], GateReport]


def write_report(run_dir: Path | str, report: GateReport) -> Path:
    """Persist a report to `runs/<id>/gates/<gate>.json`."""
    path = Path(run_dir) / "gates" / f"{report.gate}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def enforce(report: GateReport, *, strict: bool = True) -> GateReport:
    """Raise on failures in strict mode. Returns the report either way."""
    if strict and report.failures:
        lines = "\n".join(f"  [{f.rule_id}] {f.message}" for f in report.failures)
        raise GateFailure(f"{report.gate} failed:\n{lines}")
    return report


def selftest(gate_fn: GateFn, fixtures_dir: Path | str, *, expect_fail: str) -> None:
    """Assert a gate catches its own rule on a deliberately-bad fixture.

    Every gate ships one. A rule that silently stops working is otherwise
    indistinguishable from a rule that is being satisfied — which is the failure
    mode this whole layer exists to prevent.
    """
    report = gate_fn(GateContext(fixtures_dir))
    caught = {f.rule_id for f in report.failures}
    if expect_fail not in caught:
        raise AssertionError(
            f"{report.gate} did not catch {expect_fail} on {fixtures_dir}; caught {sorted(caught)}"
        )
