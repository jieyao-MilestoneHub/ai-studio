"""A pre-launch checklist's machinery: the result type, the runner, and
the one check every package in this repo shares (its own offline suite).

Two rules, both of which are the whole point:

**A check that cannot run is SKIP, never PASS.** "We could not verify this"
must never render as "this is verified" -- that is the silent-degradation
failure this codebase is built to refuse, applied to the checklist itself.

**Green means green.** `summarise` is true only when every check PASSes.
Offline, some legitimately skip and the exit code says so.

Library, not CLI: `ai_studio.cli.preflight` builds the GPU side's list on
it and `fun_workflow.cli.preflight` the request side's, so neither has to
import the other's command layer.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from ai_studio import paths


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    number: int
    name: str
    status: Status
    detail: str

    @property
    def ok(self) -> bool:
        return self.status is Status.PASS


def skip(number: int, name: str, why: str) -> CheckResult:
    return CheckResult(number, name, Status.SKIP, why)


def passed(number: int, name: str, detail: str) -> CheckResult:
    return CheckResult(number, name, Status.PASS, detail)


def fail(number: int, name: str, detail: str) -> CheckResult:
    return CheckResult(number, name, Status.FAIL, detail)


def check_offline_suite(*, run: bool = True, cwd: Path | None = None) -> CheckResult:
    """1. The whole offline toolchain: tests, ruff, import contracts, types.

    `cwd` selects which package's suite: this one by default; the request
    side passes its own directory."""
    name = "offline suite (pytest, ruff, lint-imports, mypy)"
    if not run:
        return skip(1, name, "--skip-suite was passed")

    commands = [
        (["uv", "run", "pytest", "tests", "-q", "-m", "not runpod"], "pytest"),
        (["uv", "run", "ruff", "check", "--no-cache", "src", "tests"], "ruff"),
        (["uv", "run", "lint-imports"], "lint-imports"),
        (["uv", "run", "mypy"], "mypy"),
    ]
    for argv, label in commands:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=900.0, check=False, cwd=cwd or paths.repo_root(),
            )
        except FileNotFoundError:
            return skip(1, name, f"{argv[0]} not on PATH")
        except subprocess.TimeoutExpired:
            return fail(1, name, f"{label} timed out after 900s")
        if proc.returncode != 0:
            tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-3:]
            return fail(1, name, f"{label} exited {proc.returncode}: {' | '.join(tail)}")
    return passed(1, name, "pytest, ruff, lint-imports and mypy all clean")


def run_checks(checks: list[Callable[[], CheckResult]]) -> list[CheckResult]:
    """Run every check in order. Never raises: a crashing check is a FAIL."""
    results: list[CheckResult] = []
    for number, check in enumerate(checks, start=1):
        try:
            results.append(check())
        except Exception as exc:  # a check that dies is a check that failed
            results.append(
                fail(number, f"check {number}", f"raised {type(exc).__name__}: {exc}")
            )
    return results


def summarise(results: list[CheckResult]) -> tuple[bool, str]:
    """`(all green, one-line summary)`.

    Green means green: skips do not count. A checklist that reports "fine"
    with three unknowns on it is worse than no checklist, because it gets
    believed.
    """
    tally = {status: sum(1 for r in results if r.status is status) for status in Status}
    green = tally[Status.PASS] == len(results) and bool(results)
    return green, (
        f"{tally[Status.PASS]} passed, {tally[Status.FAIL]} failed, "
        f"{tally[Status.SKIP]} skipped, of {len(results)}"
    )


def stamp(results: list[CheckResult], *, when: datetime) -> str:
    """A record to paste into the run log. Plain ASCII for the Windows console."""
    lines = [f"preflight {when.isoformat(timespec='seconds')}"]
    lines += [f"  {r.number}. [{r.status.value}] {r.name} -- {r.detail}" for r in results]
    green, summary = summarise(results)
    lines.append(f"  => {summary}{'' if green else '  (NOT green)'}")
    return "\n".join(lines)
