"""The pre-launch checklist: everything provable without spending a GPU-second.

There is no stub in this project, so there is no "prove the chain works locally
first" step. The first real run is simultaneously the implementation's
acceptance, the measurement, and the only affordable attempt — $1.556 of
approved budget against an L40S at $1.004/hr. Forty minutes.

This module is what replaces the stub. It is not "verify less"; it is **prove
everything that can be proved for free, so those forty minutes are spent only
on the part that genuinely needs a GPU**. The GPU side's checks live here,
with the result type and runner the request side (`fun_workflow.cli.
preflight`) builds its own list on.

Two rules it follows, both of which are the whole point:

**A check that cannot run is SKIP, never PASS.** "We could not verify this"
must never render as "this is verified" — that is the silent-degradation
failure this codebase is built to refuse, applied to the checklist itself. A
missing credential produces a skip with the reason attached, and the run is not
green.

**Green means green.** `preflight` exits 0 only when every check PASSes.
Offline, some legitimately skip and the exit code says so.

Nothing here creates a pod.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from ai_studio import paths
from ai_studio.config.settings import get_settings


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


def _skip(number: int, name: str, why: str) -> CheckResult:
    return CheckResult(number, name, Status.SKIP, why)


def _pass(number: int, name: str, detail: str) -> CheckResult:
    return CheckResult(number, name, Status.PASS, detail)


def _fail(number: int, name: str, detail: str) -> CheckResult:
    return CheckResult(number, name, Status.FAIL, detail)


# ------------------------------------------------------------------- checks


def check_offline_suite(*, run: bool = True, cwd: Path | None = None) -> CheckResult:
    """1. The whole offline toolchain: tests, ruff, import contracts, types.

    `cwd` selects which package's suite: this one by default; the request
    side passes its own directory."""
    name = "offline suite (pytest, ruff, lint-imports, mypy)"
    if not run:
        return _skip(1, name, "--skip-suite was passed")

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
            return _skip(1, name, f"{argv[0]} not on PATH")
        except subprocess.TimeoutExpired:
            return _fail(1, name, f"{label} timed out after 900s")
        if proc.returncode != 0:
            tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-3:]
            return _fail(1, name, f"{label} exited {proc.returncode}: {' | '.join(tail)}")
    return _pass(1, name, "pytest, ruff, lint-imports and mypy all clean")


def check_poster() -> CheckResult:
    """7. A poster comes out of ffmpeg under LINE's 1MB preview ceiling.

    Run for real on the host that will do it: the VPS is a 1 GB box, and the
    question is not whether the code is right but whether that machine can
    decode a frame.
    """
    name = "poster generation under the 1MB ceiling"
    from ai_studio import media

    settings = get_settings()
    if media.which(settings.ffmpeg_bin) is None:
        return _skip(2, name, f"{settings.ffmpeg_bin} is not on PATH")

    root = Path(tempfile.mkdtemp(prefix="ai-studio-preflight-"))
    made: list[str] = []
    try:
        for label, source, size in (
            ("clip", root / "clip.mp4", "864x480"),
            ("image", root / "flux.png", "1024x1024"),
        ):
            media.run(
                [
                    settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration=1",
                    "-frames:v", "1" if label == "image" else "24",
                    *(["-pix_fmt", "yuv420p"] if label == "clip" else []),
                    str(source),
                ],
                timeout_s=120.0,
            )
            out = media.poster(source, root / f"{label}_poster.jpg")
            made.append(f"{label}={out.stat().st_size // 1024}KB")
    except Exception as exc:
        return _fail(2, name, f"{type(exc).__name__}: {exc}")
    return _pass(2, name, ", ".join(made) + f" (ceiling {media.POSTER_MAX_BYTES // 1024}KB)")


def check_graphs() -> CheckResult:
    """3. All workflows load, bind and validate.

    A malformed graph is otherwise discovered at the moment of submission, on a
    pod that is already billing.
    """
    name = "all ComfyUI graphs load and validate"
    from ai_studio.comfy.graph import IMAGE_REQUIRED_BINDINGS, REQUIRED_BINDINGS, Workflow

    targets = [
        ("h3_fl2va_turbo.json", REQUIRED_BINDINGS),
        ("h3_fl2va_turbo_fp8.json", REQUIRED_BINDINGS),
        ("h3_i2va_turbo.json", REQUIRED_BINDINGS),
        ("h3_i2va_turbo_fp8.json", REQUIRED_BINDINGS),
        ("flux_dev.json", IMAGE_REQUIRED_BINDINGS),
        ("flux_dev_i2i.json", IMAGE_REQUIRED_BINDINGS | {"source_image", "denoise"}),
    ]
    loaded: list[str] = []
    for filename, required in targets:
        path = paths.workflow(filename)
        try:
            workflow = Workflow.load(path, required_bindings=required)
        except Exception as exc:
            return _fail(3, name, f"{path.name}: {type(exc).__name__}: {exc}")
        loaded.append(f"{path.name}({len(workflow.bindings)} bindings)")
    return _pass(3, name, ", ".join(loaded))


def check_placement() -> CheckResult:
    """4. Every ladder rung still corresponds to something the catalog offers.

    Read-only; it creates nothing. A dead rung must be found now, not at 11:00
    when all four are refused and the window is lost.
    """
    name = "placement ladder matches the live catalog"
    settings = get_settings()
    if settings.runpod_api_key is None:
        return _skip(4, name, "RUNPOD_API_KEY is not set")

    from ai_studio.runtime import session as sess
    from ai_studio.runtime.pod import LICENCE_SAFE_DATACENTERS, PodManager

    # Same verdicts as `ai-studio pod placement`, deliberately: a rung the
    # catalog does not offer is refused on deploy exactly like one that is
    # merely out of stock, so without this the ladder falls through to a softer
    # GPU for months and it reads as bad luck.
    dead: list[str] = []
    try:
        with PodManager() as manager:
            for tier in sess.CANDIDATES:
                if tier.datacenter not in LICENCE_SAFE_DATACENTERS:
                    dead.append(f"{tier.label}: outside H3's licence")
                    continue
                if manager.verify_placement(tier.gpu, tier.datacenter, cloud=tier.cloud) == (
                    "not-offered"
                ):
                    dead.append(f"{tier.label} @ {tier.datacenter}: never offered here")
    except Exception as exc:
        return _fail(4, name, f"{type(exc).__name__}: {exc}")

    if dead:
        return _fail(4, name, "; ".join(dead))
    return _pass(4, name, f"all {len(sess.CANDIDATES)} rungs licence-safe and offered")


# ------------------------------------------------------------------- runner


def run_checks(checks: list[Callable[[], CheckResult]]) -> list[CheckResult]:
    """Run every check in order. Never raises: a crashing check is a FAIL."""
    results: list[CheckResult] = []
    for number, check in enumerate(checks, start=1):
        try:
            results.append(check())
        except Exception as exc:  # a check that dies is a check that failed
            results.append(
                _fail(number, f"check {number}", f"raised {type(exc).__name__}: {exc}")
            )
    return results


def run_all(*, run_suite: bool = True) -> list[CheckResult]:
    """The GPU side's checklist, cheapest-to-prove first."""
    return run_checks([
        lambda: check_offline_suite(run=run_suite),
        check_poster,
        check_graphs,
        check_placement,
    ])


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
