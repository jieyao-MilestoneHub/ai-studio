"""The pre-launch checklist: everything provable without spending a GPU-second.

There is no stub in this project, so there is no "prove the chain works locally
first" step. The first real run is simultaneously the implementation's
acceptance, the measurement, and the only affordable attempt — $1.556 of
approved budget against an L40S at $1.004/hr. Forty minutes.

This module is what replaces the stub. It is not "verify less"; it is **prove
everything that can be proved for free, so those forty minutes are spent only
on the part that genuinely needs a GPU**. The GPU side's checks live here;
the result type, the rules (SKIP is never PASS; green means green) and the
runner are `ai_studio.checks`, which the request side builds its own list on.

Nothing here creates a pod.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ai_studio import paths
from ai_studio.checks import CheckResult, check_offline_suite, fail, passed, run_checks, skip
from ai_studio.config.settings import get_settings


def check_poster() -> CheckResult:
    """2. A poster comes out of ffmpeg under the default 1MB preview ceiling.

    Run for real on the host that will do it: the VPS is a 1 GB box, and the
    question is not whether the code is right but whether that machine can
    decode a frame.
    """
    name = "poster generation under the 1MB ceiling"
    from ai_studio import media

    settings = get_settings()
    if media.which(settings.ffmpeg_bin) is None:
        return skip(2, name, f"{settings.ffmpeg_bin} is not on PATH")

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
        return fail(2, name, f"{type(exc).__name__}: {exc}")
    return passed(2, name, ", ".join(made) + f" (ceiling {media.POSTER_MAX_BYTES // 1024}KB)")


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
            return fail(3, name, f"{path.name}: {type(exc).__name__}: {exc}")
        loaded.append(f"{path.name}({len(workflow.bindings)} bindings)")
    return passed(3, name, ", ".join(loaded))


def check_placement() -> CheckResult:
    """4. Every ladder rung still corresponds to something the catalog offers.

    Read-only; it creates nothing. A dead rung must be found now, not at 11:00
    when all four are refused and the window is lost.
    """
    name = "placement ladder matches the live catalog"
    settings = get_settings()
    if settings.runpod_api_key is None:
        return skip(4, name, "RUNPOD_API_KEY is not set")

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
        return fail(4, name, f"{type(exc).__name__}: {exc}")

    if dead:
        return fail(4, name, "; ".join(dead))
    return passed(4, name, f"all {len(sess.CANDIDATES)} rungs licence-safe and offered")


# ------------------------------------------------------------------- runner


def run_all(*, run_suite: bool = True) -> list[CheckResult]:
    """The GPU side's checklist, cheapest-to-prove first."""
    return run_checks([
        lambda: check_offline_suite(run=run_suite),
        check_poster,
        check_graphs,
        check_placement,
    ])
