"""Drain the queue while the service window is open, dispatching each job to
the H3 (video) or Flux (image) provider by its `media_kind`.

The loop is shaped by two facts about the window rather than by the work itself:

1. **A clip takes about five minutes — an image, well under one — and the
   window ends on a clock.** So the drainer stops *claiming* new work well
   before the bell and only finishes what it already holds. Starting a render
   at 12:58 means paying for four minutes of GPU and then having
   `--terminate-after` throw the result away — money spent for nothing, which
   is worse than making someone wait for tomorrow.

2. **A failure is usually the machine's, not the request's.** The window closed,
   the pod was preempted, the proxy hiccuped. Those go back in the queue
   (`requeue=True`) so the request survives; only a failure that would repeat
   identically — a prompt the model rejects — is terminal.

`drain_window` is no longer what runs on a schedule — `pipeline.worker` is, and
it owns the loop now. What it does *not* own is the two functions that actually
turn a job into a file: `render_clip` and `render_image` are shared, so there is
exactly one submit/poll/fetch sequence per media kind rather than a second copy
that drifts. `drain_window` stays as the manual, one-shot operator tool: "the
worker is wedged, empty the queue on the pod that is already open".
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ai_studio.core.enums import GenMode, MediaKind
from ai_studio.core.errors import AIStudioError, ProviderError
from ai_studio.core.image_provider_spec import ImageRequest
from ai_studio.core.provider_spec import ClipRequest
from ai_studio.pipeline.queue import Job, JobQueue

STOP_CLAIMING_BEFORE_S = 8 * 60
"""Reserve the last eight minutes of the window for finishing, not starting.

A clip measured ~5 minutes end to end on the hardware tested so far; eight gives
that headroom plus the fetch.
"""

MAX_ATTEMPTS = 3
"""After this many tries a request stops being requeued and fails for good.

Without a cap, requeue is an infinite loop: the job goes back to `parsed`, the
same drainer claims it again immediately, fails the same way, and the window
burns down retrying one broken request.
"""

MAX_CONSECUTIVE_FAILURES = 3
"""Consecutive failures that end the window early.

Three in a row is not bad luck, it is a broken pod. Stopping preserves the rest
of the window's money instead of grinding through the whole queue failing.
"""


class ClipProviderLike(Protocol):
    def capabilities(self) -> Any: ...
    async def submit(self, request: ClipRequest) -> Any: ...
    async def poll(self, job: Any) -> Any: ...
    async def fetch(self, job: Any, dest: Path) -> Any: ...
    async def cancel(self, job: Any) -> None: ...
    async def aclose(self) -> None: ...


class ImageProviderLike(Protocol):
    def capabilities(self) -> Any: ...
    async def submit(self, request: ImageRequest) -> Any: ...
    async def poll(self, job: Any) -> Any: ...
    async def fetch(self, job: Any, dest: Path) -> Any: ...
    async def cancel(self, job: Any) -> None: ...
    async def aclose(self) -> None: ...


@dataclass
class DrainReport:
    """What one window actually achieved. Written to the log and the report."""

    completed: int = 0
    failed: int = 0
    requeued: int = 0
    seconds: list[float] = field(default_factory=list)
    gpu_tier: str | None = None
    stopped_early: str | None = None

    @property
    def first_clip_s(self) -> float | None:
        return self.seconds[0] if self.seconds else None

    @property
    def later_clips_s(self) -> list[float]:
        """Second clip onward — the measurement that decides whether a window
        amortises its setup, or whether ComfyUI evicts the model between jobs."""
        return self.seconds[1:]

    def summary(self) -> str:
        parts = [f"completed={self.completed}", f"failed={self.failed}",
                 f"requeued={self.requeued}"]
        if self.seconds:
            parts.append("clip_seconds=" + ", ".join(f"{s:.0f}" for s in self.seconds))
        if self.gpu_tier:
            parts.append(f"gpu={self.gpu_tier}")
        if self.stopped_early:
            parts.append(f"stopped_early={self.stopped_early!r}")
        return " ".join(parts)


async def drain_window(
    queue: JobQueue,
    providers: dict[MediaKind, ClipProviderLike | ImageProviderLike],
    *,
    window_end: datetime,
    files_dir: Path,
    gpu_tier: str | None = None,
    poll_interval_s: float = 15.0,
    max_clips: int | None = None,
    on_activity: Any = None,
) -> DrainReport:
    """Generate clips and images until the window closes or the queue empties.

    One shared pod, one FIFO queue, dispatched per job by `media_kind` —
    `providers` must have an entry for every kind the queue might contain.

    `on_activity` is called after each render so the idle reaper's timer
    resets — otherwise a long-running window looks idle and gets closed
    underneath us.
    """
    report = DrainReport(gpu_tier=gpu_tier)
    files_dir = Path(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    caps_by_kind = {kind: provider.capabilities() for kind, provider in providers.items()}

    # Anything left `running` belongs to a window that ended badly. Reclaim it
    # before starting, or it sits there forever.
    report.requeued += queue.release_running("previous window ended")

    consecutive_failures = 0

    while may_claim(window_end):
        if max_clips is not None and report.completed >= max_clips:
            break
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            report.stopped_early = "three consecutive failures; the pod looks broken"
            break

        job = queue.claim_next(gpu_tier=gpu_tier)
        if job is None:
            break

        provider: Any = providers[job.media_kind]
        caps = caps_by_kind[job.media_kind]

        started = time.monotonic()
        try:
            if job.media_kind is MediaKind.IMAGE:
                asset = await render_image(
                    job, provider, caps, files_dir, window_end, poll_interval_s
                )
            else:
                asset = await render_clip(job, provider, caps, files_dir, window_end, poll_interval_s)
        except ProviderError as exc:
            # The backend's problem, so keep the request — but only while it has
            # attempts left, or requeue becomes an infinite loop.
            if job.attempts >= MAX_ATTEMPTS:
                queue.fail(job.id, f"provider failed {job.attempts}x: {exc}")
                report.failed += 1
            else:
                queue.fail(job.id, f"provider: {exc}", requeue=True)
                report.requeued += 1
            consecutive_failures += 1
            continue
        except AIStudioError as exc:
            # Ours or the request's, and it would repeat identically. Terminal.
            queue.fail(job.id, str(exc))
            report.failed += 1
            consecutive_failures += 1
            continue

        queue.complete(job.id, str(asset))
        report.completed += 1
        report.seconds.append(time.monotonic() - started)
        consecutive_failures = 0
        if on_activity is not None:
            on_activity()

    return report


def may_claim(deadline: datetime) -> bool:
    """Is there enough time left before `deadline` to start something new?

    `deadline` is whatever bell the caller answers to. For `drain_window` it is
    the window end; for the always-on worker it is the end of business hours.
    The reserve is the same either way, because what it protects against is the
    same: paying for four minutes of a render that `--terminate-after` throws
    away.
    """
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    return remaining > STOP_CLAIMING_BEFORE_S


async def render_clip(
    job: Job,
    provider: ClipProviderLike,
    caps: Any,
    files_dir: Path,
    window_end: datetime,
    poll_interval_s: float,
) -> Path:
    """Submit, poll, fetch. One clip."""
    plan = job.prompt or {}
    rendered = plan.get("_rendered")
    if not rendered:
        raise AIStudioError("job has no rendered prompt; conversion did not run")

    frames = round(getattr(caps, "min_clip_s", 5.0) * caps.native_fps)
    request = ClipRequest(
        shot_id=f"job{job.id}",
        mode=GenMode.T2V,
        prompt=str(rendered),
        width=caps.native_width,
        height=caps.native_height,
        duration_s=max(frames, 124) / caps.native_fps,
        fps=caps.native_fps,
        seed=job.id,
    )

    clip_job = await provider.submit(request)
    while not clip_job.is_terminal:
        # Past the bell there is no point continuing to pay: cancel and let the
        # request come back next window rather than losing both money and clip.
        if datetime.now(timezone.utc) >= window_end:
            await provider.cancel(clip_job)
            raise ProviderError("window closed while the clip was rendering")
        await asyncio.sleep(poll_interval_s)
        clip_job = await provider.poll(clip_job)

    if not clip_job.state.is_success:
        raise ProviderError(f"generation failed: {clip_job.error or clip_job.state.value}")

    dest = files_dir / f"{job.token}.mp4"
    await provider.fetch(clip_job, dest)
    return dest


async def render_image(
    job: Job,
    provider: ImageProviderLike,
    caps: Any,
    files_dir: Path,
    window_end: datetime,
    poll_interval_s: float,
) -> Path:
    """Submit, poll, fetch. One image. Mirrors `render_clip` with no frame count."""
    plan = job.prompt or {}
    rendered = plan.get("_rendered")
    if not rendered:
        raise AIStudioError("job has no rendered prompt; conversion did not run")

    request = ImageRequest(
        shot_id=f"job{job.id}",
        mode=GenMode.T2I,
        prompt=str(rendered),
        width=caps.native_width,
        height=caps.native_height,
        seed=job.id,
    )

    image_job = await provider.submit(request)
    while not image_job.is_terminal:
        if datetime.now(timezone.utc) >= window_end:
            await provider.cancel(image_job)
            raise ProviderError("window closed while the image was rendering")
        await asyncio.sleep(poll_interval_s)
        image_job = await provider.poll(image_job)

    if not image_job.state.is_success:
        raise ProviderError(f"generation failed: {image_job.error or image_job.state.value}")

    dest = files_dir / f"{job.token}.{caps.output_format}"
    await provider.fetch(image_job, dest)
    return dest
