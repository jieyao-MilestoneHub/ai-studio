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
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ai_studio.benchmark.records import msg_for, render_record
from ai_studio.config.settings import get_settings
from ai_studio.core.chat_spec import ChatRequest
from ai_studio.core.enums import GenMode, MediaKind
from ai_studio.core.errors import AIStudioError, DramaResume, ProviderError
from ai_studio.core.image_provider_spec import ImageRequest
from ai_studio.core.provider_spec import ClipRequest
from ai_studio.core.understanding_spec import UnderstandingRequest
from ai_studio.pipeline.residency import make_room_for

from fun_workflow.config.settings import get_fun_settings
from fun_workflow.core.kinds import JobKind
from fun_workflow.pipeline.convert_worker import DEFAULT_DURATION_S, snap_frames
from fun_workflow.pipeline.drama import render_drama
from fun_workflow.pipeline.queue import Job, JobQueue, JobState
from fun_workflow.prompts.chat import CHAT_THOUGHT_TOO_LONG
from fun_workflow.prompts.understanding import compose_answer

_log = logging.getLogger("ai_studio.drain")

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


class UnderstandingProviderLike(Protocol):
    def capabilities(self) -> Any: ...
    async def submit(self, request: UnderstandingRequest) -> Any: ...
    async def poll(self, job: Any) -> Any: ...
    async def fetch(self, job: Any) -> Any: ...
    async def cancel(self, job: Any) -> None: ...
    async def aclose(self) -> None: ...


class ChatProviderLike(Protocol):
    def capabilities(self) -> Any: ...
    async def submit(self, request: ChatRequest) -> Any: ...
    async def poll(self, job: Any) -> Any: ...
    async def fetch(self, job: Any) -> Any: ...
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
    providers: dict[
        MediaKind, ClipProviderLike | ImageProviderLike | UnderstandingProviderLike | ChatProviderLike
    ],
    *,
    window_end: datetime,
    files_dir: Path,
    gpu_tier: str | None = None,
    gpu_usd_per_hr: float | None = None,
    poll_interval_s: float = 15.0,
    max_clips: int | None = None,
    on_activity: Any = None,
    llm: Any = None,
) -> DrainReport:
    """Generate clips and images until the window closes or the queue empties.

    One shared pod, one FIFO queue, dispatched per job by `media_kind` —
    `providers` must have an entry for every kind the queue might contain.

    `llm` is the prompt rewriter on the pod (`pipeline.pod_llm.PodLlmClient`)
    or None; queued requests are converted with it up front, after evicting
    ComfyUI's checkpoint once, so the batch pays one gpt-oss load. The worker
    loop does the same in `worker.prepare`; this is the operator's manual
    equivalent.

    `on_activity` is called after each render so the idle reaper's timer
    resets — otherwise a long-running window looks idle and gets closed
    underneath us.
    """
    report = DrainReport(gpu_tier=gpu_tier)
    files_dir = Path(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    caps_by_kind = {kind: provider.capabilities() for kind, provider in providers.items()}

    if any(j.state is JobState.QUEUED for j in queue.pending()):
        from fun_workflow.pipeline.convert_worker import convert_pending

        if llm is not None:
            await make_room_for(providers[MediaKind.CHAT], providers)
        await convert_pending(queue, llm)

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

        job = queue.claim_next(gpu_tier=gpu_tier, usd_per_hr=gpu_usd_per_hr)
        if job is None:
            break

        # A drama drives the IMAGE and VIDEO providers itself: there is no
        # `providers[DRAMA]` entry, and indexing for one here was a KeyError
        # outside the try -- the job stayed `running` and nobody was told.
        model_kind = job.media_kind.model_kind
        provider: Any = providers.get(model_kind) if model_kind else None
        caps = caps_by_kind.get(model_kind) if model_kind else None
        if provider is None and job.media_kind is not JobKind.DRAMA:
            queue.fail(job.id, f"this pod serves no provider for {job.media_kind.value}")
            report.failed += 1
            continue

        started = time.monotonic()
        try:
            if provider is not None:
                await make_room_for(provider, providers)
            if job.media_kind is JobKind.IMAGE:
                result: Any = await render_image(
                    job, provider, caps, files_dir, window_end, poll_interval_s
                )
            elif job.media_kind is JobKind.VIDEO:
                result = await render_clip(job, provider, caps, files_dir, window_end, poll_interval_s)
            elif job.media_kind is JobKind.CHAT:
                result = await render_chat(job, provider, queue, window_end, poll_interval_s)
            elif job.media_kind is JobKind.DRAMA:
                result = await render_drama(
                    job, providers, files_dir=files_dir, runs_dir=get_settings().runs_dir,
                    deadline=window_end, poll_interval_s=poll_interval_s, on_activity=on_activity,
                )
            elif job.media_kind.is_understanding:
                result = await render_understanding(job, provider, window_end, poll_interval_s)
            else:
                raise AIStudioError(f"no renderer for media kind {job.media_kind!r}")
        except DramaResume as exc:
            # A designed stop at the lease boundary, not a failure: requeue
            # with the attempt handed back (see worker._run_one).
            queue.fail(job.id, f"resume: {exc}", requeue=True, uncounted=True)
            report.requeued += 1
            break  # the window is ending; nothing else will fit either
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

        if job.media_kind.is_understanding or job.media_kind is JobKind.CHAT:
            queue.complete_text(job.id, result)
        else:
            queue.complete(job.id, str(result))
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

    # The clip is as long as the prompt was planned for: conversion chose
    # `duration_s` and placed every cut inside it, so rendering a different
    # length would put cuts outside the clip. Snapped to the model's frame
    # grid, then the prompt's own value is the single source of that number.
    planned_s = float(plan.get("duration_s") or DEFAULT_DURATION_S)
    frames = snap_frames(round(planned_s * caps.native_fps))
    request = ClipRequest(
        shot_id=f"job{job.id}",
        mode=GenMode.I2V if job.first_frame_path else GenMode.T2V,
        prompt=str(rendered),
        width=caps.native_width,
        height=caps.native_height,
        duration_s=frames / caps.native_fps,
        fps=caps.native_fps,
        seed=job.id,
        first_frame_path=job.first_frame_path,
    )

    _submitted = time.monotonic()
    clip_job = await provider.submit(request)
    _log.info("submitted clip", extra={"stage": "render", "pod_job": getattr(clip_job, "job_id", None), "model": getattr(getattr(provider, "capabilities", lambda: None)(), "model_id", None)})
    _polls = 0
    while not clip_job.is_terminal:
        # Past the bell there is no point continuing to pay: cancel and let the
        # request come back next window rather than losing both money and clip.
        if datetime.now(timezone.utc) >= window_end:
            await provider.cancel(clip_job)
            raise ProviderError("window closed while the clip was rendering")
        await asyncio.sleep(poll_interval_s)
        clip_job = await provider.poll(clip_job)
        _polls += 1

    if not clip_job.state.is_success:
        raise ProviderError(f"generation failed: {clip_job.error or clip_job.state.value}")

    dest = files_dir / f"{job.token}.mp4"
    asset = await provider.fetch(clip_job, dest)
    _log.info(
        msg_for("video"),
        extra=render_record(
            "video", seconds=time.monotonic() - _submitted, polls=_polls,
            cost_usd=asset.cost_usd, vram_gb=asset.peak_vram_gb, gpu_tier=job.gpu_tier,
            frames=frames,
        ),
    )
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

    # `first_frame_path` is the queue's one "photo attached to this request"
    # column; for an image job it is the picture to re-render, not a frame.
    request = ImageRequest(
        shot_id=f"job{job.id}",
        mode=GenMode.I2I if job.first_frame_path else GenMode.T2I,
        prompt=str(rendered),
        width=caps.native_width,
        height=caps.native_height,
        seed=job.id,
        source_image_path=job.first_frame_path,
    )

    _submitted = time.monotonic()
    image_job = await provider.submit(request)
    _log.info("submitted image", extra={"stage": "render", "pod_job": getattr(image_job, "job_id", None), "model": getattr(getattr(provider, "capabilities", lambda: None)(), "model_id", None)})
    _polls = 0
    while not image_job.is_terminal:
        if datetime.now(timezone.utc) >= window_end:
            await provider.cancel(image_job)
            raise ProviderError("window closed while the image was rendering")
        await asyncio.sleep(poll_interval_s)
        image_job = await provider.poll(image_job)
        _polls += 1

    if not image_job.state.is_success:
        raise ProviderError(f"generation failed: {image_job.error or image_job.state.value}")

    dest = files_dir / f"{job.token}.{caps.output_format}"
    asset = await provider.fetch(image_job, dest)
    _log.info(
        msg_for("image"),
        extra=render_record(
            "image", seconds=time.monotonic() - _submitted, polls=_polls,
            cost_usd=asset.cost_usd, vram_gb=asset.peak_vram_gb, gpu_tier=job.gpu_tier,
        ),
    )
    return dest


async def render_understanding(
    job: Job,
    provider: UnderstandingProviderLike,
    window_end: datetime,
    poll_interval_s: float,
) -> str:
    """Submit, poll, fetch. One description.

    Unlike `render_clip`/`render_image` there is no output file: the result
    is text, returned directly rather than written under `files_dir`. No
    `caps` argument either -- an understanding request has no native
    width/height/fps to bind, only the input media path already saved by the
    webhook.
    """
    if not job.input_media_path:
        raise AIStudioError("understanding job has no input_media_path to describe")

    # The questions were built at conversion (prompts/understanding.py): the
    # engineered defaults, or the user's own question rewritten. None means
    # "no question" -- the server's caption path for a photo. Video carries a
    # second question for the audio model.
    plan = job.prompt or {}
    question = plan.get("_question") or None
    audio_question = plan.get("_audio_question") or None
    modality = job.media_kind.model_kind
    if modality is None or not job.media_kind.is_understanding:
        raise AIStudioError(f"{job.media_kind.value} is not an understanding kind")
    request = UnderstandingRequest(
        shot_id=f"job{job.id}",
        modality=modality,
        input_media_path=job.input_media_path,
        prompt=str(question) if question else None,
        audio_prompt=str(audio_question) if audio_question else None,
    )

    _submitted = time.monotonic()
    understanding_job = await provider.submit(request)
    _log.info("submitted understanding", extra={"stage": "render", "pod_job": getattr(understanding_job, "job_id", None), "model": getattr(getattr(provider, "capabilities", lambda: None)(), "model_id", None)})
    _polls = 0
    while not understanding_job.is_terminal:
        if datetime.now(timezone.utc) >= window_end:
            await provider.cancel(understanding_job)
            raise ProviderError("window closed while the description was running")
        await asyncio.sleep(poll_interval_s)
        understanding_job = await provider.poll(understanding_job)
        _polls += 1

    if not understanding_job.state.is_success:
        raise ProviderError(
            f"understanding failed: {understanding_job.error or understanding_job.state.value}"
        )

    asset = await provider.fetch(understanding_job)
    _log.info("fetched understanding", extra={"stage": "render", "seconds": round(time.monotonic() - _submitted, 1), "polls": _polls, "cost_usd": getattr(asset, "cost_usd", None)})
    return compose_answer(asset)


async def render_chat(
    job: Job,
    provider: ChatProviderLike,
    queue: JobQueue,
    window_end: datetime,
    poll_interval_s: float,
) -> str:
    """Submit, poll, fetch. One chat reply.

    Fetches this user's recent turns before submitting and appends the new
    pair back on success -- host-side, never on the ephemeral pod (see
    `core.chat_spec`'s module docstring). `job.user_id` can be None for a
    LINE user who has not accepted the Official Account terms; there is then
    no identity to key memory on, so the turn is answered statelessly rather
    than persisted or cross-mixed with anyone else's -- the same "None is
    never counted" precedent `JobQueue.accepted_today()` already follows.
    No `caps` argument, like `render_understanding` -- a chat reply has no
    native width/height/fps to bind.

    Checks the monthly chat sub-budget *before* submitting anything, and
    fails loudly (terminal, not requeued) rather than leaving the job
    silently stuck if it is exhausted: video/image keep running on the same
    shared $/month ceiling regardless, and the user is told plainly rather
    than watching the request vanish into an unclaimable backlog.
    """
    settings = get_fun_settings()
    if settings.max_chat_month_usd and queue.chat_spent_this_month_usd() >= settings.max_chat_month_usd:
        raise AIStudioError(
            f"本月 /himonkey 用量已達上限(${settings.max_chat_month_usd:.2f}),"
            "下個月再試。影片與圖片功能不受影響。"
        )

    history = queue.recent_chat_turns(job.user_id) if job.user_id else []

    extra: dict[str, Any] = {}
    if history:
        extra["history"] = json.dumps(history, ensure_ascii=False)
    # The developer prompt chosen at conversion (prompts/chat.py). Absent on
    # a row parsed before it existed -- the server then answers with none.
    system = (job.prompt or {}).get("_system")
    if system:
        extra["system"] = str(system)
    request = ChatRequest(shot_id=f"job{job.id}", text=job.text, extra=extra)

    _submitted = time.monotonic()
    chat_job = await provider.submit(request)
    _log.info("submitted chat", extra={"stage": "render", "pod_job": getattr(chat_job, "job_id", None), "model": getattr(getattr(provider, "capabilities", lambda: None)(), "model_id", None)})
    _polls = 0
    while not chat_job.is_terminal:
        if datetime.now(timezone.utc) >= window_end:
            await provider.cancel(chat_job)
            raise ProviderError("window closed while the reply was generating")
        await asyncio.sleep(poll_interval_s)
        chat_job = await provider.poll(chat_job)
        _polls += 1

    if not chat_job.state.is_success:
        raise ProviderError(f"chat failed: {chat_job.error or chat_job.state.value}")

    asset = await provider.fetch(chat_job)
    _log.info("fetched chat", extra={"stage": "render", "seconds": round(time.monotonic() - _submitted, 1), "polls": _polls, "cost_usd": getattr(asset, "cost_usd", None)})
    result = CHAT_THOUGHT_TOO_LONG if asset.reasoning_exhausted else str(asset.result_text)
    queue.record_chat_cost(job.id, asset.cost_usd)
    if job.user_id:
        queue.append_chat_turn(job.user_id, job.group_id, "user", job.text)
        queue.append_chat_turn(job.user_id, job.group_id, "assistant", result)
    return result
