"""Window draining.

Every assertion here is about not wasting GPU money: not starting work that
cannot finish, not losing a request to a machine failure, not paying past the
bell.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ai_studio.core.enums import GenMode, MediaKind
from ai_studio.core.enums import JobState as ClipState
from ai_studio.core.image_provider_spec import ImageAsset, ImageProviderCapabilities
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ProviderCapabilities
from ai_studio.pipeline.drain import STOP_CLAIMING_BEFORE_S, drain_window
from ai_studio.pipeline.queue import JobQueue

CAPS = ProviderCapabilities(
    provider="fake",
    model_id="fake-h3",
    native_width=864,
    native_height=480,
    native_fps=24,
    modes=frozenset({GenMode.T2V}),
    min_clip_s=5.0,
    max_clip_s=15.0,
    has_native_audio=True,
)

IMAGE_CAPS = ImageProviderCapabilities(
    provider="fake-flux",
    model_id="fake-flux-dev",
    native_width=1024,
    native_height=1024,
    modes=frozenset({GenMode.T2I}),
    output_format="png",
)

PROMPT = {"_rendered": "integrated_multimodal_description: [Shot 1] a cat", "_built_by": "llm"}
IMAGE_PROMPT = {"_rendered": "a cat", "_built_by": "template"}


def _video(provider: Any) -> dict[MediaKind, Any]:
    """Wrap a single video provider as the `providers` dict `drain_window` now
    expects — most tests here only ever exercise one media kind."""
    return {MediaKind.VIDEO: provider}


class FakeProvider:
    """Completes instantly, or fails on demand."""

    def __init__(self, *, fail_with: Exception | None = None, never_finish: bool = False) -> None:
        self.fail_with = fail_with
        self.never_finish = never_finish
        self.submitted: list[str] = []
        self.requests: list[Any] = []
        self.cancelled = 0

    def capabilities(self) -> ProviderCapabilities:
        return CAPS

    async def submit(self, request: Any) -> ClipJob:
        self.submitted.append(request.shot_id)
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        state = ClipState.RUNNING if self.never_finish else ClipState.COMPLETED
        return ClipJob(
            provider="fake", job_id=f"j-{request.shot_id}", shot_id=request.shot_id,
            state=state, submitted_at=0.0, updated_at=0.0,
        )

    async def poll(self, job: ClipJob) -> ClipJob:
        return job

    async def fetch(self, job: ClipJob, dest: Path) -> ClipAsset:
        Path(dest).write_bytes(b"\x00\x00\x00 ftypisom")
        return ClipAsset(
            shot_id=job.shot_id, key=Path(dest).name, sha256="0" * 64, size_bytes=16,
            width=864, height=480, fps=24.0, duration_s=5.17, has_audio=True,
            provider="fake", job_id=job.job_id,
        )

    async def cancel(self, job: ClipJob) -> None:
        self.cancelled += 1

    async def aclose(self) -> None:
        return None


class FakeImageProvider:
    """The image-side counterpart to `FakeProvider`. Completes instantly."""

    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.requests: list[Any] = []

    def capabilities(self) -> ImageProviderCapabilities:
        return IMAGE_CAPS

    async def submit(self, request: Any) -> ClipJob:
        self.submitted.append(request.shot_id)
        self.requests.append(request)
        return ClipJob(
            provider="fake-flux", job_id=f"j-{request.shot_id}", shot_id=request.shot_id,
            state=ClipState.COMPLETED, submitted_at=0.0, updated_at=0.0,
        )

    async def poll(self, job: ClipJob) -> ClipJob:
        return job

    async def fetch(self, job: ClipJob, dest: Path) -> ImageAsset:
        Path(dest).write_bytes(b"\x89PNG\r\n\x1a\n")
        return ImageAsset(
            shot_id=job.shot_id, key=Path(dest).name, sha256="0" * 64, size_bytes=8,
            width=1024, height=1024, format="png", provider="fake-flux", job_id=job.job_id,
        )

    async def cancel(self, job: ClipJob) -> None:
        return None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def ready(tmp_path: Path):
    """A queue with three parsed, claimable jobs."""
    q = JobQueue(tmp_path / "q.sqlite3")
    for i in range(3):
        job, _ = q.enqueue(f"evt-{i}", "Cgroup", f"貓 {i}")
        q.set_parsed(job.id, PROMPT)
    yield q, tmp_path / "files"
    q.close()


def _end(minutes: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


# ------------------------------------------------------------------- happy


@pytest.mark.asyncio
async def test_a_full_window_drains_the_queue(ready) -> None:
    q, files = ready
    report = await drain_window(
        q, _video(FakeProvider()), window_end=_end(60), files_dir=files, gpu_tier="L40S/COMMUNITY"
    )

    assert report.completed == 3
    assert report.failed == 0
    assert q.counts().get("done") == 3
    assert len(list(files.glob("*.mp4"))) == 3


@pytest.mark.asyncio
async def test_the_serving_tier_is_recorded_on_each_job(ready) -> None:
    """Falling down the capacity ladder changes quality, so it must be visible."""
    q, files = ready
    await drain_window(
        q, _video(FakeProvider()), window_end=_end(60), files_dir=files, gpu_tier="4090/COMMUNITY"
    )
    assert all(j.gpu_tier == "4090/COMMUNITY" for j in q.recent())


@pytest.mark.asyncio
async def test_max_clips_bounds_a_measurement_run(ready) -> None:
    q, files = ready
    report = await drain_window(
        q, _video(FakeProvider()), window_end=_end(60), files_dir=files, max_clips=2
    )
    assert report.completed == 2
    assert q.counts().get("parsed") == 1


# ------------------------------------------------------------- mixed kinds


@pytest.mark.asyncio
async def test_video_and_image_jobs_share_the_queue_and_dispatch_correctly(tmp_path: Path) -> None:
    """One shared pod, one FIFO queue — dispatch is entirely by media_kind."""
    with JobQueue(tmp_path / "q.sqlite3") as q:
        video_job, _ = q.enqueue("evt-v", "Cgroup", "貓", media_kind=MediaKind.VIDEO)
        q.set_parsed(video_job.id, PROMPT)
        image_job, _ = q.enqueue("evt-i", "Cgroup", "貓", media_kind=MediaKind.IMAGE)
        q.set_parsed(image_job.id, IMAGE_PROMPT)

        video_provider = FakeProvider()
        image_provider = FakeImageProvider()
        files = tmp_path / "files"
        report = await drain_window(
            q,
            {MediaKind.VIDEO: video_provider, MediaKind.IMAGE: image_provider},
            window_end=_end(60),
            files_dir=files,
        )

        assert report.completed == 2
        assert video_provider.submitted == [f"job{video_job.id}"]
        assert image_provider.submitted == [f"job{image_job.id}"]
        assert (files / f"{video_job.token}.mp4").is_file()
        assert (files / f"{image_job.token}.png").is_file()


# --------------------------------------------------------- the closing bell


@pytest.mark.asyncio
async def test_no_new_work_is_claimed_inside_the_reserved_tail(ready) -> None:
    """Starting a 5-minute render 3 minutes before the bell wastes the money."""
    q, files = ready
    provider = FakeProvider()

    report = await drain_window(
        q, _video(provider), window_end=_end(STOP_CLAIMING_BEFORE_S / 60 - 5), files_dir=files
    )

    assert report.completed == 0
    assert provider.submitted == [], "nothing should even be submitted"
    assert q.counts().get("parsed") == 3, "all three requests survive for next window"


@pytest.mark.asyncio
async def test_a_clip_still_running_at_the_bell_is_cancelled_and_requeued(ready) -> None:
    q, files = ready
    provider = FakeProvider(never_finish=True)

    # Window ends immediately, but claiming is allowed for this first pass.
    report = await drain_window(
        q, _video(provider), window_end=_end(-1), files_dir=files, poll_interval_s=0.01
    )

    assert report.completed == 0
    assert provider.cancelled == 0, "claiming stops before the bell, so nothing starts"
    assert q.counts().get("parsed") == 3


# ------------------------------------------------------------- resilience


@pytest.mark.asyncio
async def test_a_provider_failure_requeues_rather_than_losing_the_request(ready) -> None:
    """The machine's fault is not the requester's fault."""
    from ai_studio.core.errors import ProviderError

    q, files = ready
    report = await drain_window(
        q, _video(FakeProvider(fail_with=ProviderError("pod preempted"))),
        window_end=_end(60), files_dir=files,
    )

    # claim_next always takes the oldest, so the same request is retried until
    # its attempt budget runs out (requeue, requeue, then terminal), and the
    # breaker then stops the window. The other two are never touched, which is
    # the point: they survive intact for the next window instead of being
    # ground through a pod that is evidently broken.
    assert report.completed == 0
    assert (report.requeued, report.failed) == (2, 1)
    assert report.stopped_early is not None
    assert q.counts().get("parsed") == 2, "the untouched requests survive"
    assert q.counts().get("failed") == 1, "the one that used up its attempts"


@pytest.mark.asyncio
async def test_a_request_level_failure_is_terminal(tmp_path: Path) -> None:
    """A job with no rendered prompt would fail identically forever."""
    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-x", "Cgroup", "貓")
        q.set_parsed(job.id, {"no_rendered_key": True})

        report = await drain_window(
            q, _video(FakeProvider()), window_end=_end(60), files_dir=tmp_path / "files"
        )

        assert report.failed == 1
        assert report.requeued == 0
        assert q.counts().get("failed") == 1


@pytest.mark.asyncio
async def test_jobs_orphaned_by_a_dead_pod_are_reclaimed_at_window_open(tmp_path: Path) -> None:
    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-orphan", "Cgroup", "貓")
        q.set_parsed(job.id, PROMPT)
        q.claim_next()  # left running by a window that died

        report = await drain_window(
            q, _video(FakeProvider()), window_end=_end(60), files_dir=tmp_path / "files"
        )

        assert report.requeued >= 1
        assert report.completed == 1, "the reclaimed job then runs normally"


# ------------------------------------------------------------ measurement


@pytest.mark.asyncio
async def test_the_report_separates_the_first_clip_from_later_ones(ready) -> None:
    """Whether clip two is faster is what decides if a window amortises setup."""
    q, files = ready
    report = await drain_window(q, _video(FakeProvider()), window_end=_end(60), files_dir=files)

    assert report.first_clip_s is not None
    assert len(report.later_clips_s) == 2
    assert "completed=3" in report.summary()


@pytest.mark.asyncio
async def test_activity_is_reported_so_the_idle_reaper_does_not_close_a_busy_window(
    ready,
) -> None:
    q, files = ready
    beats = []
    await drain_window(
        q, _video(FakeProvider()), window_end=_end(60), files_dir=files,
        on_activity=lambda: beats.append(1),
    )
    assert len(beats) == 3


@pytest.mark.asyncio
async def test_an_empty_queue_returns_immediately(tmp_path: Path) -> None:
    with JobQueue(tmp_path / "q.sqlite3") as q:
        report = await drain_window(
            q, _video(FakeProvider()), window_end=_end(60), files_dir=tmp_path / "files"
        )
        assert report.completed == 0 and report.failed == 0


def test_the_prompt_payload_shape_the_drainer_expects(ready) -> None:
    """Guards the contract between convert_worker and drain."""
    q, _ = ready
    stored = json.loads(q.recent()[0].prompt_json or "{}")
    assert "_rendered" in stored


@pytest.mark.asyncio
async def test_requeue_is_capped_so_a_broken_pod_cannot_loop_forever(tmp_path: Path) -> None:
    """Without a cap, requeue is an infinite loop: fail, requeue, reclaim, fail."""
    from ai_studio.core.errors import ProviderError
    from ai_studio.pipeline.drain import MAX_ATTEMPTS

    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-loop", "Cgroup", "貓")
        q.set_parsed(job.id, PROMPT)

        # Burn through the attempt budget so the next failure must be terminal.
        for _ in range(MAX_ATTEMPTS):
            q.claim_next()
            q.fail(job.id, "provider: boom", requeue=True)

        report = await drain_window(
            q, _video(FakeProvider(fail_with=ProviderError("still broken"))),
            window_end=_end(60), files_dir=tmp_path / "files",
        )

        assert report.failed == 1
        assert report.requeued == 0
        assert q.counts().get("failed") == 1


@pytest.mark.asyncio
async def test_the_breaker_stops_the_window_after_three_failures_in_a_row(
    tmp_path: Path,
) -> None:
    """Three in a row is a broken pod, not bad luck. Stop paying."""
    from ai_studio.pipeline.drain import MAX_CONSECUTIVE_FAILURES

    with JobQueue(tmp_path / "q.sqlite3") as q:
        for i in range(6):
            job, _ = q.enqueue(f"evt-b{i}", "Cgroup", f"貓 {i}")
            q.set_parsed(job.id, {"no_rendered_key": True})  # terminal every time

        report = await drain_window(
            q, _video(FakeProvider()), window_end=_end(60), files_dir=tmp_path / "files"
        )

        assert report.failed == MAX_CONSECUTIVE_FAILURES
        assert report.stopped_early is not None
        assert q.counts().get("parsed") == 6 - MAX_CONSECUTIVE_FAILURES, "the rest survive"


# ------------------------------------------------------------- clip length


@pytest.mark.asyncio
async def test_the_clip_is_as_long_as_the_prompt_was_planned_for(tmp_path: Path) -> None:
    """Conversion places every cut inside `duration_s`; rendering another
    length would put cuts outside the clip. The default is 243 frames
    (10.1 s), measured to cost the same time and VRAM as 124 on a 4090."""
    from ai_studio.pipeline.convert_worker import DEFAULT_FRAMES

    with JobQueue(tmp_path / "q.sqlite3") as q:
        job, _ = q.enqueue("evt-len", "Cgroup", "貓")
        q.set_parsed(job.id, PROMPT)
        provider = FakeProvider()
        await drain_window(
            q, {MediaKind.VIDEO: provider, MediaKind.IMAGE: FakeImageProvider()},
            window_end=_end(60), files_dir=tmp_path / "files",
        )
    (request,) = provider.requests
    assert round(request.duration_s * request.fps) == DEFAULT_FRAMES


def test_frame_counts_snap_up_to_the_17k_plus_5_grid() -> None:
    from ai_studio.pipeline.convert_worker import MIN_FRAMES, snap_frames

    assert snap_frames(124) == 124
    assert snap_frames(125) == 141
    assert snap_frames(240) == 243
    assert snap_frames(243) == 243
    assert snap_frames(10) == MIN_FRAMES
