"""The always-on worker.

Every assertion here is about one of two things: not opening a pod that nobody
asked for, and not losing a request that someone did. The loop's whole reason
for existing is that a timer could not tell those apart.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ai_studio.core.enums import GenMode, MediaKind
from ai_studio.core.enums import JobState as ClipState
from ai_studio.core.errors import AIStudioError, CostCeilingExceeded, ProviderError
from ai_studio.core.image_provider_spec import ImageAsset, ImageProviderCapabilities
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ProviderCapabilities
from ai_studio.pipeline import worker
from ai_studio.pipeline.drain import STOP_CLAIMING_BEFORE_S
from ai_studio.pipeline.queue import JobQueue, JobState

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


class FakeProvider:
    """Completes instantly, or fails on demand."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.submitted: list[str] = []

    def capabilities(self) -> ProviderCapabilities:
        return CAPS

    async def submit(self, request: Any) -> ClipJob:
        self.submitted.append(request.shot_id)
        if self.fail_with is not None:
            raise self.fail_with
        return ClipJob(
            provider="fake", job_id=f"j-{request.shot_id}", shot_id=request.shot_id,
            state=ClipState.COMPLETED, submitted_at=0.0, updated_at=0.0,
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
        return None

    async def aclose(self) -> None:
        return None


class FakeImageProvider(FakeProvider):
    def capabilities(self) -> ImageProviderCapabilities:  # type: ignore[override]
        return IMAGE_CAPS

    async def fetch(self, job: ClipJob, dest: Path) -> ImageAsset:  # type: ignore[override]
        Path(dest).write_bytes(b"\x89PNG\r\n\x1a\n")
        return ImageAsset(
            shot_id=job.shot_id, key=Path(dest).name, sha256="0" * 64, size_bytes=8,
            width=1024, height=1024, format="png", provider="fake-flux", job_id=job.job_id,
        )


class FakeSession:
    pod_id = "pod-1"
    tier_label = "L40S/SECURE"


class FakeHost:
    """The runtime half of the seam, with no pod and no clock.

    Counting `opens` is the point of most of this file: the number of times a
    tick reaches `ensure_pod` is the number of times a real run would have
    created a machine.
    """

    def __init__(
        self,
        *,
        open_now: bool = True,
        video: FakeProvider | None = None,
        image: FakeProvider | None = None,
        ensure_raises: Exception | None = None,
        minutes_left: float = 120.0,
    ) -> None:
        self.open_now = open_now
        self.ensure_raises = ensure_raises
        self.minutes_left = minutes_left
        self.video = video or FakeProvider()
        self.image = image or FakeImageProvider()
        self.opens = 0
        self.waits = 0
        self.touches = 0
        self.session = FakeSession()

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def is_open(self, now: datetime | None = None) -> bool:
        return self.open_now

    def claim_deadline(self, now: datetime | None = None) -> datetime:
        return datetime.now(timezone.utc) + timedelta(minutes=self.minutes_left)

    def ensure_pod(self, queue: Any) -> Any:
        if self.ensure_raises is not None:
            raise self.ensure_raises
        self.opens += 1
        return self.session

    def wait_ready(self, session: Any) -> float:
        self.waits += 1
        return 0.0

    def providers_for(self, session: Any) -> dict[MediaKind, Any]:
        return {MediaKind.VIDEO: self.video, MediaKind.IMAGE: self.image}

    def touch_activity(self) -> None:
        self.touches += 1


@pytest.fixture
def queue(tmp_path: Path):
    q = JobQueue(tmp_path / "q.sqlite3")
    yield q
    q.close()


def _parsed(q: JobQueue, text: str = "a cat", *, kind: MediaKind = MediaKind.VIDEO) -> Any:
    job, _ = q.enqueue(f"evt-{text}-{kind.value}", "Cgroup", text, user_id="U1", media_kind=kind)
    q.set_parsed(job.id, PROMPT)
    return q.by_token(job.token)


def _files(tmp_path: Path) -> Path:
    """`serve` creates this once at startup, so a bare `tick` is handed one
    that already exists — same contract, without running the loop."""
    directory = tmp_path / "files"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def _tick(q: JobQueue, host: FakeHost, tmp_path: Path) -> tuple[str, worker.WorkerReport]:
    report = worker.WorkerReport()
    action = await worker.tick(q, host, files_dir=_files(tmp_path), report=report)
    return action, report


# ------------------------------------------------------- not opening a pod


@pytest.mark.asyncio
async def test_outside_business_hours_nothing_is_opened(queue, tmp_path: Path) -> None:
    """The single most expensive bug available here: a pod at 03:00."""
    _parsed(queue)
    host = FakeHost(open_now=False)

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "closed"
    assert host.opens == 0


@pytest.mark.asyncio
async def test_an_empty_queue_does_not_open_a_pod(queue, tmp_path: Path) -> None:
    """Request-driven means exactly this: no request, no machine."""
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "idle"
    assert host.opens == 0


@pytest.mark.asyncio
async def test_an_unconverted_request_does_not_open_a_pod(queue, tmp_path: Path) -> None:
    """`queued` is not `parsed`. A request whose prompt does not exist yet may
    never become a valid one, and a pod opened for it is paid for either way."""
    queue.enqueue("evt-1", "Cgroup", "a cat", user_id="U1")
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "idle"
    assert host.opens == 0
    assert worker.claimable(queue) == []


@pytest.mark.asyncio
async def test_no_new_work_is_started_inside_the_closing_reserve(queue, tmp_path: Path) -> None:
    """A render begun at 12:58 is billed and then thrown away by the lease."""
    _parsed(queue)
    host = FakeHost(minutes_left=(STOP_CLAIMING_BEFORE_S / 60) - 1)

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "too-late"
    assert host.opens == 0
    assert queue.by_token(worker.claimable(queue)[0].token).state is JobState.PARSED


# ------------------------------------------------------------- doing the work


@pytest.mark.asyncio
async def test_a_parsed_request_opens_a_pod_and_renders(queue, tmp_path: Path) -> None:
    job = _parsed(queue)
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "completed"
    assert host.opens == 1 and host.waits == 1
    assert queue.by_token(job.token).state is JobState.DONE
    assert Path(queue.by_token(job.token).output_path).is_file()


@pytest.mark.asyncio
async def test_a_render_resets_the_idle_reapers_timer(queue, tmp_path: Path) -> None:
    """Without this a long render looks like idleness and the reaper closes the
    window out from under the next job."""
    _parsed(queue)
    host = FakeHost()

    await _tick(queue, host, tmp_path)

    assert host.touches == 1


@pytest.mark.asyncio
async def test_a_second_request_reuses_the_open_pod(queue, tmp_path: Path) -> None:
    """`ensure_pod` is called per tick but must not create per tick — the
    reuse is what keeps a two-request day from costing two setups."""
    _parsed(queue, "cat one")
    _parsed(queue, "cat two")
    host = FakeHost()

    report = worker.WorkerReport()
    for _ in range(2):
        await worker.tick(queue, host, files_dir=_files(tmp_path), report=report)

    assert report.completed == 2
    assert host.opens == 2, "the host is asked every tick; dedupe belongs in ensure_pod"
    assert len(host.video.submitted) == 2


@pytest.mark.asyncio
async def test_an_image_request_goes_to_the_image_backend(queue, tmp_path: Path) -> None:
    """One queue, two models, dispatched by `media_kind`."""
    job = _parsed(queue, "a fox", kind=MediaKind.IMAGE)
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "completed"
    assert host.image.submitted and not host.video.submitted
    assert queue.by_token(job.token).output_path.endswith(".png")


# ------------------------------------------------------------------ failure


@pytest.mark.asyncio
async def test_a_backend_failure_keeps_the_request(queue, tmp_path: Path) -> None:
    """The window closed, the proxy hiccuped — the machine's problem, not the
    request's. It goes back in the queue rather than being lost."""
    job = _parsed(queue)
    host = FakeHost(video=FakeProvider(fail_with=ProviderError("proxy hiccup")))

    action, report = await _tick(queue, host, tmp_path)

    assert action == "requeued"
    assert report.requeued == 1
    assert queue.by_token(job.token).state is JobState.PARSED


@pytest.mark.asyncio
async def test_a_request_stops_being_retried_after_max_attempts(queue, tmp_path: Path) -> None:
    """Without a cap, requeue is an infinite loop that burns the whole day."""
    job = _parsed(queue)
    host = FakeHost(video=FakeProvider(fail_with=ProviderError("still broken")))

    report = worker.WorkerReport()
    for _ in range(4):
        await worker.tick(queue, host, files_dir=_files(tmp_path), report=report)

    assert queue.by_token(job.token).state is JobState.FAILED
    assert report.failed == 1


@pytest.mark.asyncio
async def test_a_request_the_model_rejects_fails_immediately(queue, tmp_path: Path) -> None:
    """It would repeat identically, so retrying it only spends more GPU."""
    job = _parsed(queue)
    host = FakeHost(video=FakeProvider(fail_with=AIStudioError("prompt rejected")))

    action, report = await _tick(queue, host, tmp_path)

    assert action == "failed"
    assert report.failed == 1
    assert queue.by_token(job.token).state is JobState.FAILED


# -------------------------------------------------------------------- serve


@pytest.mark.asyncio
async def test_serve_reclaims_work_left_running_by_a_dead_worker(
    queue, tmp_path: Path
) -> None:
    """A job left `running` is one whose worker died mid-render. Without this
    it sits there forever and the user waits forever."""
    job = _parsed(queue)
    queue.claim_next()
    assert queue.by_token(job.token).state is JobState.RUNNING

    host = FakeHost(open_now=False)
    report = await worker.serve(
        queue, host, files_dir=tmp_path / "files", max_ticks=1, sleep=_no_sleep
    )

    assert report.requeued == 1
    assert queue.by_token(job.token).state is JobState.PARSED


@pytest.mark.asyncio
async def test_a_refused_tick_does_not_kill_the_worker(queue, tmp_path: Path) -> None:
    """Hours, budget, the daily open cap — a worker that exits on a refusal is
    one somebody has to remember to restart before eleven tomorrow."""
    _parsed(queue)
    host = FakeHost(ensure_raises=CostCeilingExceeded("2 pods already opened today"))

    report = await worker.serve(
        queue, host, files_dir=tmp_path / "files", max_ticks=3, sleep=_no_sleep
    )

    assert report.ticks == 3
    assert report.last_action == "refused"


@pytest.mark.asyncio
async def test_serve_sleeps_the_long_interval_when_closed(queue, tmp_path: Path) -> None:
    """Outside business hours the loop does exactly one thing."""
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    host = FakeHost(open_now=False)
    await worker.serve(
        queue, host, files_dir=tmp_path / "files", max_ticks=2,
        closed_poll_s=60.0, idle_poll_s=10.0, sleep=record,
    )

    assert slept == [60.0, 60.0]
    assert host.opens == 0


@pytest.mark.asyncio
async def test_serve_does_not_sleep_between_completed_jobs(queue, tmp_path: Path) -> None:
    """The pod is hot and already paid for; waiting ten seconds per job on it
    is the one delay with no upside."""
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    _parsed(queue, "cat one")
    _parsed(queue, "cat two")
    host = FakeHost()

    report = await worker.serve(
        queue, host, files_dir=tmp_path / "files", max_ticks=2,
        idle_poll_s=10.0, sleep=record,
    )

    assert report.completed == 2
    assert slept == []


async def _no_sleep(seconds: float) -> None:
    return None
