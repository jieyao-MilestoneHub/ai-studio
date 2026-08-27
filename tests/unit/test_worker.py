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

from ai_studio.core.chat_spec import ChatAsset, ChatCapabilities, ChatJob
from ai_studio.core.enums import GenMode, MediaKind
from ai_studio.core.enums import JobState as ClipState
from ai_studio.core.errors import AIStudioError, CostCeilingExceeded, ProviderError
from ai_studio.core.image_provider_spec import ImageAsset, ImageProviderCapabilities
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ProviderCapabilities
from ai_studio.core.understanding_spec import (
    UnderstandingAsset,
    UnderstandingCapabilities,
    UnderstandingJob,
)
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

UNDERSTANDING_CAPS = UnderstandingCapabilities(
    provider="fake-understanding",
    model_id="fake-moondream",
    modality=MediaKind.IMAGE_UNDERSTAND,
)

CHAT_CAPS = ChatCapabilities(provider="fake-chat", model_id="fake-gpt-oss-20b")

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


class FakeUnderstandingProvider:
    """Completes instantly with a canned description, or fails on demand.

    Deliberately has no `evict()` -- `make_room_for` must tolerate a provider
    that does not implement the GPU hand-off (the offline path has nothing
    to evict) rather than requiring every provider to define it.
    """

    def __init__(
        self, *, fail_with: Exception | None = None, result_text: str = "a photo of a cat"
    ) -> None:
        self.fail_with = fail_with
        self.result_text = result_text
        self.submitted: list[str] = []

    def capabilities(self) -> UnderstandingCapabilities:
        return UNDERSTANDING_CAPS

    async def submit(self, request: Any) -> UnderstandingJob:
        self.submitted.append(request.shot_id)
        self.prompts: list[str | None] = getattr(self, "prompts", [])
        self.prompts.append(request.prompt)
        if self.fail_with is not None:
            raise self.fail_with
        return UnderstandingJob(
            provider="fake-understanding", job_id=f"j-{request.shot_id}",
            shot_id=request.shot_id, state=ClipState.COMPLETED,
            submitted_at=0.0, updated_at=0.0,
        )

    async def poll(self, job: UnderstandingJob) -> UnderstandingJob:
        return job

    async def fetch(self, job: UnderstandingJob) -> UnderstandingAsset:
        return UnderstandingAsset(
            shot_id=job.shot_id, provider="fake-understanding", job_id=job.job_id,
            modality=MediaKind.IMAGE_UNDERSTAND, result_text=self.result_text,
        )

    async def cancel(self, job: UnderstandingJob) -> None:
        return None

    async def aclose(self) -> None:
        return None


class FakeChatProvider:
    """Completes instantly with a canned reply, or fails on demand.

    Deliberately has no `evict()`, same as `FakeUnderstandingProvider` --
    `make_room_for` must tolerate a provider that does not implement the GPU
    hand-off.
    """

    def __init__(self, *, fail_with: Exception | None = None, result_text: str = "hi there") -> None:
        self.fail_with = fail_with
        self.result_text = result_text
        self.submitted: list[str] = []

    def capabilities(self) -> ChatCapabilities:
        return CHAT_CAPS

    async def submit(self, request: Any) -> ChatJob:
        self.submitted.append(request.shot_id)
        self.extras: list[dict[str, Any]] = getattr(self, "extras", [])
        self.extras.append(dict(request.extra))
        if self.fail_with is not None:
            raise self.fail_with
        return ChatJob(
            provider="fake-chat", job_id=f"j-{request.shot_id}",
            shot_id=request.shot_id, state=ClipState.COMPLETED,
            submitted_at=0.0, updated_at=0.0,
        )

    async def poll(self, job: ChatJob) -> ChatJob:
        return job

    async def fetch(self, job: ChatJob) -> ChatAsset:
        return ChatAsset(
            shot_id=job.shot_id, provider="fake-chat", job_id=job.job_id,
            result_text=self.result_text,
        )

    async def cancel(self, job: ChatJob) -> None:
        return None

    async def aclose(self) -> None:
        return None


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
        
        video: FakeProvider | None = None,
        image: FakeProvider | None = None,
        understand: FakeUnderstandingProvider | None = None,
        chat: FakeChatProvider | None = None,
        ensure_raises: Exception | None = None,
        deliver_raises: Exception | None = None,
        minutes_left: float = 120.0,
        llm: Any = None,
    ) -> None:
        self.llm = llm
        self.ensure_raises = ensure_raises
        self.minutes_left = minutes_left
        self.video = video or FakeProvider()
        self.image = image or FakeImageProvider()
        self.understand = understand or FakeUnderstandingProvider()
        self.chat = chat or FakeChatProvider()
        self.opens = 0
        self.waits = 0
        self.touches = 0
        self.touched_kinds: list[str] = []
        self.session = FakeSession()
        self.delivered: list[tuple[int, str | None]] = []
        self.deliver_raises = deliver_raises

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

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

    def llm_for(self, session: Any) -> Any:
        return self.llm

    def providers_for(self, session: Any) -> dict[MediaKind, Any]:
        return {
            MediaKind.VIDEO: self.video,
            MediaKind.IMAGE: self.image,
            MediaKind.IMAGE_UNDERSTAND: self.understand,
            MediaKind.AUDIO_UNDERSTAND: self.understand,
            MediaKind.VIDEO_UNDERSTAND: self.understand,
            MediaKind.CHAT: self.chat,
        }

    def touch_activity(self, media_kind: str) -> None:
        self.touches += 1
        self.touched_kinds.append(media_kind)

    async def deliver(self, job: Any, asset: Any) -> str:
        if self.deliver_raises is not None:
            raise self.deliver_raises
        self.delivered.append((job.id, str(asset) if asset is not None else None))
        return "pushed"


@pytest.fixture
def queue(tmp_path: Path):
    q = JobQueue(tmp_path / "q.sqlite3")
    yield q
    q.close()


def _parsed(
    q: JobQueue,
    text: str = "a cat",
    *,
    kind: MediaKind = MediaKind.VIDEO,
    input_media_path: str | None = None,
) -> Any:
    job, _ = q.enqueue(
        f"evt-{text}-{kind.value}", "Cgroup", text, user_id="U1", media_kind=kind,
        input_media_path=input_media_path,
    )
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
async def test_there_are_no_business_hours(queue, tmp_path: Path) -> None:
    """A parsed request at 03:00 opens a pod like one at noon. What guards
    money now is the budget guard and the daily cap inside ensure_pod, and
    the reaper minutes after the render -- not the clock."""
    _parsed(queue)
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "completed"
    assert host.opens == 1


@pytest.mark.asyncio
async def test_an_empty_queue_does_not_open_a_pod(queue, tmp_path: Path) -> None:
    """Request-driven means exactly this: no request, no machine."""
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "idle"
    assert host.opens == 0


@pytest.mark.asyncio
async def test_an_unconverted_request_is_converted_on_the_pod_then_rendered(
    queue, tmp_path: Path
) -> None:
    """The rewriter is gpt-oss on the pod, so a queued request opens the pod,
    is converted in the prepare phase, and is rendered in the same tick.
    With no LLM bound (`llm_for` -> None) conversion is the template
    fallback -- labelled, never silent."""
    queue.enqueue("evt-raw", "Cgroup", "a cat")
    host = FakeHost()

    report = worker.WorkerReport()
    action = await worker.tick(
        queue, host, files_dir=_files(tmp_path), report=report, prompt_mode="structured"
    )

    assert action == "completed"
    assert host.opens == 1 and host.waits == 1
    job = queue.recent()[0]
    assert job.state is JobState.DONE
    assert job.prompt["_built_by"].startswith("template")
    assert host.video.submitted == [f"job{job.id}"]


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
    assert host.touched_kinds == ["video"]


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


@pytest.mark.asyncio
async def test_an_understanding_request_completes_with_text_not_a_file(
    queue, tmp_path: Path
) -> None:
    """A third+ `media_kind` must route to the text-completion path -- the
    hazard a binary `if image else video` if/else would miss silently,
    misrouting an understanding job into `render_clip`/H3 conversion."""
    job = _parsed(
        queue, "", kind=MediaKind.IMAGE_UNDERSTAND, input_media_path="/incoming/x.jpg"
    )
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "completed"
    assert host.understand.submitted
    assert not host.video.submitted and not host.image.submitted
    done = queue.by_token(job.token)
    assert done.result_text == "a photo of a cat"
    assert done.output_path is None
    assert host.delivered == [(done.id, None)], (
        "no file was produced, so delivery must receive asset=None and rely "
        "on job.result_text"
    )


@pytest.mark.asyncio
async def test_an_understanding_failure_is_requeued_like_any_other(
    queue, tmp_path: Path
) -> None:
    job = _parsed(
        queue, "", kind=MediaKind.AUDIO_UNDERSTAND, input_media_path="/incoming/x.m4a"
    )
    host = FakeHost(
        understand=FakeUnderstandingProvider(fail_with=ProviderError("cold load timed out"))
    )

    action, report = await _tick(queue, host, tmp_path)

    assert action == "requeued"
    assert report.requeued == 1
    assert queue.by_token(job.token).state is JobState.PARSED


@pytest.mark.asyncio
async def test_a_chat_request_completes_with_text_not_a_file(queue, tmp_path: Path) -> None:
    """A CHAT job must not silently fall into `render_understanding` (which
    requires `input_media_path`) via a catchall dispatch branch -- it needs
    its own, and its completion must route through `complete_text` the same
    as understanding does."""
    job = _parsed(queue, "你好嗎", kind=MediaKind.CHAT)
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "completed"
    assert host.chat.submitted
    assert not host.video.submitted and not host.image.submitted and not host.understand.submitted
    done = queue.by_token(job.token)
    assert done.result_text == "hi there"
    assert done.output_path is None
    assert host.delivered == [(done.id, None)]


@pytest.mark.asyncio
async def test_a_chat_request_remembers_and_isolates_by_user(queue, tmp_path: Path) -> None:
    """`render_chat` must fetch/append this user's history, and never another
    user's -- the structural isolation `core.chat_spec` promises."""
    queue.append_chat_turn("U1", "Cgroup", "user", "earlier message")
    job = _parsed(queue, "second message", kind=MediaKind.CHAT)
    host = FakeHost()

    await _tick(queue, host, tmp_path)

    turns = queue.recent_chat_turns("U1")
    assert ("user", "earlier message") in turns
    assert ("user", "second message") in turns
    assert ("assistant", "hi there") in turns
    assert queue.recent_chat_turns("Uother") == []
    assert job.token  # the job itself rendered fine alongside the memory writes


@pytest.mark.asyncio
async def test_a_chat_failure_is_requeued_like_any_other(queue, tmp_path: Path) -> None:
    job = _parsed(queue, "你好", kind=MediaKind.CHAT)
    host = FakeHost(chat=FakeChatProvider(fail_with=ProviderError("cold load timed out")))

    action, report = await _tick(queue, host, tmp_path)

    assert action == "requeued"
    assert report.requeued == 1
    assert queue.by_token(job.token).state is JobState.PARSED


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

    host = FakeHost()
    report = await worker.serve(
        queue, host, files_dir=tmp_path / "files", max_ticks=1, sleep=_no_sleep
    )

    assert report.requeued == 1
    # Reclaimed, then rendered by this very run: there is no clock gate to
    # stop the loop picking it straight back up.
    assert queue.by_token(job.token).state is JobState.DONE


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
async def test_serve_sleeps_the_idle_interval_with_nothing_queued(queue, tmp_path: Path) -> None:
    """With nothing queued the loop does exactly one thing."""
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    host = FakeHost()
    await worker.serve(
        queue, host, files_dir=tmp_path / "files", max_ticks=2,
        closed_poll_s=60.0, idle_poll_s=10.0, sleep=record,
    )

    assert slept == [10.0, 10.0]
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


# ------------------------------------------------------------------ delivery

# `done` is not `delivered`. Push is billed per recipient, so a second send
# costs as much as the first and tells the user nothing new — and a delivery
# that fails silently is indistinguishable from a bot that is broken.


@pytest.mark.asyncio
async def test_a_finished_job_is_pushed_and_then_marked(queue, tmp_path: Path) -> None:
    job = _parsed(queue)
    host = FakeHost()

    await _tick(queue, host, tmp_path)

    assert [job_id for job_id, _ in host.delivered] == [job.id]
    assert queue.by_id(job.id).delivered_at is not None
    assert queue.undelivered() == []


@pytest.mark.asyncio
async def test_delivery_receives_the_finished_row_not_the_claimed_one(
    queue, tmp_path: Path
) -> None:
    """`complete` has just rewritten the row; the claimed copy still says
    `running` with no output path, which is exactly what delivery needs."""
    job = _parsed(queue)
    host = FakeHost()

    await _tick(queue, host, tmp_path)

    (_, asset) = host.delivered[0]
    assert asset is not None
    assert asset == queue.by_id(job.id).output_path


@pytest.mark.asyncio
async def test_a_terminal_failure_is_delivered_with_no_asset(queue, tmp_path: Path) -> None:
    """On success the user eventually sees something appear. On failure,
    silence is the only signal they get."""
    job = _parsed(queue)
    host = FakeHost(video=FakeProvider(fail_with=AIStudioError("prompt rejected")))

    await _tick(queue, host, tmp_path)

    assert host.delivered == [(job.id, None)]
    assert queue.by_id(job.id).delivered_at is not None


@pytest.mark.asyncio
async def test_a_requeue_is_not_announced(queue, tmp_path: Path) -> None:
    """The request is still alive and will be tried again. Announcing every
    transient hiccup would bill a push per attempt for nothing."""
    _parsed(queue)
    host = FakeHost(video=FakeProvider(fail_with=ProviderError("proxy hiccup")))

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "requeued"
    assert host.delivered == []


@pytest.mark.asyncio
async def test_a_delivery_failure_does_not_lose_the_render(queue, tmp_path: Path) -> None:
    """The media exists and was paid for in GPU-minutes. Turning a push problem
    into a render failure throws that away — and it stays `undelivered`, so it
    is visible rather than forgotten."""
    job = _parsed(queue)
    host = FakeHost(deliver_raises=RuntimeError("LINE is down"))

    action, report = await _tick(queue, host, tmp_path)

    assert action == "completed"
    assert queue.by_id(job.id).state is JobState.DONE
    assert queue.by_id(job.id).delivered_at is None
    assert [j.id for j in queue.undelivered()] == [job.id]
    assert report.undelivered == 1
    assert "UNDELIVERED=1" in report.summary()


@pytest.mark.asyncio
async def test_a_job_is_only_delivered_once(queue, tmp_path: Path) -> None:
    """The whole reason `delivered_at` exists: a restart must not re-push what
    it finds already finished."""
    _parsed(queue)
    host = FakeHost()

    await _tick(queue, host, tmp_path)
    await _tick(queue, host, tmp_path)

    assert len(host.delivered) == 1


def test_provision_feeds_hf_token_on_stdin_not_argv(monkeypatch, tmp_path) -> None:
    """The gated Tarsier2 repo needs HF_TOKEN on the pod; it must reach the
    setup script through stdin (read into the env) -- never in argv, which
    `ps` shows on both ends, and never written to the pod's disk."""
    from pydantic import SecretStr

    from ai_studio.runtime import session as sess

    script = tmp_path / "setup.sh"
    script.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
    inference = tmp_path / "srv.py"
    inference.write_text("print()\n", encoding="utf-8")
    seen: dict[str, object] = {}

    class _Proc:
        returncode = 0
        stdout = "started"
        stderr = ""

    def fake_ssh(argv, *, stdin, timeout_s):
        seen["argv"] = argv
        seen["stdin"] = stdin
        return _Proc()

    monkeypatch.setattr(sess, "_runpodctl", lambda *a: {"ip": "1.2.3.4", "port": "22"})
    monkeypatch.setattr(sess, "_ssh_deposit", lambda *a, **k: None)
    monkeypatch.setattr(sess, "_ssh", fake_ssh)
    monkeypatch.setattr(sess, "get_settings", lambda: type("S", (), {"hf_token": SecretStr("hf_secret")})())
    live = sess.Session.__new__(sess.Session)
    live.__dict__.update(pod_id="p1", vram_gb=24)
    sess.provision(live, script=script, inference_script=inference)

    assert str(seen["stdin"]).startswith("hf_secret\n#!/bin/bash")
    assert not any("hf_secret" in a for a in seen["argv"])
    assert "IFS= read -r HF_TOKEN; export HF_TOKEN;" in " ".join(seen["argv"])


@pytest.mark.asyncio
async def test_a_silent_quota_failure_leaves_the_job_for_the_pull_trigger(queue, tmp_path: Path) -> None:
    """Push and its text fallback both died on quota: nothing reached the
    group. The row must stay undelivered so「讓我看看」can hand it over, and
    the queue must remember the month is out of push quota."""
    job = _parsed(queue)
    host = FakeHost()

    async def silent(job, asset):
        return "quota-exhausted-and-silent"

    host.deliver = silent  # type: ignore[method-assign]
    action, report = await _tick(queue, host, tmp_path)

    assert action == "completed"
    assert queue.by_id(job.id).delivered_at is None
    assert queue.push_quota_exhausted()
    assert report.undelivered == 1


def test_push_quota_marker_expires_with_the_calendar_month(queue) -> None:
    import calendar
    from datetime import datetime, timezone

    queue.note_push_quota_exhausted(now=datetime(2026, 8, 27, tzinfo=timezone.utc).timestamp())
    assert queue.push_quota_exhausted(now=datetime(2026, 8, 31, 23, tzinfo=timezone.utc).timestamp())
    assert not queue.push_quota_exhausted(now=datetime(2026, 9, 1, 1, tzinfo=timezone.utc).timestamp())
    assert calendar.monthrange(2026, 8)[1] == 31


# ------------------------------------------------ the prepare phase (rewrites)


class _Evicting(FakeProvider):
    """A generation provider that counts evictions, so a test can assert how
    many model swaps a batch caused."""

    def __init__(self) -> None:
        super().__init__()
        self.evictions = 0

    async def evict(self) -> None:
        self.evictions += 1


def _scripted(n: int) -> Any:
    from ai_studio.llm.endpoint import ScriptedLlmClient

    reply = (
        '{"shots":[{"style":"Live-action","description":"An orange cat walks slowly. No text.",'
        '"camera":{"motion":"static_shot"}}],"overall_soundscape":"Rain. No dialogue.",'
        '"non_diegetic_music":"N/A"}'
    )
    return ScriptedLlmClient(*([reply] * n))


@pytest.mark.asyncio
async def test_prepare_rewrites_every_queued_job_in_one_llm_residency(queue, tmp_path: Path) -> None:
    """Three clips queued together pay ONE eviction of the checkpoint (one
    gpt-oss load), and all three are parsed before any render starts."""
    for i in range(3):
        queue.enqueue(f"evt-{i}", "Cgroup", f"cat {i}", user_id="U1")
    video = _Evicting()
    host = FakeHost(video=video, llm=_scripted(3))
    state = worker.WorkerState()

    report = worker.WorkerReport()
    action = await worker.tick(
        queue, host, files_dir=_files(tmp_path), report=report, state=state,
        prompt_mode="structured",
    )

    assert action == "completed"
    assert video.evictions == 1, "one make_room_for(CHAT) for the whole batch"
    assert host.llm.calls and len(host.llm.calls) == 3
    states = {j.state for j in queue.recent(3)}
    assert states == {JobState.DONE, JobState.PARSED}  # one rendered, two claimable
    assert all((j.prompt or {}).get("_built_by") == "llm" for j in queue.recent(3))


@pytest.mark.asyncio
async def test_rewrites_are_deferred_while_the_resident_checkpoint_has_work(
    queue, tmp_path: Path
) -> None:
    """H3 is loaded and has a parsed clip waiting: a newly queued request must
    not evict it for a single rewrite. The parsed clip renders first; the
    rewrite waits for the next tick."""
    _parsed(queue, "already parsed")
    queue.enqueue("evt-late", "Cgroup", "late arrival", user_id="U1")
    video = _Evicting()
    host = FakeHost(video=video, llm=_scripted(1))
    state = worker.WorkerState(resident=MediaKind.VIDEO)

    report = worker.WorkerReport()
    action = await worker.tick(
        queue, host, files_dir=_files(tmp_path), report=report, state=state,
        prompt_mode="structured",
    )

    assert action == "completed"
    assert video.evictions == 0
    assert host.llm.calls == []
    late = next(j for j in queue.recent(2) if j.text == "late arrival")
    assert late.state is JobState.QUEUED

    action = await worker.tick(
        queue, host, files_dir=_files(tmp_path), report=report, state=state,
        prompt_mode="structured",
    )
    assert action == "completed" and len(host.llm.calls) == 1


@pytest.mark.asyncio
async def test_chat_and_bare_describe_jobs_never_call_the_llm(queue, tmp_path: Path) -> None:
    queue.enqueue("evt-chat", "Cgroup", "你好", user_id="U1", media_kind=MediaKind.CHAT)
    queue.enqueue(
        "evt-img", "Cgroup", "", user_id="U1", media_kind=MediaKind.IMAGE_UNDERSTAND,
        input_media_path=str(tmp_path / "p.jpg"),
    )
    video = _Evicting()
    host = FakeHost(video=video, llm=_scripted(0))
    state = worker.WorkerState()

    prepared = await worker.prepare(
        queue, host, host.session, host.providers_for(host.session), state,
        prompt_mode="structured",
    )

    assert prepared == 2
    assert host.llm.calls == []
    assert video.evictions == 0, "prepare did not touch the card for these"
    assert state.resident is None
    kinds = {j.media_kind: j for j in queue.recent(2)}
    assert kinds[MediaKind.CHAT].prompt["_system"].startswith("# Instructions")
    assert kinds[MediaKind.IMAGE_UNDERSTAND].prompt["_built_by"] == "understanding-default"
    assert kinds[MediaKind.IMAGE_UNDERSTAND].prompt["_question"] is None


@pytest.mark.asyncio
async def test_the_question_and_system_prompt_reach_the_providers(queue, tmp_path: Path) -> None:
    from ai_studio.llm.endpoint import ScriptedLlmClient

    queue.enqueue(
        "evt-q", "Cgroup", "這是誰", user_id="U1", media_kind=MediaKind.AUDIO_UNDERSTAND,
        input_media_path=str(tmp_path / "a.m4a"),
    )
    queue.enqueue("evt-c", "Cgroup", "嗨", user_id="U1", media_kind=MediaKind.CHAT)
    host = FakeHost(llm=ScriptedLlmClient('{"question": "請只用繁體中文說出說話者是誰"}'))
    state = worker.WorkerState()
    report = worker.WorkerReport()
    for _ in range(2):
        await worker.tick(queue, host, files_dir=_files(tmp_path), report=report, state=state,
                          prompt_mode="structured")

    assert host.understand.prompts == ["請只用繁體中文說出說話者是誰"]
    assert host.chat.extras and host.chat.extras[0]["system"].startswith("# Instructions")


@pytest.mark.asyncio
async def test_affinity_prefers_the_resident_kind_but_is_bounded(queue, tmp_path: Path) -> None:
    """With Flux resident and an image parsed behind an older clip, the image
    goes first (no swap); after MAX_AFFINITY_RUN the clip is not starved."""
    clip = _parsed(queue, "old clip", kind=MediaKind.VIDEO)
    _parsed(queue, "new image", kind=MediaKind.IMAGE)
    host = FakeHost()
    state = worker.WorkerState(resident=MediaKind.IMAGE)
    report = worker.WorkerReport()

    await worker.tick(queue, host, files_dir=_files(tmp_path), report=report, state=state)
    assert host.image.submitted and not host.video.submitted
    assert queue.by_id(clip.id).state is JobState.PARSED

    state.affinity_run = worker.MAX_AFFINITY_RUN
    _parsed(queue, "another image", kind=MediaKind.IMAGE)
    await worker.tick(queue, host, files_dir=_files(tmp_path), report=report, state=state)
    assert host.video.submitted, "the affinity run is bounded; FIFO resumes"


# --------------------------------------------------------------------- /短劇


def test_a_drama_always_needs_the_screenwriter(queue) -> None:
    """Raw mode is for one-line clips. A drama has nothing to be raw from."""
    from ai_studio.pipeline.convert_worker import needs_llm

    job, _ = queue.enqueue("evt-drama-needs", "Cgroup", "一個故事", user_id="U1", media_kind=MediaKind.DRAMA)
    assert needs_llm(job, "raw") is True
    assert needs_llm(job, "structured") is True


@pytest.mark.asyncio
async def test_a_drama_whose_screenplay_fails_is_failed_and_the_group_is_told(queue, tmp_path: Path) -> None:
    """No template fallback: with a screenwriter that keeps returning garbage
    the job is FAILED at conversion and delivered as such -- not left queued
    forever, and not rendered as six clips of nothing."""
    from ai_studio.llm.endpoint import ScriptedLlmClient

    job, _ = queue.enqueue("evt-drama-fail", "Cgroup", "一個故事", user_id="U1", media_kind=MediaKind.DRAMA)
    host = FakeHost(llm=ScriptedLlmClient("not json", "still not json"))

    action, _report = await _tick(queue, host, tmp_path)

    failed = queue.by_id(job.id)
    assert failed.state is JobState.FAILED and "編劇失敗" in (failed.error or "")
    assert host.delivered == [(job.id, None)], "the failure reached the group"
    assert action in ("prepared", "raced")


@pytest.mark.asyncio
async def test_a_drama_with_no_screenwriter_on_the_pod_is_failed_not_stuck(queue, tmp_path: Path) -> None:
    job, _ = queue.enqueue("evt-drama-nollm", "Cgroup", "一個故事", user_id="U1", media_kind=MediaKind.DRAMA)
    host = FakeHost(llm=None)

    await _tick(queue, host, tmp_path)

    assert queue.by_id(job.id).state is JobState.FAILED
    assert host.delivered == [(job.id, None)]


@pytest.mark.asyncio
async def test_a_parsed_drama_is_dispatched_to_render_drama(queue, tmp_path: Path, monkeypatch) -> None:
    """The dispatch is explicit. Before this branch existed an unknown kind
    fell through to `render_understanding` -- silently, with the wrong
    provider."""
    from ai_studio.pipeline import worker as worker_mod

    seen: dict[str, Any] = {}

    async def fake_render_drama(job: Any, providers: Any, **kw: Any) -> Path:
        seen["job"] = job.id
        seen["providers"] = set(providers)
        seen["kw"] = kw
        kw["on_activity"]()
        out = kw["files_dir"] / f"{job.token}.mp4"
        out.write_bytes(b"mp4")
        return out

    monkeypatch.setattr(worker_mod, "render_drama", fake_render_drama)
    job, _ = queue.enqueue("evt-drama-run", "Cgroup", "一個故事", user_id="U1", media_kind=MediaKind.DRAMA)
    queue.set_parsed(job.id, {"_built_by": "llm", "_rendered": "t", "screenplay": {"stub": True}, "shots": []})
    host = FakeHost()

    action, _ = await _tick(queue, host, tmp_path)

    assert action == "completed"
    assert seen["job"] == job.id
    assert {MediaKind.IMAGE, MediaKind.VIDEO} <= seen["providers"]
    assert seen["kw"]["runs_dir"] is not None
    assert host.touched_kinds.count("drama") >= 2, "per-artifact touch plus the completion touch"
    done = queue.by_id(job.id)
    assert done.state is JobState.DONE and done.output_path.endswith(".mp4")
    assert host.delivered[-1][0] == job.id


@pytest.mark.asyncio
async def test_a_drama_that_pauses_at_the_lease_boundary_never_burns_an_attempt(
    queue, tmp_path: Path, monkeypatch
) -> None:
    """The cost-guard finding on PR #31: a drama's *designed* resume raised
    a plain ProviderError, so three honest short windows counted as three
    provider failures and the job was failed for good -- with its paid
    stills and clips orphaned on disk and a daily drama slot burned. A
    DramaResume must requeue with the attempt handed back, every time."""
    from ai_studio.core.errors import DramaResume
    from ai_studio.pipeline import worker as worker_mod
    from ai_studio.pipeline.drain import MAX_ATTEMPTS

    async def paused(job: Any, providers: Any, **kw: Any) -> Path:
        raise DramaResume("lease ends in 100s, under the 360s a drama video needs")

    monkeypatch.setattr(worker_mod, "render_drama", paused)
    job, _ = queue.enqueue("evt-drama-pause", "Cgroup", "一個故事", user_id="U1", media_kind=MediaKind.DRAMA)
    queue.set_parsed(job.id, {"_built_by": "llm", "_rendered": "t", "screenplay": {"stub": True}, "shots": []})
    host = FakeHost()

    for _ in range(2 * MAX_ATTEMPTS):
        action, report = await _tick(queue, host, tmp_path)
        assert action == "resumed-later"
        assert report.requeued == 1 and report.failed == 0

    row = queue.by_id(job.id)
    assert row.state is JobState.PARSED, "still claimable next window"
    assert row.attempts == 0, "a designed stop hands its attempt back"
    assert host.delivered == [], "the group is not told about a pause"


@pytest.mark.asyncio
async def test_a_dramas_real_provider_failure_still_counts_toward_max_attempts(
    queue, tmp_path: Path, monkeypatch
) -> None:
    from ai_studio.pipeline import worker as worker_mod
    from ai_studio.pipeline.drain import MAX_ATTEMPTS

    async def broken(job: Any, providers: Any, **kw: Any) -> Path:
        raise ProviderError("ComfyUI returned 500")

    monkeypatch.setattr(worker_mod, "render_drama", broken)
    job, _ = queue.enqueue("evt-drama-broken", "Cgroup", "一個故事", user_id="U1", media_kind=MediaKind.DRAMA)
    queue.set_parsed(job.id, {"_built_by": "llm", "_rendered": "t", "screenplay": {"stub": True}, "shots": []})
    host = FakeHost()

    actions = [(await _tick(queue, host, tmp_path))[0] for _ in range(MAX_ATTEMPTS)]
    assert actions == ["requeued"] * (MAX_ATTEMPTS - 1) + ["failed"]
    assert queue.by_id(job.id).state is JobState.FAILED


# ------------------------------------------------------------ the trace


@pytest.mark.asyncio
async def test_a_completed_job_leaves_a_traceable_record(queue, tmp_path: Path, caplog) -> None:
    """One `bind` around the render is what lets `grep token=` follow a
    request: every line inside carries job_id, token and kind as record
    attributes -- claimed, submitted, fetched, done, delivered -- without any
    of those callers being told the job (📏 the first live drama left no
    worker line at all, 2026-08-28)."""
    import logging

    from ai_studio.core.observability import configure_logging

    # What every composition root does first; without it the context filter
    # is not on the logger path and records carry no job/token (the exact
    # production gap this work closes).
    configure_logging(service="worker", log_dir=None, level="INFO")
    job = _parsed(queue)
    host = FakeHost()
    with caplog.at_level(logging.INFO, logger="ai_studio"):
        action, _ = await _tick(queue, host, tmp_path)
    assert action == "completed"

    traced = [r for r in caplog.records if getattr(r, "token", None) == job.token]
    msgs = [r.getMessage() for r in traced]
    assert msgs, "no line carried the job's token"
    assert all(r.job_id == job.id and r.kind == "video" for r in traced)
    assert "claimed" in msgs
    assert any(m.startswith("submitted clip") for m in msgs)
    assert any(m.startswith("fetched clip") for m in msgs)
    done = next(r for r in traced if r.getMessage().startswith(f"job {job.id} done in"))
    assert done.outcome == "completed" and done.seconds >= 0 and done.stage == "render"
    assert "delivered" in msgs
