"""The always-on worker: do the queued work as soon as there is any.

`drain.drain_window` answers "the window is open, empty the queue before the
bell". This answers "there is a request, and the shop is open — do it now".
Same rendering, different owner of the loop, and the difference is the whole of
"instant": the pod is created by the first request that
arrives while the shop is open, not by a timer that fires whether or not anyone
asked for anything.

With nothing queued the loop does exactly one thing: sleep. It does not
drain, it does not open a pod, it does not fail. Requests keep arriving and keep
waiting — refusing them would mean whatever someone thought of at midnight is
simply lost.

**Everything to do with pods and clocks arrives by injection.** `pipeline` sits
below `runtime` in the layer contract, so this module cannot import
`runtime.session` or `runtime.hours` however much it would like to. `WindowHost`
is the seam; `cli.main`, the composition root, is the only place that knows both
halves. That is the same shape `prompts.convert` uses for its LLM client, and it
buys the same thing: this loop is testable with no pod, no clock, and no money.

Concurrency is one. That is not an oversight and not a placeholder — one pod,
one ComfyUI, one model resident in VRAM. A second concurrent render on the same
card is slower than the two runs in sequence, not faster.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ai_studio.config.settings import get_settings
from ai_studio.core.enums import MediaKind
from ai_studio.core.errors import AIStudioError, DramaResume, ProviderError
from ai_studio.core.observability import bind
from ai_studio.pipeline.convert_worker import convert_job, needs_llm
from ai_studio.pipeline.drain import (
    MAX_ATTEMPTS,
    MAX_CONSECUTIVE_FAILURES,
    may_claim,
    render_chat,
    render_clip,
    render_image,
    render_understanding,
)
from ai_studio.pipeline.drama import render_drama
from ai_studio.pipeline.queue import Job, JobQueue, JobState
from ai_studio.pipeline.residency import make_room_for
from ai_studio.storage.index import append_delivery

_log = logging.getLogger("ai_studio.worker")

IDLE_POLL_S = 10.0
"""How often the queue is checked while the shop is open.

Ten seconds, not one: the wait a user actually notices is the two to six
minutes of generation, and a tighter loop only spends VPS CPU on a 1 GB box.
"""

CLOSED_POLL_S = 60.0
"""How long to sleep after a refusal (budget, daily cap, no stock, a pod that
never answered). Nothing useful can happen sooner, and a tight loop here
would only ask RunPod the same question sixty times a minute."""


class WindowHost(Protocol):
    """Everything the loop needs from the layers above it.

    Injected rather than imported because `pipeline` cannot import `runtime`.
    Implemented for real in `cli.main`, and by a fake in the tests.
    """

    def now(self) -> datetime: ...

    def claim_deadline(self, now: datetime | None = None) -> datetime:
        """The bell new work must finish before — the end of the pod's lease."""

    def ensure_pod(self, queue: JobQueue) -> Any:
        """A live pod, opening one if budget allows. May raise."""

    def wait_ready(self, session: Any) -> float:
        """Provision the pod if it is fresh, then block until it can render."""

    def providers_for(self, session: Any) -> dict[MediaKind, Any]:
        """The clip and image backends bound to that pod."""

    def llm_for(self, session: Any) -> Any:
        """The prompt rewriter bound to that pod (gpt-oss-20b through the
        inference server, `pipeline.pod_llm.PodLlmClient`), or None to
        convert with the template fallback and no LLM at all."""

    def touch_activity(self, media_kind: str) -> None:
        """Reset the idle reaper's timer, recording what was just rendered."""

    async def deliver(self, job: Job, asset: Path | None) -> str:
        """Tell the group. `asset` is None when the job failed.

        Injected for the same reason as the rest: `bots` is a leaf that no
        library layer may import, so the loop cannot reach the push client and
        the composition root hands it down instead.
        """


MAX_AFFINITY_RUN = 6
"""How many jobs of the resident kind may be claimed in a row before the
worker falls back to strict FIFO. Model affinity is what keeps a loaded
checkpoint busy instead of swapping it out (📏 45-90 s per swap on the RTX
4090), but unbounded it would let a stream of images starve a queued clip."""


@dataclass
class WorkerState:
    """What the loop believes about the card, carried between ticks.

    `resident` is the kind whose model the worker last put on the GPU --
    CHAT after a rewrite batch, the rendered kind after a render. It is a
    belief, not a probe: the inference server and ComfyUI do the actual
    eviction (`make_room_for`), this only steers which job to claim next.
    """

    resident: MediaKind | None = None
    affinity_run: int = 0


@dataclass
class WorkerReport:
    """What the loop has done since it started. Read by the CLI on exit."""

    ticks: int = 0
    completed: int = 0
    failed: int = 0
    requeued: int = 0
    delivered: int = 0
    undelivered: int = 0
    seconds: list[float] = field(default_factory=list)
    last_action: str = "not started"
    stopped_early: str | None = None

    def summary(self) -> str:
        parts = [
            f"ticks={self.ticks}",
            f"completed={self.completed}",
            f"failed={self.failed}",
            f"requeued={self.requeued}",
            f"delivered={self.delivered}",
        ]
        if self.undelivered:
            # Louder than the rest: the media exists and nobody has been told.
            parts.append(f"UNDELIVERED={self.undelivered}")
        if self.seconds:
            parts.append("job_seconds=" + ", ".join(f"{s:.0f}" for s in self.seconds))
        if self.stopped_early:
            parts.append(f"stopped_early={self.stopped_early!r}")
        return " ".join(parts)


def claimable(queue: JobQueue) -> list[Job]:
    """Requests that are ready for a GPU — `parsed`, not merely `queued`.

    The distinction is what a pod costs. A `queued` job has not been through
    the LLM yet and may never become a valid prompt; opening a pod for one
    would be paying for a conversion that has not happened. Only `parsed` work
    is worth a machine.
    """
    return [job for job in queue.pending() if job.state is JobState.PARSED]


async def tick(
    queue: JobQueue,
    host: WindowHost,
    *,
    files_dir: Path,
    report: WorkerReport,
    poll_interval_s: float = 15.0,
    state: WorkerState | None = None,
    prompt_mode: str | None = None,
) -> str:
    """One pass of the loop. Returns what it did, for the log and the tests.

    Ordered cheapest-first so the expensive calls are never reached on the
    ticks that do not need them: closed, then empty, then — and only then — a
    pod.
    """
    report.ticks += 1
    now = host.now()
    state = state if state is not None else WorkerState()

    if not queue.pending():
        report.last_action = "idle"
        return "idle"

    deadline = host.claim_deadline(now)
    if not may_claim(deadline):
        # The last minutes of the day belong to finishing, not starting. A
        # render begun now is billed and then thrown away by the lease.
        report.last_action = "too-late"
        return "too-late"

    # Conversion needs the pod too now (the rewriter is gpt-oss-20b on it), so
    # a queued-but-unconverted request opens the pod the same as a parsed one.
    session = host.ensure_pod(queue)
    host.wait_ready(session)
    providers = host.providers_for(session)

    prepared = await prepare(
        queue, host, session, providers, state, prompt_mode=prompt_mode, report=report
    )

    kind = next_kind(queue, state)
    tier_label = getattr(session, "tier_label", None)
    usd_per_hr = getattr(session, "cost_per_hr", None)
    job = queue.claim_next(gpu_tier=tier_label, usd_per_hr=usd_per_hr, media_kind=kind)
    if job is None and kind is not None:
        job = queue.claim_next(gpu_tier=tier_label, usd_per_hr=usd_per_hr)
    if job is None:
        # Nothing claimable: either everything is still queued behind a
        # deferred rewrite (see prepare) or someone else took it between the
        # check and the claim. Both harmless -- the atomic claim is what makes
        # the race a non-event rather than a double charge.
        report.last_action = "prepared" if prepared else "raced"
        return report.last_action

    if job.media_kind is state.resident:
        state.affinity_run += 1
    else:
        state.affinity_run = 0
    # One bind covers every line inside the render: drain, drama stages, the
    # model swap, the pod LLM, the push -- all run in this task.
    with bind(job_id=job.id, token=job.token, kind=job.media_kind.value):
        _log.info(
            "claimed", extra={"attempts": job.attempts, "gpu_tier": job.gpu_tier, "stage": "claim"},
        )
        outcome = await _run_one(
            queue, host, job, providers, files_dir=files_dir,
            deadline=deadline, report=report, poll_interval_s=poll_interval_s,
        )
    state.resident = job.media_kind
    return outcome


async def prepare(
    queue: JobQueue,
    host: WindowHost,
    session: Any,
    providers: dict[MediaKind, Any],
    state: WorkerState,
    *,
    prompt_mode: str | None = None,
    report: WorkerReport | None = None,
) -> int:
    """Convert queued requests to claimable ones on the pod. Returns how many.

    Two groups. Jobs that need no rewriter (chat; a bare describe trigger;
    raw-mode generation) are converted at once -- nothing to load. Jobs that
    need gpt-oss (structured-mode generation; a describe trigger with a
    question) are converted **all together while it is resident**: one
    `make_room_for(CHAT)` evicts whatever ComfyUI held, then every pending
    rewrite runs against the loaded model. N clips pay one gpt-oss load and
    one H3 load, not 2N (📏 57-90 s per load).

    The costly group is *deferred* while a generation checkpoint is resident
    and still has claimable work of its own kind: finishing that batch first
    avoids paying H3's reload twice. A late arrival waits one batch, then gets
    its own rewrite. Chat is never in the costly group, so a chat job cannot
    force a swap on its own.

    A rewrite failure never drops a job: `prompts.convert` / `prompts.flux` /
    `prompts.understanding` all fall back to a labelled template payload
    (`_built_by` says so), and a client that cannot reach the pod at all
    surfaces as `template (llm failed: ...)` the same way.
    """
    queued = [j for j in queue.unparsed(limit=50) if j.state is JobState.QUEUED]
    if not queued:
        return 0
    mode = prompt_mode or get_settings().prompt_mode
    cheap = [j for j in queued if not needs_llm(j, mode)]
    costly = [j for j in queued if needs_llm(j, mode)]

    done = 0
    for job in cheap:
        await convert_job(queue, job.id, None, prompt_mode=mode)
        done += 1

    if costly and not should_defer_llm(queue, state):
        llm = host.llm_for(session)
        if llm is not None:
            await make_room_for(MediaKind.CHAT, providers)
            state.resident = MediaKind.CHAT
            state.affinity_run = 0
        for job in costly:
            with bind(job_id=job.id, token=job.token, kind=job.media_kind.value):
                how = await convert_job(queue, job.id, llm, prompt_mode=mode)
                _log.info(
                    "job %d prepared: %s", job.id, how,
                    extra={"built_by": how, "seconds": getattr(llm, "last_total_s", None),
                           "stage": "prepare"},
                )
            done += 1
            if how.startswith("failed"):
                # A drama whose screenplay could not be written is FAILED
                # already (convert_worker); the group must hear it now, since
                # nothing else will ever claim that row.
                await _deliver(queue, host, job.id, None, report or WorkerReport())
    elif costly:
        _log.info(
            "%d rewrite(s) deferred: %s still has claimable work",
            len(costly), state.resident.value if state.resident else "?",
            extra={"deferred": len(costly), "resident": state.resident.value if state.resident else None,
                   "stage": "prepare"},
        )
    return done


def should_defer_llm(queue: JobQueue, state: WorkerState) -> bool:
    """Hold the rewrite batch while the resident generation checkpoint still
    has work, so it is not evicted and reloaded around a single rewrite."""
    if state.resident is None or not state.resident.is_generation:
        return False
    return any(
        j.state is JobState.PARSED and j.media_kind is state.resident for j in queue.pending()
    )


def next_kind(queue: JobQueue, state: WorkerState) -> MediaKind | None:
    """The kind to claim next: the resident one while it has claimable work
    and the affinity run is under `MAX_AFFINITY_RUN`; else None for FIFO."""
    if state.resident is None or state.affinity_run >= MAX_AFFINITY_RUN:
        return None
    if any(j.state is JobState.PARSED and j.media_kind is state.resident for j in queue.pending()):
        return state.resident
    return None


async def _run_one(
    queue: JobQueue,
    host: WindowHost,
    job: Job,
    providers: dict[MediaKind, Any],
    *,
    files_dir: Path,
    deadline: datetime,
    report: WorkerReport,
    poll_interval_s: float,
) -> str:
    """Render one claimed job. Failure handling mirrors `drain_window` exactly.

    Deliberately the same rules rather than new ones: a machine failure keeps
    the request (`requeue`) until it has used its attempts, and a failure that
    would repeat identically is terminal. Two different answers to "the pod
    died mid-render" is how a request gets silently dropped.
    """
    # A drama drives the IMAGE and VIDEO providers itself; there is no
    # `providers[DRAMA]` entry and nothing to ask capabilities of here.
    provider: Any = providers.get(job.media_kind)
    if provider is None and job.media_kind is not MediaKind.DRAMA:
        raise AIStudioError(f"this pod serves no provider for {job.media_kind.value}")
    caps = provider.capabilities() if provider is not None else None
    started = time.monotonic()

    try:
        await make_room_for(job.media_kind, providers)
        if job.media_kind is MediaKind.IMAGE:
            result: Any = await render_image(
                job, provider, caps, files_dir, deadline, poll_interval_s
            )
        elif job.media_kind is MediaKind.VIDEO:
            result = await render_clip(
                job, provider, caps, files_dir, deadline, poll_interval_s
            )
        elif job.media_kind is MediaKind.CHAT:
            result = await render_chat(job, provider, queue, deadline, poll_interval_s)
        elif job.media_kind is MediaKind.DRAMA:
            result = await render_drama(
                job, providers, files_dir=files_dir, runs_dir=get_settings().runs_dir,
                deadline=deadline, poll_interval_s=poll_interval_s,
                # Per artifact, not per job: a drama is 15-30 minutes and the
                # reaper's grace is 10. Every fetched still or clip is activity.
                on_activity=lambda: host.touch_activity(MediaKind.DRAMA.value),
            )
        elif job.media_kind.is_understanding:
            result = await render_understanding(job, provider, deadline, poll_interval_s)
        else:
            raise AIStudioError(f"no renderer for media kind {job.media_kind!r}")
    except DramaResume as exc:
        # Not a failure: the drama stopped itself at the lease boundary with
        # its state file intact. Requeue and hand the attempt back -- three
        # honest short windows must not read as three provider failures and
        # orphan the clips already paid for.
        queue.fail(job.id, f"resume: {exc}", requeue=True, uncounted=True)
        report.requeued += 1
        _log.info("job %d paused for the next window: %s", job.id, exc,
                  extra={"outcome": "resumed-later", "reason": str(exc)[:200]})
        report.last_action = "resumed-later"
        return "resumed-later"
    except ProviderError as exc:
        if job.attempts >= MAX_ATTEMPTS:
            queue.fail(job.id, f"provider failed {job.attempts}x: {exc}")
            report.failed += 1
            _log.warning("job %d failed for good: %s", job.id, exc,
                         extra={"outcome": "failed", "attempts": job.attempts, "reason": str(exc)[:200]})
            await _deliver(queue, host, job.id, None, report)
            report.last_action = "failed"
            return "failed"
        # No delivery on a requeue: the request is still alive and will be
        # tried again. Announcing every transient hiccup would bill a push per
        # attempt for something the user does not need to know about.
        queue.fail(job.id, f"provider: {exc}", requeue=True)
        report.requeued += 1
        _log.warning("job %d requeued (attempt %d): %s", job.id, job.attempts, exc,
                     extra={"outcome": "requeued", "attempts": job.attempts, "reason": str(exc)[:200]})
        report.last_action = "requeued"
        return "requeued"
    except AIStudioError as exc:
        queue.fail(job.id, str(exc))
        report.failed += 1
        _log.warning("job %d failed terminally: %s", job.id, exc,
                     extra={"outcome": "failed", "reason": str(exc)[:200]})
        await _deliver(queue, host, job.id, None, report)
        report.last_action = "failed"
        return "failed"

    # Understanding and chat jobs produce text, not a file: `asset` stays
    # None so `_deliver` pushes `job.result_text` instead of a media message.
    asset: Path | None = None
    if job.media_kind.is_understanding or job.media_kind is MediaKind.CHAT:
        queue.complete_text(job.id, result)
    else:
        asset = result
        queue.complete(job.id, str(asset))
        # The file's name is a random token; this line is the only map from
        # it back to the request. Best-effort, never blocks the delivery.
        append_delivery(files_dir, token=job.token, job_id=job.id, kind=job.media_kind.value, path=asset)
    report.completed += 1
    report.seconds.append(time.monotonic() - started)
    # Without this a long render looks like idleness and the reaper closes the
    # window out from under the next job.
    host.touch_activity(job.media_kind.value)
    finished = queue.by_id(job.id)
    _log.info(
        "job %d done in %.0fs -> %s", job.id, report.seconds[-1], result,
        extra={"outcome": "completed", "seconds": round(report.seconds[-1], 1),
               "cost_usd": finished.cost_usd if finished is not None else None,
               "stage": "render"},
    )
    await _deliver(queue, host, job.id, asset, report)
    report.last_action = "completed"
    return "completed"


async def _deliver(
    queue: JobQueue,
    host: WindowHost,
    job_id: int,
    asset: Path | None,
    report: WorkerReport,
) -> str:
    """Push the result to the group, then mark it delivered.

    **The order is complete -> push -> mark, and it is not arbitrary.** A push
    that succeeds and a mark that then fails sends one extra message next
    restart; a mark that lands before a push that fails loses the delivery
    entirely and the user waits forever. One duplicate beats one silence.

    The job is re-read rather than reused: `complete`/`fail` have just rewritten
    its row, and the stale in-memory copy still says `running` with no output
    path — which is exactly what the delivery needs.

    A delivery failure never fails the *job*. The media exists, the queue says
    so, and the status page will show it. Turning a push problem into a render
    failure would throw away work that was paid for in GPU-minutes.
    """
    job = queue.by_id(job_id)
    if job is None:  # pragma: no cover - the row was just written
        return "vanished"
    try:
        outcome = await host.deliver(job, asset)
    except Exception as exc:  # delivery must never lose a finished render
        _log.error("job %d could not be delivered: %s", job.id, exc,
                   extra={"outcome": "delivery-failed", "stage": "deliver"})
        report.undelivered += 1
        return "failed"

    if outcome.startswith("quota-exhausted"):
        # A money condition that lasts the rest of the month; the webhook
        # reads this to tell the next requester to pull with /讓我看看.
        queue.note_push_quota_exhausted()
    if outcome.endswith("-and-silent"):
        # Nothing reached the group. Leave the row undelivered so the pull
        # trigger (/讓我看看, a free reply) can hand it over later -- marking
        # it delivered here would make a finished, paid-for result vanish.
        _log.warning("job %d finished but the group was told nothing (%s); held for pull",
                     job.id, outcome,
                     extra={"outcome": outcome, "quota_exhausted": True, "stage": "deliver"})
        report.undelivered += 1
        return outcome

    queue.mark_delivered(job.id)
    report.delivered += 1
    _log.info("delivered", extra={"outcome": outcome, "stage": "deliver"})
    return outcome


async def serve(
    queue: JobQueue,
    host: WindowHost,
    *,
    files_dir: Path,
    idle_poll_s: float = IDLE_POLL_S,
    closed_poll_s: float = CLOSED_POLL_S,
    poll_interval_s: float = 15.0,
    max_ticks: int | None = None,
    sleep: Any = asyncio.sleep,
) -> WorkerReport:
    """Run the loop. `max_ticks` bounds it for tests; production passes None.

    Anything still `running` at startup belongs to a worker that died mid-job —
    reclaim it first, or it sits there forever and the user waits forever.
    """
    report = WorkerReport()
    files_dir = Path(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)
    report.requeued += queue.release_running("worker restarted")

    consecutive_failures = 0
    state = WorkerState()

    while max_ticks is None or report.ticks < max_ticks:
        try:
            action = await tick(
                queue, host, files_dir=files_dir, report=report,
                poll_interval_s=poll_interval_s, state=state,
            )
        except AIStudioError as exc:
            # Hours, budget, the daily open cap, a pod that never answered.
            # None of these is a reason to exit: the shop opens again tomorrow,
            # and a worker that dies on a refusal is a worker someone has to
            # remember to restart.
            _log.warning("tick refused: %s", exc)
            report.last_action = "refused"
            await sleep(closed_poll_s)
            continue

        if action == "failed":
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                # Three in a row is a broken pod, not bad luck. Back off to the
                # slow poll rather than grinding the whole queue through it.
                report.stopped_early = "three consecutive failures; the pod looks broken"
                _log.error("%s", report.stopped_early)
                consecutive_failures = 0
                await sleep(closed_poll_s)
                continue
        elif action == "completed":
            consecutive_failures = 0

        if action == "completed":
            # More work may be waiting and the pod is hot. Do not sleep on it.
            continue
        await sleep(closed_poll_s if action == "closed" else idle_poll_s)

    return report
