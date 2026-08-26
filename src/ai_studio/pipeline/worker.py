"""The always-on worker: do the queued work as soon as there is any.

`drain.drain_window` answers "the window is open, empty the queue before the
bell". This answers "there is a request, and the shop is open — do it now".
Same rendering, different owner of the loop, and the difference is the whole of
"instant inside business hours": the pod is created by the first request that
arrives while the shop is open, not by a timer that fires whether or not anyone
asked for anything.

Outside business hours the loop does exactly one thing: sleep. It does not
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

from ai_studio.core.enums import MediaKind
from ai_studio.core.errors import AIStudioError, ProviderError
from ai_studio.pipeline.drain import (
    MAX_ATTEMPTS,
    MAX_CONSECUTIVE_FAILURES,
    may_claim,
    render_clip,
    render_image,
)
from ai_studio.pipeline.queue import Job, JobQueue, JobState

_log = logging.getLogger("ai_studio.worker")

IDLE_POLL_S = 10.0
"""How often the queue is checked while the shop is open.

Ten seconds, not one: the wait a user actually notices is the two to six
minutes of generation, and a tighter loop only spends VPS CPU on a 1 GB box.
"""

CLOSED_POLL_S = 60.0
"""How long to sleep outside business hours. Nothing happens on these ticks, so
the only thing this number costs is how quickly the shop opens at 11:00."""


class WindowHost(Protocol):
    """Everything the loop needs from the layers above it.

    Injected rather than imported because `pipeline` cannot import `runtime`.
    Implemented for real in `cli.main`, and by a fake in the tests.
    """

    def now(self) -> datetime: ...

    def is_open(self, now: datetime | None = None) -> bool:
        """Is the shop open? The only question asked outside a render."""

    def claim_deadline(self, now: datetime | None = None) -> datetime:
        """The bell new work must finish before — the end of business hours."""

    def ensure_pod(self, queue: JobQueue) -> Any:
        """A live window, opening one if hours and budget allow. May raise."""

    def wait_ready(self, session: Any) -> float:
        """Block until ComfyUI answers on that pod."""

    def providers_for(self, session: Any) -> dict[MediaKind, Any]:
        """The clip and image backends bound to that pod."""

    def touch_activity(self) -> None:
        """Reset the idle reaper's timer. Called after every render."""

    async def deliver(self, job: Job, asset: Path | None) -> str:
        """Tell the group. `asset` is None when the job failed.

        Injected for the same reason as the rest: `bots` is a leaf that no
        library layer may import, so the loop cannot reach the push client and
        the composition root hands it down instead.
        """


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
) -> str:
    """One pass of the loop. Returns what it did, for the log and the tests.

    Ordered cheapest-first so the expensive calls are never reached on the
    ticks that do not need them: closed, then empty, then — and only then — a
    pod.
    """
    report.ticks += 1
    now = host.now()

    if not host.is_open(now):
        report.last_action = "closed"
        return "closed"

    if not claimable(queue):
        report.last_action = "idle"
        return "idle"

    deadline = host.claim_deadline(now)
    if not may_claim(deadline):
        # The last minutes of the day belong to finishing, not starting. A
        # render begun now is billed and then thrown away by the lease.
        report.last_action = "too-late"
        return "too-late"

    session = host.ensure_pod(queue)
    host.wait_ready(session)
    providers = host.providers_for(session)

    job = queue.claim_next(gpu_tier=getattr(session, "tier_label", None))
    if job is None:
        # Someone else took it between the check and the claim. Harmless: the
        # atomic claim is exactly what makes that a non-event rather than a
        # double charge.
        report.last_action = "raced"
        return "raced"

    return await _run_one(
        queue, host, job, providers, files_dir=files_dir,
        deadline=deadline, report=report, poll_interval_s=poll_interval_s,
    )


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
    provider = providers[job.media_kind]
    caps = provider.capabilities()
    started = time.monotonic()

    try:
        if job.media_kind is MediaKind.IMAGE:
            asset = await render_image(
                job, provider, caps, files_dir, deadline, poll_interval_s
            )
        else:
            asset = await render_clip(
                job, provider, caps, files_dir, deadline, poll_interval_s
            )
    except ProviderError as exc:
        if job.attempts >= MAX_ATTEMPTS:
            queue.fail(job.id, f"provider failed {job.attempts}x: {exc}")
            report.failed += 1
            _log.warning("job %d failed for good: %s", job.id, exc)
            await _deliver(queue, host, job.id, None, report)
            report.last_action = "failed"
            return "failed"
        # No delivery on a requeue: the request is still alive and will be
        # tried again. Announcing every transient hiccup would bill a push per
        # attempt for something the user does not need to know about.
        queue.fail(job.id, f"provider: {exc}", requeue=True)
        report.requeued += 1
        _log.warning("job %d requeued (attempt %d): %s", job.id, job.attempts, exc)
        report.last_action = "requeued"
        return "requeued"
    except AIStudioError as exc:
        queue.fail(job.id, str(exc))
        report.failed += 1
        _log.warning("job %d failed terminally: %s", job.id, exc)
        await _deliver(queue, host, job.id, None, report)
        report.last_action = "failed"
        return "failed"

    queue.complete(job.id, str(asset))
    report.completed += 1
    report.seconds.append(time.monotonic() - started)
    # Without this a long render looks like idleness and the reaper closes the
    # window out from under the next job.
    host.touch_activity()
    _log.info("job %d done in %.0fs -> %s", job.id, report.seconds[-1], asset)
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
        _log.error("job %d could not be delivered: %s", job.id, exc)
        report.undelivered += 1
        return "failed"

    queue.mark_delivered(job.id)
    report.delivered += 1
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

    while max_ticks is None or report.ticks < max_ticks:
        try:
            action = await tick(
                queue, host, files_dir=files_dir, report=report,
                poll_interval_s=poll_interval_s,
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
