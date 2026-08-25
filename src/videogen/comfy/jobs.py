"""Shared ComfyUI job-lifecycle translation.

Every provider driving ComfyUI (H3's `ComfyUIProvider`, Flux's
`FluxComfyUIProvider`) submits a different request shape and builds a
different asset type, but translating ComfyUI's own `/history`/`/queue`/
`/interrupt` semantics into this project's `JobState` is identical work
regardless of the model behind it — a node's completion looks the same to
`/history` whether the graph is MiniMax H3 or Flux. Sharing it here, in
`comfy` rather than duplicated per-provider, means a fix to that translation
(a new ComfyUI status string, a queue-position race) is one edit, not one per
model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, TypeVar

from videogen.comfy.client import ComfyClient, ComfyOutput
from videogen.core.enums import JobState
from videogen.core.errors import ProviderError, ProviderJobFailed


class _JobLike(Protocol):
    """Structural shape shared by `ClipJob` and `ImageJob`.

    Declared as read-only properties, not plain attributes: both are frozen
    pydantic models, so their fields are get-only — a Protocol's plain
    `x: T` attribute annotation implies both get *and* set, which a frozen
    model's field can never satisfy.
    """

    @property
    def job_id(self) -> str: ...
    @property
    def raw(self) -> dict[str, Any]: ...
    @property
    def state(self) -> JobState: ...

    def with_state(self, state: JobState, *, now: float, **changes: Any) -> Any: ...


J = TypeVar("J", bound=_JobLike)


async def poll_job(client: ComfyClient, job: J) -> J:
    """Refresh a job's state from ComfyUI's `/history` and `/queue`."""
    now = time.time()
    entry = await client.history(job.job_id)

    if entry is None:
        position = await client.queue_position(job.job_id)
        state = JobState.QUEUED if position is not None else JobState.RUNNING
        return job.with_state(state, now=now, queue_position=position)  # type: ignore[no-any-return]

    status, error = ComfyClient.status_of(entry)
    if status == "success":
        outputs = ComfyClient.outputs_of(entry)
        if not outputs:
            return job.with_state(  # type: ignore[no-any-return]
                JobState.FAILED,
                now=now,
                error="ComfyUI reported success but produced no output files",
            )
        return job.with_state(  # type: ignore[no-any-return]
            JobState.COMPLETED, now=now, raw={**job.raw, "output": outputs[0].__dict__}
        )
    if status == "error":
        return job.with_state(JobState.FAILED, now=now, error=error or "execution error")  # type: ignore[no-any-return]
    return job.with_state(JobState.RUNNING, now=now)  # type: ignore[no-any-return]


async def cancel_job(client: ComfyClient) -> None:
    """ComfyUI can only interrupt the running prompt, not a queued one."""
    try:
        await client.interrupt()
    except ProviderError:
        return None


async def fetch_output(client: ComfyClient, job: J, dest: Path) -> Path:
    """Validate a completed job, download its recorded output, write it to
    `dest`. Returns `dest`; the caller probes the file and builds whichever
    asset type (`ClipAsset`/`ImageAsset`) fits what was actually produced."""
    raw_output = job.raw.get("output")
    if not isinstance(raw_output, dict):
        raise ProviderError(f"job {job.job_id} has no recorded output; poll it first")
    if job.state is not JobState.COMPLETED:
        raise ProviderJobFailed(f"job {job.job_id} is {job.state.value}, not completed")

    output = ComfyOutput(
        filename=str(raw_output["filename"]),
        subfolder=str(raw_output.get("subfolder", "")),
        type=str(raw_output.get("type", "output")),
    )
    payload = await client.download(output)

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return dest
