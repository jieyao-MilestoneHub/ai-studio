"""MiniMax H3 through ComfyUI.

The provider protocol maps onto ComfyUI's HTTP API one-to-one, which is the
main reason this shape was chosen:

| protocol | ComfyUI                              |
|----------|--------------------------------------|
| submit   | `POST /prompt` -> `prompt_id`        |
| poll     | `GET /history/<prompt_id>`           |
| fetch    | `GET /view?filename=...`             |
| cancel   | `POST /interrupt`                    |

Cost note: a pod bills by wall-clock hour, not by second of video, so
`cost_per_second_usd` here is *derived* — generation seconds per output second,
divided into the hourly rate. That makes a slow render correctly more expensive
than a fast one of the same length, which is what the budget guard needs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ai_studio import media
from ai_studio.comfy.client import ComfyClient
from ai_studio.comfy.graph import Workflow
from ai_studio.comfy.jobs import cancel_job, fetch_output, poll_job
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import GenMode, JobState
from ai_studio.core.errors import ProviderSubmitError, UnknownKeyError
from ai_studio.core.model_profile import MINIMAX_H3
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ClipRequest, ProviderCapabilities
from ai_studio.storage.base import sha256_file

DEFAULT_HOURLY_USD = 0.74
"""RTX 4090 Secure Cloud, verified against the live RunPod catalogue.

Note this differs from the 0.69 quoted in some write-ups; 0.69 is the RTX 5090
*community* rate. Community 4090 is 0.34/hr but runs on third-party consumer
machines that can be pre-empted without warning.
"""


def h3_capabilities(
    width: int = 864,
    height: int = 480,
    *,
    hourly_usd: float = DEFAULT_HOURLY_USD,
    clip_seconds: float = 5.0,
) -> ProviderCapabilities:
    """Capabilities for MiniMax H3 at a given canvas, derived from the profile.

    The canvas table used to live here as `MEASURED_LATENCY_S` — named
    "measured" while its own comment graded it `[reported]` — with a
    `.get((w, h), 300.0)` fallback. That fallback was the bug: an off-table
    canvas silently inherited 1344x768's timing, so both its cost estimate and
    its job timeout became someone else's numbers. `require_canvas` raises
    instead.
    """
    canvas = MINIMAX_H3.require_canvas(width, height)
    if canvas.latency_s is None:  # pragma: no cover - every H3 canvas has one
        raise UnknownKeyError("measured latency for canvas", canvas.label, [])

    cost_per_clip = hourly_usd * canvas.latency_s / 3600.0
    grid = MINIMAX_H3.frame_grid
    assert grid is not None and MINIMAX_H3.fps is not None  # H3 produces video
    return ProviderCapabilities(
        provider="comfyui",
        model_id=f"minimax-h3-fl2va@{canvas.label}",
        native_width=canvas.width,
        native_height=canvas.height,
        native_fps=MINIMAX_H3.fps,
        modes=frozenset({GenMode.T2V, GenMode.I2V, GenMode.KEYFRAME, GenMode.REF2V}),
        min_clip_s=MINIMAX_H3.duration_for(grid.minimum),
        max_clip_s=15.0,
        # Frames come in steps of 17, so durations come in steps of 17/24s. The
        # grid's *offset* (n % 17 == 5) has no home in a float quantum, which is
        # why `ModelProfile.frames_for` is the only supported way to turn a
        # duration into a length. This field is the shadow that fact casts on a
        # provider-agnostic type, not a replacement for it.
        clip_duration_quantum=grid.step / MINIMAX_H3.fps,
        has_native_audio=MINIMAX_H3.has_native_audio,
        supports_seed=True,
        supports_negative_prompt=MINIMAX_H3.supports_negative_prompt,
        max_prompt_chars=MINIMAX_H3.max_prompt_chars,
        url_ttl_s=None,
        cost_per_second_usd=round(cost_per_clip / clip_seconds, 6),
        expected_latency_s=canvas.latency_s,
        max_concurrent_jobs=1,
    )


class ComfyUIProvider:
    """Drives a ComfyUI instance running MiniMax H3."""

    name = "comfyui"

    def __init__(
        self,
        workflow: Path | str,
        *,
        base_url: str | None = None,
        width: int = 864,
        height: int = 480,
        hourly_usd: float = DEFAULT_HOURLY_USD,
        **_: Any,
    ) -> None:
        settings = get_settings()
        self.workflow = Workflow.load(workflow)
        self._i2va_workflow = self._load_i2va_sibling(workflow)
        self.client = ComfyClient(
            base_url or settings.comfy_url, timeout_s=settings.comfy_timeout_s
        )
        # Stored, not just forwarded. `fetch` used to bill from the module
        # constant, so every clip rendered on the L40S rung ($1.004/hr) recorded
        # its cost at the 4090's $0.74 — about 26% low, feeding a wrong number
        # into anything that reads ClipAsset.cost_usd.
        self._hourly_usd = hourly_usd
        self._caps = h3_capabilities(width, height, hourly_usd=hourly_usd)

    @staticmethod
    def _load_i2va_sibling(workflow: Path | str) -> Workflow | None:
        """The image-conditioned graph next to a text-only one, if it exists.

        A separate file rather than a `first_frame` left disconnected in one
        shared graph: ComfyUI's JSON is a static graph, not an expression
        engine, so there is no way to make one file both wire and not-wire a
        `LoadImage` node depending on the request. `None` rather than raising
        -- a caller that passes some other workflow.json with no i2va sibling
        should still get ordinary text-to-video; the failure belongs at the
        moment an image actually needs it and there is nowhere to put it.
        """
        path = Path(workflow)
        sibling = path.with_name(path.name.replace("fl2va", "i2va"))
        if sibling == path or not sibling.is_file():
            return None
        return Workflow.load(sibling)

    def capabilities(self) -> ProviderCapabilities:
        return self._caps

    # ---------------------------------------------------------------- submit

    async def submit(self, request: ClipRequest) -> ClipJob:
        # Both checks are free and both run before a single GPU-second. This is
        # the boundary where a provider-agnostic ClipRequest becomes an H3
        # submission, so it is the only place that can catch a request H3 will
        # quietly reinterpret. Ahead of picking a graph, because a bad length is
        # bad on the i2va path too.
        MINIMAX_H3.require_canvas(request.width, request.height)

        grid = MINIMAX_H3.frame_grid
        assert grid is not None  # H3 produces video
        frames = round(request.duration_s * request.fps)
        if not grid.is_valid(frames):
            below, above = grid.neighbours(frames)
            raise ProviderSubmitError(
                f"{request.duration_s:.4f}s at {request.fps}fps is {frames} frames, "
                f"which MiniMax H3 cannot produce — legal lengths are {grid.step}k+"
                f"{grid.base} ({below} or {above} here). ComfyUI would snap it up "
                f"silently, so the clip would not be the length that was asked for. "
                f"Use ModelProfile.frames_for() to pick a duration."
            )

        workflow = self.workflow
        values: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "length": frames,
        }

        if request.first_frame_path is not None:
            if self._i2va_workflow is None:
                raise ProviderSubmitError(
                    f"a first frame was given but {self.workflow.source} has no "
                    "image-conditioned sibling workflow"
                )
            workflow = self._i2va_workflow
            source = Path(request.first_frame_path)
            try:
                image_bytes = source.read_bytes()
            except OSError as exc:
                raise ProviderSubmitError(f"could not read {source}: {exc}") from exc
            values["first_frame"] = await self.client.upload_image(image_bytes, source.name)

        for name, value in (
            ("seed", request.seed),
            ("steps", request.steps),
            ("filename", f"ai_studio_{request.shot_id}"),
            ("fps", request.fps),
        ):
            if value is not None and name in workflow.bindings:
                values[name] = value

        graph = workflow.with_values(values)
        now = time.time()
        prompt_id = await self.client.queue_prompt(graph)

        return ClipJob(
            provider=self.name,
            job_id=prompt_id,
            shot_id=request.shot_id,
            state=JobState.QUEUED,
            submitted_at=now,
            updated_at=now,
            raw={"width": request.width, "height": request.height, "fps": request.fps},
        )

    # ------------------------------------------------------------------ poll

    async def poll(self, job: ClipJob) -> ClipJob:
        return await poll_job(self.client, job)

    # ----------------------------------------------------------------- fetch

    async def fetch(self, job: ClipJob, dest: Path) -> ClipAsset:
        dest = await fetch_output(self.client, job, dest)

        info = media.probe(dest)
        elapsed = max(job.elapsed_s, 1.0)
        return ClipAsset(
            shot_id=job.shot_id,
            key=dest.name,
            sha256=sha256_file(dest),
            size_bytes=info.size_bytes,
            width=info.width,
            height=info.height,
            fps=info.fps,
            duration_s=info.duration_s,
            has_audio=info.has_audio,
            provider=self.name,
            job_id=job.job_id,
            # Billed by wall clock: what this clip actually occupied the GPU for.
            cost_usd=round(self._hourly_usd * elapsed / 3600.0, 6),
        )

    # ---------------------------------------------------------------- cancel

    async def cancel(self, job: ClipJob) -> None:
        """ComfyUI can only interrupt the running prompt, not a queued one."""
        await cancel_job(self.client)

    async def aclose(self) -> None:
        await self.client.aclose()
