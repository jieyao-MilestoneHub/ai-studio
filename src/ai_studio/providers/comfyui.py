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
from ai_studio.comfy.jobs import cancel_job, fetch_output, poll_job, upload_reference_image
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import GenMode, JobState
from ai_studio.core.errors import ProviderSubmitError
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ClipRequest, ProviderCapabilities
from ai_studio.storage.base import sha256_file

DEFAULT_HOURLY_USD = 0.74
"""RTX 4090 Secure Cloud, verified against the live RunPod catalogue.

Note this differs from the 0.69 quoted in some write-ups; 0.69 is the RTX 5090
*community* rate. Community 4090 is 0.34/hr but runs on third-party consumer
machines that can be pre-empted without warning.
"""

# Generation seconds for a 5s clip on an RTX 4090, by native canvas. [reported]
# The 608x352 figure is anomalous — it is *slower* than the larger 864x480,
# while the same source's RTX 3050 column scales normally with resolution. Most
# likely a first-run warm-up artifact. Re-measure before trusting it.
#
# `runs/benchmark/<YYYY-MM>.json` (storage.archive.update_benchmark_report)
# accumulates real per-canvas timing from actual /影片 traffic -- the natural
# source for re-measuring these, once a person has looked at it and decided
# a number is worth promoting (see CLAUDE.md, "Number honesty").
MEASURED_LATENCY_S: dict[tuple[int, int], float] = {
    (608, 352): 182.0,
    (864, 480): 133.0,
    (1280, 736): 361.0,
    (1344, 768): 300.0,
}


def h3_capabilities(
    width: int = 864,
    height: int = 480,
    *,
    hourly_usd: float = DEFAULT_HOURLY_USD,
    clip_seconds: float = 5.0,
) -> ProviderCapabilities:
    """Capabilities for MiniMax H3 at a given native canvas."""
    latency = MEASURED_LATENCY_S.get((width, height), 300.0)
    cost_per_clip = hourly_usd * latency / 3600.0
    return ProviderCapabilities(
        provider="comfyui",
        model_id=f"minimax-h3-fl2va@{width}x{height}",
        native_width=width,
        native_height=height,
        native_fps=24,
        modes=frozenset({GenMode.T2V, GenMode.I2V, GenMode.KEYFRAME, GenMode.REF2V}),
        min_clip_s=1.0,
        max_clip_s=15.0,
        clip_duration_quantum=None,
        has_native_audio=True,
        supports_seed=True,
        supports_negative_prompt=False,
        max_prompt_chars=8000,
        url_ttl_s=None,
        cost_per_second_usd=round(cost_per_clip / clip_seconds, 6),
        expected_latency_s=latency,
        max_concurrent_jobs=1,
    )


class ComfyUIProvider:
    """Drives a ComfyUI instance running MiniMax H3."""

    name = "comfyui"
    residency_group = "comfyui"

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
        self._i2va_workflow = Workflow.sibling(workflow, "fl2va", "i2va")
        self.client = ComfyClient(
            base_url or settings.comfy_url, timeout_s=settings.comfy_timeout_s
        )
        self._caps = h3_capabilities(width, height, hourly_usd=hourly_usd)

    def capabilities(self) -> ProviderCapabilities:
        return self._caps

    # ---------------------------------------------------------------- submit

    async def submit(self, request: ClipRequest) -> ClipJob:
        workflow = self.workflow
        values: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "length": round(request.duration_s * request.fps),
        }

        if request.first_frame_path is not None:
            if self._i2va_workflow is None:
                raise ProviderSubmitError(
                    f"a first frame was given but {self.workflow.source} has no "
                    "image-conditioned sibling workflow"
                )
            workflow = self._i2va_workflow
            values["first_frame"] = await upload_reference_image(
                self.client, request.first_frame_path
            )

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
        peak_vram_bytes = job.raw.get("peak_vram_bytes")
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
            cost_usd=round(DEFAULT_HOURLY_USD * elapsed / 3600.0, 6),
            peak_vram_gb=round(peak_vram_bytes / 2**30, 3) if peak_vram_bytes else None,
        )

    # ---------------------------------------------------------------- cancel

    async def cancel(self, job: ClipJob) -> None:
        """ComfyUI can only interrupt the running prompt, not a queued one."""
        await cancel_job(self.client)

    async def evict(self) -> None:
        """Release this workflow's resident model so an understanding job can
        use the same 24GB card. See `comfy.client.ComfyClient.free_memory`."""
        await self.client.free_memory()

    async def aclose(self) -> None:
        await self.client.aclose()
