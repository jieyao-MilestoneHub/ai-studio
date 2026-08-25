"""Flux.1-dev through ComfyUI.

Same protocol as `providers/comfyui.py` (submit/poll/fetch/cancel over
ComfyUI's `/prompt /history /view /interrupt`), but for a still image rather
than a clip: there is no frame count, so `submit()` never binds `length`, and
`fetch()` cannot use `media.probe()` (it raises on a file with no video
stream) — it uses `media.probe_image()` instead.

⚠️ Flux.1-dev ships under a non-commercial licence from Black Forest Labs;
Flux.1-schnell is Apache-2.0. See docs/model-flux.md before any commercial use.

Every size/timing figure in this module is `[speculative]` — nothing here has
been measured on real hardware yet. Re-measure and promote to 📏 once it has,
exactly as `comfyui.py`'s own H3 latency table does.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from videogen import media
from videogen.comfy.client import ComfyClient
from videogen.comfy.graph import IMAGE_REQUIRED_BINDINGS, Workflow
from videogen.comfy.jobs import cancel_job, fetch_output, poll_job
from videogen.config.settings import get_settings
from videogen.core.enums import GenMode, JobState
from videogen.core.image_provider_spec import (
    ImageAsset,
    ImageJob,
    ImageProviderCapabilities,
    ImageRequest,
)
from videogen.storage.base import sha256_file

DEFAULT_HOURLY_USD = 0.74
"""Same shared-pod rate as H3 — Flux runs on whichever ladder rung answered."""

DEFAULT_STEPS = 28
"""[speculative] placeholder for Flux.1-dev; schnell needs 1-4, dev needs more.
Re-measure on the actual pod before trusting this."""

MEASURED_LATENCY_S = 30.0
"""[speculative] single figure, not a per-resolution table — there is no
measurement yet at any resolution. Negligible next to H3's 2-6min/clip either
way, so a rough number is enough for cost estimation until re-measured."""


def flux_capabilities(
    width: int = 1024,
    height: int = 1024,
    *,
    hourly_usd: float = DEFAULT_HOURLY_USD,
    steps: int = DEFAULT_STEPS,
) -> ImageProviderCapabilities:
    """Capabilities for Flux.1-dev at a given native canvas."""
    cost_per_image = round(hourly_usd * MEASURED_LATENCY_S / 3600.0, 6)
    return ImageProviderCapabilities(
        provider="flux",
        model_id=f"flux-dev@{width}x{height}",
        native_width=width,
        native_height=height,
        modes=frozenset({GenMode.T2I}),
        supports_seed=True,
        supports_negative_prompt=False,
        max_prompt_chars=2000,
        output_format="png",
        cost_per_image_usd=cost_per_image,
        expected_latency_s=MEASURED_LATENCY_S,
        max_concurrent_jobs=1,
    )


class FluxComfyUIProvider:
    """Drives a ComfyUI instance running Flux.1-dev."""

    name = "flux"

    def __init__(
        self,
        workflow: Path | str,
        *,
        base_url: str | None = None,
        width: int = 1024,
        height: int = 1024,
        hourly_usd: float = DEFAULT_HOURLY_USD,
        steps: int = DEFAULT_STEPS,
        **_: Any,
    ) -> None:
        settings = get_settings()
        self.workflow = Workflow.load(workflow, required_bindings=IMAGE_REQUIRED_BINDINGS)
        self.client = ComfyClient(
            base_url or settings.comfy_url, timeout_s=settings.comfy_timeout_s
        )
        self._hourly_usd = hourly_usd
        self._caps = flux_capabilities(width, height, hourly_usd=hourly_usd, steps=steps)

    def capabilities(self) -> ImageProviderCapabilities:
        return self._caps

    # ---------------------------------------------------------------- submit

    async def submit(self, request: ImageRequest) -> ImageJob:
        values: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
        }
        for name, value in (
            ("seed", request.seed),
            ("steps", request.steps),
            ("filename", f"videogen_{request.shot_id}"),
        ):
            if value is not None and name in self.workflow.bindings:
                values[name] = value

        graph = self.workflow.with_values(values)
        now = time.time()
        prompt_id = await self.client.queue_prompt(graph)

        return ImageJob(
            provider=self.name,
            job_id=prompt_id,
            shot_id=request.shot_id,
            state=JobState.QUEUED,
            submitted_at=now,
            updated_at=now,
            raw={"width": request.width, "height": request.height},
        )

    # ------------------------------------------------------------------ poll

    async def poll(self, job: ImageJob) -> ImageJob:
        return await poll_job(self.client, job)

    # ----------------------------------------------------------------- fetch

    async def fetch(self, job: ImageJob, dest: Path) -> ImageAsset:
        dest = await fetch_output(self.client, job, dest)

        info = media.probe_image(dest)
        elapsed = max(job.elapsed_s, 1.0)
        return ImageAsset(
            shot_id=job.shot_id,
            key=dest.name,
            sha256=sha256_file(dest),
            size_bytes=info.size_bytes,
            width=info.width,
            height=info.height,
            format=dest.suffix.lstrip("."),
            provider=self.name,
            job_id=job.job_id,
            cost_usd=round(self._hourly_usd * elapsed / 3600.0, 6),
        )

    # ---------------------------------------------------------------- cancel

    async def cancel(self, job: ImageJob) -> None:
        """ComfyUI can only interrupt the running prompt, not a queued one."""
        await cancel_job(self.client)

    async def aclose(self) -> None:
        await self.client.aclose()
