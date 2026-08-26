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

from ai_studio import media
from ai_studio.comfy.client import ComfyClient
from ai_studio.comfy.graph import IMAGE_REQUIRED_BINDINGS, Workflow
from ai_studio.comfy.jobs import cancel_job, fetch_output, poll_job
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import GenMode, JobState
from ai_studio.core.image_provider_spec import (
    ImageAsset,
    ImageJob,
    ImageProviderCapabilities,
    ImageRequest,
)
from ai_studio.storage.base import sha256_file

DEFAULT_HOURLY_USD = 0.74
"""Same shared-pod rate as H3 — Flux runs on whichever ladder rung answered."""

DEFAULT_STEPS = 28
"""[speculative] placeholder for Flux.1-dev; schnell needs 1-4, dev needs more.
Re-measure on the actual pod before trusting this."""

MEASURED_LATENCY_S = 30.0
"""[speculative] single figure, not a per-resolution table — there is no
measurement yet at any resolution. Negligible next to H3's 2-6min/clip either
way, so a rough number is enough for cost estimation until re-measured."""

DEFAULT_LORA_STRENGTH = 1.0
"""Weight given to `flux_nsfw_uncensored_v1.safetensors`, the value its own
model card uses `[reported]`.

Deliberately a constructor argument rather than a constant baked into the
graph: the first live check for this LoRA is two images at the same seed with
this at 1.0 and 0.0 (PLAN.md Phase 7.1). If they come out identical, the LoRA
is wired to nothing — and that failure is silent, because a graph whose LoRA
feeds no consumer renders perfectly happily."""


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
        lora_strength: float = DEFAULT_LORA_STRENGTH,
        **_: Any,
    ) -> None:
        settings = get_settings()
        self.workflow = Workflow.load(workflow, required_bindings=IMAGE_REQUIRED_BINDINGS)
        self.client = ComfyClient(
            base_url or settings.comfy_url, timeout_s=settings.comfy_timeout_s
        )
        self._hourly_usd = hourly_usd
        self._lora_strength = lora_strength
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
            # Set explicitly on every submission rather than left to the JSON's
            # own default. The whole failure this guards against is a LoRA that
            # is present in the graph and doing nothing, which produces a
            # perfectly good picture of the wrong thing and raises no error.
            ("lora_strength", self._lora_strength),
            ("filename", f"ai_studio_{request.shot_id}"),
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
