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
from ai_studio.comfy.jobs import cancel_job, fetch_output, poll_job, upload_reference_image
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import GenMode, JobState
from ai_studio.core.errors import ProviderSubmitError
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

DEFAULT_I2I_DENOISE = 0.7
"""[speculative] How much of the source photo image-to-image is allowed to
repaint: 1.0 ignores it entirely (that is text-to-image), 0.0 returns it
untouched. 0.7 is the usual starting point for "keep the composition, change
the content"; re-tune on the pod. Only the `flux_dev_i2i.json` sibling has a
`denoise` binding — the text-to-image graph runs at 1.0 by construction."""

FACE_REPAIR_NODES = ("FaceDetailer", "UltralyticsDetectorProvider")
"""The Impact-Pack (+ Impact-Subpack) node classes `flux_dev_i2i_face.json`
uses. `deploy/pod_setup.sh` installs them best-effort; if either is missing
from `/object_info` the face sibling is never submitted."""

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
) -> ImageProviderCapabilities:
    """Capabilities for Flux.1-dev at a given native canvas.

    `steps` is not here on purpose: the capabilities snapshot has no field for
    it, and an argument that is accepted and ignored is the silent kind of
    wrong. It lives on the provider, which is what binds it into the graph."""
    cost_per_image = round(hourly_usd * MEASURED_LATENCY_S / 3600.0, 6)
    return ImageProviderCapabilities(
        provider="flux",
        model_id=f"flux-dev@{width}x{height}",
        native_width=width,
        native_height=height,
        modes=frozenset({GenMode.T2I, GenMode.I2I}),
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
        i2i_denoise: float = DEFAULT_I2I_DENOISE,
        **_: Any,
    ) -> None:
        settings = get_settings()
        self.workflow = Workflow.load(workflow, required_bindings=IMAGE_REQUIRED_BINDINGS)
        # Same pattern as H3's i2va sibling: image-to-image is a second static
        # graph with a LoadImage in it, found by name next to the text-only one.
        self._i2i_workflow = Workflow.sibling(
            workflow, "flux_dev", "flux_dev_i2i",
            required_bindings=IMAGE_REQUIRED_BINDINGS | {"source_image", "denoise"},
        )
        # A third sibling for `/短劇` keyframes: the same i2i graph with an
        # Impact-Pack FaceDetailer pass on the decoded image. Loaded lazily
        # and only *used* when the pod actually registers the nodes -- see
        # `supports_face_repair`. Absent file, absent nodes: plain i2i.
        self._i2i_face_workflow = Workflow.sibling(
            workflow, "flux_dev", "flux_dev_i2i_face",
            required_bindings=IMAGE_REQUIRED_BINDINGS | {"source_image", "denoise"},
        )
        self._face_repair_available: bool | None = None
        self._i2i_denoise = i2i_denoise
        self.client = ComfyClient(
            base_url or settings.comfy_url, timeout_s=settings.comfy_timeout_s
        )
        self._hourly_usd = hourly_usd
        self._lora_strength = lora_strength
        # The sampler step count the graph actually runs. A request may
        # override it; when it does not (the queue path never does), this is
        # bound explicitly rather than leaving the JSON's own default to decide,
        # so re-measuring DEFAULT_STEPS changes what renders.
        self._steps = steps
        self._caps = flux_capabilities(width, height, hourly_usd=hourly_usd)

    def capabilities(self) -> ImageProviderCapabilities:
        return self._caps

    # ---------------------------------------------------------------- submit

    async def submit(self, request: ImageRequest) -> ImageJob:
        workflow = self.workflow
        values: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
        }

        if request.source_image_path is not None:
            if self._i2i_workflow is None:
                raise ProviderSubmitError(
                    f"a source image was given but {self.workflow.source} has no "
                    "image-to-image sibling workflow"
                )
            workflow = self._i2i_workflow
            # `extra` is the sanctioned knob bag (core/image_provider_spec.py).
            # `face_repair` asks for the FaceDetailer sibling; it is honoured
            # only when the pod has the nodes, and the caller can read which
            # graph ran back off the returned job (`raw["face_repair"]`).
            if request.extra.get("face_repair") and await self.supports_face_repair():
                workflow = self._i2i_face_workflow  # type: ignore[assignment]
            values["source_image"] = await upload_reference_image(
                self.client, request.source_image_path
            )
            denoise = request.extra.get("denoise")
            values["denoise"] = float(denoise) if denoise is not None else self._i2i_denoise

        for name, value in (
            ("seed", request.seed),
            ("steps", request.steps if request.steps is not None else self._steps),
            # Set explicitly on every submission rather than left to the JSON's
            # own default. The whole failure this guards against is a LoRA that
            # is present in the graph and doing nothing, which produces a
            # perfectly good picture of the wrong thing and raises no error.
            ("lora_strength", self._lora_strength),
            ("filename", f"ai_studio_{request.shot_id}"),
        ):
            if value is not None and name in workflow.bindings:
                values[name] = value

        graph = workflow.with_values(values)
        now = time.time()
        prompt_id = await self.client.queue_prompt(graph)

        return ImageJob(
            provider=self.name,
            job_id=prompt_id,
            shot_id=request.shot_id,
            state=JobState.QUEUED,
            submitted_at=now,
            updated_at=now,
            raw={
                "width": request.width,
                "height": request.height,
                "face_repair": workflow is self._i2i_face_workflow and workflow is not None,
            },
        )

    async def supports_face_repair(self) -> bool:
        """Whether this pod can run the FaceDetailer sibling: the file exists
        next to the base workflow *and* ComfyUI registers the Impact-Pack
        nodes it uses. Probed once per provider (per pod) and cached; a probe
        failure counts as "no" -- degrading to plain i2i is the honest
        answer, not an error, and the drama records which one it got."""
        if self._face_repair_available is None:
            if self._i2i_face_workflow is None:
                self._face_repair_available = False
            else:
                try:
                    info = await self.client.object_info()
                except Exception:  # unreachable /object_info -> plain i2i
                    info = {}
                self._face_repair_available = all(n in info for n in FACE_REPAIR_NODES)
        return self._face_repair_available

    # ------------------------------------------------------------------ poll

    async def poll(self, job: ImageJob) -> ImageJob:
        return await poll_job(self.client, job)

    # ----------------------------------------------------------------- fetch

    async def fetch(self, job: ImageJob, dest: Path) -> ImageAsset:
        dest = await fetch_output(self.client, job, dest)

        info = media.probe_image(dest)
        elapsed = max(job.elapsed_s, 1.0)
        peak_vram_bytes = job.raw.get("peak_vram_bytes")
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
            peak_vram_gb=round(peak_vram_bytes / 2**30, 3) if peak_vram_bytes else None,
        )

    # ---------------------------------------------------------------- cancel

    async def cancel(self, job: ImageJob) -> None:
        """ComfyUI can only interrupt the running prompt, not a queued one."""
        await cancel_job(self.client)

    async def evict(self) -> None:
        """Release this workflow's resident model so an understanding job can
        use the same 24GB card. See `comfy.client.ComfyClient.free_memory`."""
        await self.client.free_memory()

    async def aclose(self) -> None:
        await self.client.aclose()
