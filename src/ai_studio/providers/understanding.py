"""Understanding models (moondream3, Qwen3-Omni-Captioner, Tarsier2) through
the pod-side inference server (`deploy/inference_server.py`).

One parametrized class rather than three concrete ones: the three models
share one wire protocol by construction -- the inference server hides which
model is loaded behind one HTTP surface -- so three concrete classes would be
near-total duplication of this submit/poll/fetch boilerplate. Registered
three times under three names (`understand-image`/`understand-audio`/
`understand-video`) in `providers/registry.py` so each can still be selected
directly, e.g. from the CLI.

Every cost/latency figure below is `[speculative]` -- none of the three
models have run on this project's own hardware yet. See `docs/model-
moondream3.md`, `docs/model-qwen3-omni-captioner.md`, `docs/model-
tarsier2.md`.
"""

from __future__ import annotations

import time
from typing import Any

from ai_studio.config.settings import get_settings
from ai_studio.core.enums import JobState, MediaKind
from ai_studio.core.errors import ProviderJobFailed, ProviderSubmitError
from ai_studio.core.understanding_spec import (
    UnderstandingAsset,
    UnderstandingCapabilities,
    UnderstandingJob,
    UnderstandingRequest,
)
from ai_studio.inference.client import InferenceClient

DEFAULT_HOURLY_USD = 0.74
"""Same shared-pod rate as H3/Flux -- understanding runs on whichever ladder
rung answered, same as generation does."""

_CAPABILITY_DEFAULTS: dict[MediaKind, dict[str, Any]] = {
    MediaKind.IMAGE_UNDERSTAND: dict(
        provider="understanding",
        model_id="moondream/moondream3-preview",
        accepts_prompt=True,
        max_input_seconds=None,
        expected_latency_s=15.0,
    ),
    MediaKind.AUDIO_UNDERSTAND: dict(
        provider="understanding",
        model_id="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        accepts_prompt=False,
        max_input_seconds=30.0,
        expected_latency_s=45.0,
    ),
    MediaKind.VIDEO_UNDERSTAND: dict(
        provider="understanding",
        model_id="omni-research/Tarsier2-7b-0115",
        accepts_prompt=True,
        max_input_seconds=120.0,
        expected_latency_s=30.0,
    ),
}

_MODALITY_WIRE_NAME = {
    MediaKind.IMAGE_UNDERSTAND: "image",
    MediaKind.AUDIO_UNDERSTAND: "audio",
    MediaKind.VIDEO_UNDERSTAND: "video",
}


def understanding_capabilities(
    modality: MediaKind, *, hourly_usd: float = DEFAULT_HOURLY_USD
) -> UnderstandingCapabilities:
    """Capabilities for one of the three understanding modalities.

    `[speculative]` cost/latency until measured on real hardware -- the
    latency figures above are a rough starting point for cost estimation,
    negligible next to H3's 2-6min/clip either way.
    """
    try:
        defaults = _CAPABILITY_DEFAULTS[modality]
    except KeyError:
        raise ProviderSubmitError(f"not an understanding modality: {modality!r}") from None
    cost = round(hourly_usd * defaults["expected_latency_s"] / 3600.0, 6)
    return UnderstandingCapabilities(modality=modality, cost_per_call_usd=cost, **defaults)


class UnderstandingProvider:
    """Drives the pod-side inference server for one modality.

    Constructed once per modality by `providers_for()` -- three instances,
    one each for `IMAGE_UNDERSTAND`/`AUDIO_UNDERSTAND`/`VIDEO_UNDERSTAND` --
    exactly the way `ComfyUIProvider` and `FluxComfyUIProvider` are two
    separate instances sharing one `ComfyClient`-shaped transport pattern.
    """

    name = "understanding"

    def __init__(
        self,
        *,
        modality: MediaKind,
        base_url: str | None = None,
        hourly_usd: float = DEFAULT_HOURLY_USD,
        **_: Any,
    ) -> None:
        settings = get_settings()
        self.modality = modality
        self.client = InferenceClient(
            base_url or settings.inference_url, timeout_s=settings.inference_timeout_s
        )
        self._hourly_usd = hourly_usd
        self._caps = understanding_capabilities(modality, hourly_usd=hourly_usd)

    def capabilities(self) -> UnderstandingCapabilities:
        return self._caps

    # ---------------------------------------------------------------- submit

    async def submit(self, request: UnderstandingRequest) -> UnderstandingJob:
        self._caps.check_prompt(request.prompt)
        now = time.time()
        job_id = await self.client.submit_job(
            _MODALITY_WIRE_NAME[self.modality], request.input_media_path, request.prompt
        )
        return UnderstandingJob(
            provider=self.name,
            job_id=job_id,
            shot_id=request.shot_id,
            state=JobState.QUEUED,
            submitted_at=now,
            updated_at=now,
        )

    # ------------------------------------------------------------------ poll

    async def poll(self, job: UnderstandingJob) -> UnderstandingJob:
        now = time.time()
        payload = await self.client.poll_job(job.job_id)
        state = str(payload.get("state") or "running")
        if state == "completed":
            return job.with_state(
                JobState.COMPLETED,
                now=now,
                raw={**job.raw, "result_text": payload.get("result_text") or ""},
            )
        if state == "failed":
            return job.with_state(
                JobState.FAILED, now=now, error=str(payload.get("error") or "unknown")
            )
        if state == "queued":
            return job.with_state(JobState.QUEUED, now=now)
        return job.with_state(JobState.RUNNING, now=now)

    # ----------------------------------------------------------------- fetch

    async def fetch(self, job: UnderstandingJob) -> UnderstandingAsset:
        if job.state is not JobState.COMPLETED:
            raise ProviderJobFailed(f"job {job.job_id} is {job.state.value}, not completed")
        result_text = str(job.raw.get("result_text") or "")
        elapsed = max(job.elapsed_s, 1.0)
        return UnderstandingAsset(
            shot_id=job.shot_id,
            provider=self.name,
            job_id=job.job_id,
            modality=self.modality,
            result_text=result_text[: self._caps.max_output_chars],
            cost_usd=round(self._hourly_usd * elapsed / 3600.0, 6),
        )

    # ---------------------------------------------------------------- cancel

    async def cancel(self, job: UnderstandingJob) -> None:
        await self.client.cancel_job(job.job_id)

    # ----------------------------------------------------------- GPU hand-off

    async def evict(self) -> None:
        """Release the currently-loaded model so a ComfyUI generation job can
        use the same card. See `inference.client.InferenceClient.unload`."""
        await self.client.unload()

    async def aclose(self) -> None:
        await self.client.aclose()
