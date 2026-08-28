"""gpt-oss-20b, through the pod-side inference server
(`deploy/inference_server.py`).

Registered once under `"chat"` in `providers/registry.py` -- unlike
`UnderstandingProvider`, which is parametrized over three modalities sharing
one wire protocol, there is only ever one chat model, so one concrete class
is enough.

Cost/latency figures below are `[speculative]` -- gpt-oss-20b has not run on
this project's own hardware yet. See `docs/model-gpt-oss-20b.md`.
"""

from __future__ import annotations

import time
from typing import Any

from ai_studio.config.settings import get_settings
from ai_studio.core.chat_spec import ChatAsset, ChatCapabilities, ChatJob, ChatRequest
from ai_studio.core.enums import JobState
from ai_studio.core.errors import ProviderError, ProviderJobFailed
from ai_studio.inference.client import InferenceClient

DEFAULT_HOURLY_USD = 0.74
"""Same shared-pod rate as H3/Flux/understanding -- chat runs on whichever
ladder rung answered, same as everything else on this pod."""


def chat_capabilities(*, hourly_usd: float = DEFAULT_HOURLY_USD) -> ChatCapabilities:
    """Capabilities for the one chat model.

    `[speculative]` cost/latency until measured on real hardware -- negligible
    next to H3's 2-6min/clip either way, the same reasoning
    `understanding_capabilities()` uses.
    """
    expected_latency_s = 20.0
    cost = round(hourly_usd * expected_latency_s / 3600.0, 6)
    return ChatCapabilities(
        provider="chat",
        model_id="openai/gpt-oss-20b",
        expected_latency_s=expected_latency_s,
        cost_per_call_usd=cost,
    )


class ChatProvider:
    """Drives the pod-side inference server's `"chat"` modality.

    Constructed once by `providers_for()` -- a single instance, unlike the
    three per-modality `UnderstandingProvider` instances, since there is only
    one chat model to talk to.
    """

    name = "chat"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        hourly_usd: float = DEFAULT_HOURLY_USD,
        **_: Any,
    ) -> None:
        settings = get_settings()
        self.client = InferenceClient(
            base_url or settings.inference_url, timeout_s=settings.inference_timeout_s
        )
        self._hourly_usd = hourly_usd
        self._caps = chat_capabilities(hourly_usd=hourly_usd)

    def capabilities(self) -> ChatCapabilities:
        return self._caps

    # ---------------------------------------------------------------- submit

    async def submit(self, request: ChatRequest) -> ChatJob:
        now = time.time()
        history = request.extra.get("history")
        # The developer/system instruction block for the reply, chosen by
        # the pipeline (prompts/chat.py) -- the provider only carries it.
        system = request.extra.get("system")
        job_id = await self.client.submit_chat_job(request.text, history=history, system=system)
        return ChatJob(
            provider=self.name,
            job_id=job_id,
            shot_id=request.shot_id,
            state=JobState.QUEUED,
            submitted_at=now,
            updated_at=now,
        )

    # ------------------------------------------------------------------ poll

    async def poll(self, job: ChatJob) -> ChatJob:
        now = time.time()
        payload = await self.client.poll_job(job.job_id)
        state = str(payload.get("state") or "")
        if state == "completed":
            return job.with_state(
                JobState.COMPLETED,
                now=now,
                raw={
                    **job.raw,
                    "result_text": payload.get("result_text") or "",
                    "reasoning_exhausted": bool(payload.get("reasoning_exhausted")),
                },
            )
        if state == "failed":
            return job.with_state(
                JobState.FAILED, now=now, error=str(payload.get("error") or "unknown")
            )
        if state == "queued":
            return job.with_state(JobState.QUEUED, now=now)
        if state == "running":
            return job.with_state(JobState.RUNNING, now=now)
        # Fail loudly: a state this client does not know is a wire-protocol
        # drift against deploy/inference_server.py, not a job that is still
        # running. Treating it as "running" would hang the job until the
        # window deadline and hide the actual mismatch.
        raise ProviderError(f"job {job.job_id}: unknown state {state!r} from the inference server")

    # ----------------------------------------------------------------- fetch

    async def fetch(self, job: ChatJob) -> ChatAsset:
        if job.state is not JobState.COMPLETED:
            raise ProviderJobFailed(f"job {job.job_id} is {job.state.value}, not completed")
        result_text = str(job.raw.get("result_text") or "")
        elapsed = max(job.elapsed_s, 1.0)
        return ChatAsset(
            shot_id=job.shot_id,
            provider=self.name,
            job_id=job.job_id,
            result_text=result_text[: self._caps.max_output_chars],
            reasoning_exhausted=bool(job.raw.get("reasoning_exhausted")),
            cost_usd=round(self._hourly_usd * elapsed / 3600.0, 6),
        )

    # ---------------------------------------------------------------- cancel

    async def cancel(self, job: ChatJob) -> None:
        await self.client.cancel_job(job.job_id)

    # ----------------------------------------------------------- GPU hand-off

    async def evict(self) -> None:
        """Release the currently-loaded model so a ComfyUI generation job can
        use the same card. See `inference.client.InferenceClient.unload`."""
        await self.client.unload()

    async def aclose(self) -> None:
        await self.client.aclose()
