"""gpt-oss-20b on the pod, presented as the `LlmClient` the prompt builders take.

`prompts.convert.convert`, `prompts.flux.convert` and
`prompts.understanding.convert_question` all rewrite a request through one
protocol: `complete(system, user, *, max_tokens) -> str`. Until 2026-08-27 the
concrete client behind it was a RunPod serverless Qwen2.5-7B endpoint
(`llm.endpoint.RunpodLlmClient`, 📏 183 s cold). By decision that endpoint is
not used any more: the rewriter is the same gpt-oss-20b the pod already serves
for `/himonkey`, reached through the inference server's chat modality with a
`system` instruction block and `json_only` decoding.

**The caller owns the GPU slot.** One 24 GB card holds one model; this client
does not know which one is resident and never evicts anything. `pipeline.
worker.prepare` calls `make_room_for(MediaKind.CHAT, providers)` once, then
rewrites every pending job while gpt-oss is loaded -- N clips pay one load,
not N. Calling `complete()` while ComfyUI's checkpoint is resident is not an
error; it is just a slow first call (📏 57-68 s load) and, if the card is
full, a failed one.

Sits in `pipeline` (may import `inference`; `prompts` may not).
"""

from __future__ import annotations

import asyncio
import logging
import time

from ai_studio.core.errors import ProviderError
from ai_studio.inference.client import InferenceClient

_log = logging.getLogger("ai_studio.pod_llm")


class PodLlmClient:
    """`LlmClient` over the inference server's chat modality."""

    def __init__(
        self,
        client: InferenceClient,
        *,
        job_timeout_s: float = 300.0,
        poll_interval_s: float = 3.0,
        reasoning_effort: str = "low",
    ) -> None:
        self.client = client
        self.job_timeout_s = job_timeout_s
        self.poll_interval_s = poll_interval_s
        self.reasoning_effort = reasoning_effort
        self.last_total_s: float | None = None
        """Wall-clock of the last `complete()`, for the worker's log line; a
        first call well over ~30 s means the model was loaded for it."""

    async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
        started = time.monotonic()
        job_id = await self.client.submit_chat_job(
            user,
            system=system,
            max_new_tokens=max_tokens,
            reasoning_effort=self.reasoning_effort,
            json_only=True,
        )
        deadline = started + self.job_timeout_s
        while True:
            await asyncio.sleep(self.poll_interval_s)
            payload = await self.client.poll_job(job_id)
            state = str(payload.get("state") or "")
            if state == "completed":
                self.last_total_s = time.monotonic() - started
                _log.info("rewrite done", extra={"stage": "prepare", "seconds": round(self.last_total_s, 1),
                                                  "model": "gpt-oss-20b", "pod_job": job_id})
                return str(payload.get("result_text") or "")
            if state == "failed":
                raise ProviderError(f"rewrite failed on the pod: {payload.get('error') or 'unknown'}")
            if state not in ("queued", "running"):
                raise ProviderError(f"rewrite job {job_id}: unknown state {state!r}")
            if time.monotonic() >= deadline:
                try:
                    await self.client.cancel_job(job_id)
                finally:
                    _log.warning("rewrite job %s exceeded %.0fs; cancelled", job_id, self.job_timeout_s)
                raise ProviderError(f"rewrite exceeded {self.job_timeout_s:.0f}s on the pod")

    async def aclose(self) -> None:
        await self.client.aclose()
