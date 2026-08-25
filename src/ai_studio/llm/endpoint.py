"""The prompt-conversion LLM, on a RunPod serverless endpoint.

Serverless rather than on the GPU pod, for two reasons:

1. **No VRAM contention.** A measured H3 run peaked at 43.3GB on a 48GB card,
   leaving no room for a useful instruct model alongside it.
2. **It is available when the pod is not.** The GPU window is two hours a day,
   but requests arrive around the clock. Converting at request time means a user
   learns within seconds how their sentence was understood, instead of finding
   out twelve hours later.

`--model-reference` is what makes this cheap: RunPod caches the weights host-side
and documents that download time is not billed and cold starts drop to seconds.

**Async `/run` + poll, never the synchronous OpenAI route.** RunPod's own golden
path records a sync `/openai` call hitting a Cloudflare 524 at the 100-second
edge timeout during a cold load. Submitting and polling has no such ceiling.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

API_BASE = "https://api.runpod.ai/v2"

DEPLOY_COMMAND = """\
runpodctl serverless create --name ai-studio-prompt \\
  --hub-id runpod-workers/worker-vllm \\
  --model-reference https://huggingface.co/Qwen/Qwen2.5-7B-Instruct:main \\
  --workers-min 0 --workers-max 1 --idle-timeout 120
"""
"""How the endpoint is created. `workers-min 0` is the point: idle costs nothing.

`idle-timeout 120` keeps a worker warm for two minutes, so a burst of requests
in a group chat pays one cold start rather than one per message.
"""


class LlmError(Exception):
    """The endpoint refused, failed, or never finished."""


@dataclass
class LlmMetrics:
    """What a call actually cost and how long each phase took.

    Recorded because the plan's cold-start estimate is unmeasured, and this is
    the thing that measures it.
    """

    submitted_at: float = 0.0
    delay_s: float = 0.0
    execution_s: float = 0.0
    total_s: float = 0.0
    was_cold: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class RunpodLlmClient:
    """Calls a RunPod serverless vLLM endpoint via `/run` + `/status`."""

    def __init__(
        self,
        endpoint_id: str,
        api_key: str,
        *,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout_s: float = 300.0,
        poll_interval_s: float = 3.0,
        cold_threshold_s: float = 15.0,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.model = model
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.cold_threshold_s = cold_threshold_s
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.last_metrics = LlmMetrics()

    @property
    def _base(self) -> str:
        return f"{API_BASE}/{self.endpoint_id}"

    async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
        """Submit a chat completion and wait for it. Returns the reply text."""
        payload = {
            "input": {
                # The vLLM worker's OpenAI-compatible passthrough. Kept in one
                # place because the exact key names are worker-specific and this
                # is the line to change if the worker's contract moves.
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.4,
                },
            }
        }

        started = time.time()
        async with httpx.AsyncClient(timeout=60.0, headers=self._headers) as client:
            job_id = await self._submit(client, payload)
            result = await self._await_result(client, job_id, started)

        self.last_metrics = LlmMetrics(
            submitted_at=started,
            delay_s=float(result.get("delayTime", 0) or 0) / 1000,
            execution_s=float(result.get("executionTime", 0) or 0) / 1000,
            total_s=time.time() - started,
            was_cold=(float(result.get("delayTime", 0) or 0) / 1000) > self.cold_threshold_s,
            raw={k: result.get(k) for k in ("delayTime", "executionTime", "status", "workerId")},
        )
        return _reply_text(result.get("output"))

    # --------------------------------------------------------------- internals

    async def _submit(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> str:
        try:
            response = await client.post(f"{self._base}/run", json=payload)
        except httpx.HTTPError as exc:
            raise LlmError(f"could not reach the endpoint: {exc}") from exc
        if response.status_code >= 400:
            raise LlmError(f"submit refused ({response.status_code}): {response.text[:400]}")
        job_id = response.json().get("id")
        if not job_id:
            raise LlmError(f"no job id in the response: {response.text[:200]}")
        return str(job_id)

    async def _await_result(
        self, client: httpx.AsyncClient, job_id: str, started: float
    ) -> dict[str, Any]:
        while True:
            if time.time() - started > self.timeout_s:
                raise LlmError(f"job {job_id} exceeded {self.timeout_s}s")
            await asyncio.sleep(self.poll_interval_s)
            try:
                response = await client.get(f"{self._base}/status/{job_id}")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise LlmError(f"status poll failed: {exc}") from exc

            body = response.json()
            if not isinstance(body, dict):
                raise LlmError(f"status was not a JSON object: {str(body)[:200]}")
            status = str(body.get("status", "")).upper()
            if status == "COMPLETED":
                return body
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise LlmError(f"job {job_id} ended {status}: {str(body.get('error'))[:300]}")


def _reply_text(output: Any) -> str:
    """Dig the assistant text out of whatever shape the worker returned.

    Tolerant on purpose: workers wrap OpenAI responses differently across
    versions, and a shape change should degrade to a clear error rather than an
    AttributeError deep in a parser.
    """
    if isinstance(output, list) and output:
        output = output[0]
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content"):
                    return str(message["content"])
                if first.get("text"):
                    return str(first["text"])
        for key in ("text", "content", "generated_text"):
            if output.get(key):
                return str(output[key])
    raise LlmError(f"could not find reply text in the output: {str(output)[:300]}")


class ScriptedLlmClient:
    """Returns canned replies. For tests and for offline development."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
        self.calls.append((system, user))
        if not self._replies:
            raise LlmError("no scripted replies left")
        return self._replies.pop(0)
