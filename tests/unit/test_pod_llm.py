"""`PodLlmClient`: gpt-oss-20b on the pod behind the `LlmClient` protocol."""

from __future__ import annotations

import httpx
import pytest

from ai_studio.core.errors import ProviderError
from ai_studio.inference.client import InferenceClient
from ai_studio.pipeline.pod_llm import PodLlmClient


def _client(handler, **kw) -> PodLlmClient:
    inner = InferenceClient("http://pod:8189", transport=httpx.MockTransport(handler))
    return PodLlmClient(inner, poll_interval_s=0.0, **kw)


@pytest.mark.asyncio
async def test_complete_submits_a_json_only_chat_job_with_the_system_block() -> None:
    seen: dict[str, str] = {}
    polls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            seen.update({k: httpx.URL(f"http://x/?{k}={v}").params[k] for k, v in form.items()})
            return httpx.Response(200, json={"job_id": "rw-1"})
        polls["n"] += 1
        if polls["n"] == 1:
            return httpx.Response(200, json={"state": "running"})
        return httpx.Response(200, json={"state": "completed", "result_text": '{"prompt": "a cat"}'})

    client = _client(handler)
    reply = await client.complete("SYS", "USER", max_tokens=400)

    assert reply == '{"prompt": "a cat"}'
    assert seen["modality"] == "chat" and seen["prompt"] == "USER"
    assert seen["system"] == "SYS"
    assert seen["max_new_tokens"] == "400"
    assert seen["json_only"] == "true"
    assert seen["reasoning_effort"] == "low"
    assert client.last_total_s is not None
    await client.aclose()


@pytest.mark.asyncio
async def test_a_failed_rewrite_raises_provider_error_with_the_pods_reason() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            return httpx.Response(200, json={"job_id": "rw-2"})
        return httpx.Response(200, json={"state": "failed", "error": "CUDA out of memory"})

    client = _client(handler)
    with pytest.raises(ProviderError, match="CUDA out of memory"):
        await client.complete("SYS", "USER")
    await client.aclose()


@pytest.mark.asyncio
async def test_a_rewrite_past_the_deadline_is_cancelled_and_raises() -> None:
    cancelled: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            return httpx.Response(200, json={"job_id": "rw-3"})
        if request.url.path.startswith("/cancel/"):
            cancelled.append(request.url.path)
            return httpx.Response(200, json={"cancelled": True})
        return httpx.Response(200, json={"state": "running"})

    client = _client(handler, job_timeout_s=0.0)
    with pytest.raises(ProviderError, match="exceeded"):
        await client.complete("SYS", "USER")
    assert cancelled == ["/cancel/rw-3"]
    await client.aclose()


@pytest.mark.asyncio
async def test_an_unknown_poll_state_raises_rather_than_hanging() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            return httpx.Response(200, json={"job_id": "rw-4"})
        return httpx.Response(200, json={"state": "paused"})

    client = _client(handler)
    with pytest.raises(ProviderError, match="unknown state 'paused'"):
        await client.complete("SYS", "USER")
    await client.aclose()
