"""The chat provider: `ChatCapabilities`, and the real `ChatProvider`
(HTTP mocked, same style as `test_understanding_provider.py`).
"""

from __future__ import annotations

import httpx
import pytest

from ai_studio.core.chat_spec import ChatRequest
from ai_studio.core.enums import JobState
from ai_studio.core.errors import ProviderError, ProviderJobFailed
from ai_studio.inference.client import InferenceClient
from ai_studio.providers.chat import ChatProvider, chat_capabilities

# --------------------------------------------------------- capabilities


def test_chat_capabilities_name_the_model() -> None:
    caps = chat_capabilities()
    assert caps.model_id == "openai/gpt-oss-20b"
    assert caps.max_concurrent_jobs == 1
    assert caps.cost_per_call_usd >= 0


# ------------------------------------------------------------ real provider


def _provider(handler) -> ChatProvider:
    provider = ChatProvider(base_url="http://pod:8189")
    provider.client = InferenceClient("http://pod:8189", transport=httpx.MockTransport(handler))
    return provider


@pytest.mark.asyncio
async def test_submit_poll_fetch_roundtrip() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            return httpx.Response(200, json={"job_id": "job-1"})
        if request.url.path == "/poll/job-1":
            return httpx.Response(200, json={"state": "completed", "result_text": "hi there"})
        return httpx.Response(404)

    provider = _provider(handler)
    request = ChatRequest(shot_id="job1", text="hello")

    job = await provider.submit(request)
    assert job.state is JobState.QUEUED
    job = await provider.poll(job)
    assert job.state is JobState.COMPLETED

    asset = await provider.fetch(job)
    assert asset.result_text == "hi there"
    assert asset.cost_usd >= 0
    await provider.aclose()


@pytest.mark.asyncio
async def test_submit_forwards_history_from_request_extra() -> None:
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        from urllib.parse import parse_qs

        fields = parse_qs(request.content.decode("utf-8"))
        seen["has_history"] = "prior turn" in fields.get("history", [""])[0]
        return httpx.Response(200, json={"job_id": "job-1"})

    provider = _provider(handler)
    request = ChatRequest(
        shot_id="job1", text="hello",
        extra={"history": '[["user", "prior turn"]]'},
    )
    await provider.submit(request)
    assert seen["has_history"]
    await provider.aclose()


@pytest.mark.asyncio
async def test_a_failed_backend_job_polls_to_failed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={"state": "failed", "error": "generation timed out"})

    provider = _provider(handler)
    request = ChatRequest(shot_id="job1", text="hello")
    job = await provider.submit(request)
    job = await provider.poll(job)
    assert job.state is JobState.FAILED
    assert job.error == "generation timed out"
    await provider.aclose()


@pytest.mark.asyncio
async def test_an_unknown_poll_state_raises_instead_of_hanging() -> None:
    """A state string this client does not know is a wire-protocol drift
    against deploy/inference_server.py; treating it as "running" would keep
    the job alive until the window deadline and hide the mismatch."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={"state": "paused"})

    provider = _provider(handler)
    job = await provider.submit(ChatRequest(shot_id="job1", text="hello"))
    with pytest.raises(ProviderError, match="unknown state 'paused'"):
        await provider.poll(job)
    await provider.aclose()


@pytest.mark.asyncio
async def test_fetch_before_completion_fails_loudly() -> None:
    from ai_studio.core.chat_spec import ChatJob

    provider = ChatProvider(base_url="http://pod:8189")
    job = ChatJob(
        provider="chat", job_id="nope", shot_id="s", state=JobState.RUNNING,
        submitted_at=0.0, updated_at=0.0,
    )
    with pytest.raises(ProviderJobFailed):
        await provider.fetch(job)
    await provider.aclose()


@pytest.mark.asyncio
async def test_evict_calls_unload() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200)

    provider = _provider(handler)
    await provider.evict()
    assert calls == ["/unload"]
    await provider.aclose()


@pytest.mark.asyncio
async def test_submit_forwards_the_system_prompt_from_extra() -> None:
    bodies: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            bodies.append(request.content.decode())
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={"state": "completed", "result_text": "ok"})

    provider = _provider(handler)
    await provider.submit(ChatRequest(shot_id="j", text="hi", extra={"system": "# Instructions"}))
    await provider.aclose()
    assert "system=%23+Instructions" in bodies[0] or "system=%23%20Instructions" in bodies[0]
