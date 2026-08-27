"""`InferenceClient`: the HTTP surface for deploy/inference_server.py, the
pod-side process serving moondream3/Qwen3-Omni-Captioner/Tarsier2. Mirrors
the readiness/submit/poll/cancel coverage style used for `runtime.pod`'s
`httpx.MockTransport`-based tests.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ai_studio.core.errors import ProviderError, ProviderSubmitError
from ai_studio.inference.client import InferenceClient


def _client(handler) -> InferenceClient:
    return InferenceClient("http://pod:8189", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_is_ready_true_on_200() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        return httpx.Response(200, json={"ok": True})

    client = _client(handler)
    assert await client.is_ready() is True
    await client.aclose()


@pytest.mark.asyncio
async def test_is_ready_false_on_error_or_bad_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = _client(handler)
    assert await client.is_ready() is False
    await client.aclose()

    def raising_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client2 = _client(raising_handler)
    assert await client2.is_ready() is False
    await client2.aclose()


@pytest.mark.asyncio
async def test_submit_job_uploads_the_file_and_returns_a_job_id(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"fake-jpeg-bytes")
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/submit"
        body = request.content.decode("utf-8", errors="ignore")
        seen["modality"] = "image" in body
        return httpx.Response(200, json={"job_id": "job-1"})

    client = _client(handler)
    job_id = await client.submit_job("image", source, None)
    assert job_id == "job-1"
    assert seen["modality"]
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_job_raises_on_a_missing_file(tmp_path: Path) -> None:
    client = _client(lambda request: httpx.Response(200, json={"job_id": "x"}))
    with pytest.raises(ProviderSubmitError):
        await client.submit_job("image", tmp_path / "nope.jpg", None)
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_job_raises_on_a_rejected_request(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="unsupported modality")

    client = _client(handler)
    with pytest.raises(ProviderSubmitError):
        await client.submit_job("image", source, None)
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_job_raises_when_no_job_id_is_returned(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _client(handler)
    with pytest.raises(ProviderSubmitError):
        await client.submit_job("image", source, None)
    await client.aclose()


@pytest.mark.asyncio
async def test_poll_job_returns_the_raw_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/poll/job-1"
        return httpx.Response(200, json={"state": "completed", "result_text": "a cat"})

    client = _client(handler)
    payload = await client.poll_job("job-1")
    assert payload == {"state": "completed", "result_text": "a cat"}
    await client.aclose()


@pytest.mark.asyncio
async def test_poll_job_raises_on_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(handler)
    with pytest.raises(ProviderError):
        await client.poll_job("job-1")
    await client.aclose()


@pytest.mark.asyncio
async def test_cancel_job_is_best_effort() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(handler)
    await client.cancel_job("job-1")  # must not raise
    await client.aclose()


@pytest.mark.asyncio
async def test_unload_posts_to_unload_and_raises_on_failure() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200)

    client = _client(handler)
    await client.unload()
    assert calls == ["/unload"]
    await client.aclose()

    def raising_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client2 = _client(raising_handler)
    with pytest.raises(ProviderError):
        await client2.unload()
    await client2.aclose()
