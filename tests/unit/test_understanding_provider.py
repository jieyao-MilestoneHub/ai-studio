"""The understanding provider quartet: `UnderstandingCapabilities` per
modality, the real `UnderstandingProvider` (HTTP mocked, same style as
`test_inference_client.py`), and the offline `StubUnderstandingProvider`
that CI runs against with no GPU.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ai_studio.core.enums import JobState, MediaKind
from ai_studio.core.errors import ProviderJobFailed
from ai_studio.core.understanding_spec import UnderstandingJob, UnderstandingRequest
from ai_studio.inference.client import InferenceClient
from ai_studio.providers.stub import StubUnderstandingProvider
from ai_studio.providers.understanding import UnderstandingProvider, understanding_capabilities

# --------------------------------------------------------- capabilities


def test_understanding_capabilities_reflect_the_model_per_modality() -> None:
    image_caps = understanding_capabilities(MediaKind.IMAGE_UNDERSTAND)
    audio_caps = understanding_capabilities(MediaKind.AUDIO_UNDERSTAND)
    video_caps = understanding_capabilities(MediaKind.VIDEO_UNDERSTAND)

    assert "moondream" in image_caps.model_id
    assert image_caps.accepts_prompt is True
    assert image_caps.max_input_seconds is None

    assert "Qwen3-Omni" in audio_caps.model_id
    assert audio_caps.accepts_prompt is False, "Qwen3-Omni-Captioner rejects a text prompt"
    assert audio_caps.max_input_seconds == 30.0

    assert "Tarsier2" in video_caps.model_id
    assert video_caps.accepts_prompt is True


def test_check_prompt_raises_loudly_rather_than_dropping_it() -> None:
    caps = understanding_capabilities(MediaKind.AUDIO_UNDERSTAND)
    with pytest.raises(ValueError):
        caps.check_prompt("what mood is this?")
    caps.check_prompt(None)  # no prompt is always fine
    understanding_capabilities(MediaKind.IMAGE_UNDERSTAND).check_prompt("who is this?")


# ------------------------------------------------------------ real provider


def _provider(modality: MediaKind, handler) -> UnderstandingProvider:
    provider = UnderstandingProvider(modality=modality, base_url="http://pod:8189")
    provider.client = InferenceClient("http://pod:8189", transport=httpx.MockTransport(handler))
    return provider


@pytest.mark.asyncio
async def test_submit_poll_fetch_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            return httpx.Response(200, json={"job_id": "job-1"})
        if request.url.path == "/poll/job-1":
            return httpx.Response(200, json={"state": "completed", "result_text": "a cat on a mat"})
        return httpx.Response(404)

    provider = _provider(MediaKind.IMAGE_UNDERSTAND, handler)
    request = UnderstandingRequest(
        shot_id="job1", modality=MediaKind.IMAGE_UNDERSTAND, input_media_path=str(source)
    )

    job = await provider.submit(request)
    assert job.state is JobState.QUEUED
    job = await provider.poll(job)
    assert job.state is JobState.COMPLETED

    asset = await provider.fetch(job)
    assert asset.result_text == "a cat on a mat"
    assert asset.modality is MediaKind.IMAGE_UNDERSTAND
    await provider.aclose()


@pytest.mark.asyncio
async def test_a_failed_backend_job_polls_to_failed(tmp_path: Path) -> None:
    source = tmp_path / "clip.m4a"
    source.write_bytes(b"x")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/submit":
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={"state": "failed", "error": "OOM during model load"})

    provider = _provider(MediaKind.AUDIO_UNDERSTAND, handler)
    request = UnderstandingRequest(
        shot_id="job1", modality=MediaKind.AUDIO_UNDERSTAND, input_media_path=str(source)
    )
    job = await provider.submit(request)
    job = await provider.poll(job)
    assert job.state is JobState.FAILED
    assert job.error == "OOM during model load"
    await provider.aclose()


@pytest.mark.asyncio
async def test_submit_rejects_a_prompt_the_backend_does_not_accept() -> None:
    """Qwen3-Omni-Captioner takes no text prompt -- a caller that supplies one
    must be told loudly, before anything is uploaded, not silently ignored."""
    provider = UnderstandingProvider(modality=MediaKind.AUDIO_UNDERSTAND, base_url="http://pod:8189")
    request = UnderstandingRequest(
        shot_id="job1", modality=MediaKind.AUDIO_UNDERSTAND,
        input_media_path="/incoming/x.m4a", prompt="describe the mood",
    )
    with pytest.raises(ValueError):
        await provider.submit(request)
    await provider.aclose()


@pytest.mark.asyncio
async def test_evict_calls_unload() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200)

    provider = _provider(MediaKind.VIDEO_UNDERSTAND, handler)
    await provider.evict()
    assert calls == ["/unload"]
    await provider.aclose()


# ---------------------------------------------------------------- stub

_KIND_TO_MODALITY = {
    "image": MediaKind.IMAGE_UNDERSTAND,
    "audio": MediaKind.AUDIO_UNDERSTAND,
    "video": MediaKind.VIDEO_UNDERSTAND,
}


@pytest.mark.asyncio
async def test_stub_roundtrip_is_deterministic_and_names_the_file(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x" * 100)

    provider = StubUnderstandingProvider(modality=MediaKind.IMAGE_UNDERSTAND)
    request = UnderstandingRequest(
        shot_id="s1", modality=MediaKind.IMAGE_UNDERSTAND, input_media_path=str(source)
    )

    job = await provider.submit(request)
    assert job.state is JobState.COMPLETED
    job = await provider.poll(job)  # a no-op, but must not raise
    asset = await provider.fetch(job)

    assert "photo.jpg" in asset.result_text
    assert "100 bytes" in asset.result_text
    assert asset.cost_usd == 0.0
    await provider.aclose()


@pytest.mark.asyncio
async def test_stub_also_rejects_a_prompt_for_audio(tmp_path: Path) -> None:
    source = tmp_path / "clip.m4a"
    source.write_bytes(b"x")
    provider = StubUnderstandingProvider(modality=MediaKind.AUDIO_UNDERSTAND)
    request = UnderstandingRequest(
        shot_id="s1", modality=MediaKind.AUDIO_UNDERSTAND,
        input_media_path=str(source), prompt="what mood?",
    )
    with pytest.raises(ValueError):
        await provider.submit(request)


@pytest.mark.asyncio
async def test_stub_fetch_without_a_prior_submit_fails_loudly() -> None:
    provider = StubUnderstandingProvider(modality=MediaKind.VIDEO_UNDERSTAND)
    job = UnderstandingJob(
        provider="stub", job_id="nope", shot_id="s", state=JobState.COMPLETED,
        submitted_at=0.0, updated_at=0.0,
    )
    with pytest.raises(ProviderJobFailed):
        await provider.fetch(job)
