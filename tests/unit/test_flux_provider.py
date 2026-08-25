"""Flux.1-dev through ComfyUI.

Mirrors the H3 provider's shape but for a still image: no frame count, so
`submit()` never attempts `length`/`fps`, and `fetch()` goes through
`media.probe_image()` (which has no fps/duration expectation) rather than
`media.probe()` (which raises on a file with no video stream).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai_studio import media
from ai_studio.comfy.client import ComfyOutput
from ai_studio.core.enums import GenMode, JobState
from ai_studio.core.image_provider_spec import ImageJob, ImageRequest
from ai_studio.providers import flux as flux_provider
from ai_studio.providers.flux import FluxComfyUIProvider, flux_capabilities

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / "workflows" / "flux_dev.json"


def test_flux_capabilities_has_no_clip_length_concept() -> None:
    caps = flux_capabilities(hourly_usd=0.74)
    assert caps.modes == frozenset({GenMode.T2I})
    assert caps.output_format == "png"
    # These are `ClipRequest`/`ProviderCapabilities`-shaped concepts that a
    # still image has no equivalent of; the type simply has no such fields.
    for absent in ("min_clip_s", "max_clip_s", "clip_duration_quantum", "native_fps"):
        assert not hasattr(caps, absent)


@pytest.fixture
def provider() -> FluxComfyUIProvider:
    return FluxComfyUIProvider(WORKFLOW, base_url="http://fake-pod:8188")


@pytest.mark.asyncio
async def test_submit_never_attempts_length_or_fps(provider: FluxComfyUIProvider) -> None:
    captured: dict[str, Any] = {}

    async def fake_queue_prompt(graph: dict[str, Any]) -> str:
        captured["graph"] = graph
        return "prompt-1"

    provider.client.queue_prompt = fake_queue_prompt  # type: ignore[method-assign]

    request = ImageRequest(
        shot_id="job1", prompt="a cat in the rain", width=1024, height=1024, seed=7, steps=20
    )
    job = await provider.submit(request)

    assert job.job_id == "prompt-1"
    assert job.shot_id == "job1"
    graph = captured["graph"]
    prompt_node = graph["4"]["inputs"]
    assert prompt_node["text"] == "a cat in the rain"
    # There is no frame-count equivalent for a still image -- confirm neither
    # key appears anywhere in the submitted graph.
    for node in graph.values():
        assert "length" not in node.get("inputs", {})
        assert "fps" not in node.get("inputs", {})


@pytest.mark.asyncio
async def test_poll_reports_completion_from_history(provider: FluxComfyUIProvider) -> None:
    async def fake_history(prompt_id: str) -> dict[str, Any]:
        return {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {"13": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}},
        }

    provider.client.history = fake_history  # type: ignore[method-assign]

    job = ImageJob(
        provider="flux", job_id="prompt-1", shot_id="job1", state=JobState.QUEUED,
        submitted_at=0.0, updated_at=0.0,
    )
    updated = await provider.poll(job)
    assert updated.state is JobState.COMPLETED
    assert updated.raw["output"]["filename"] == "out.png"


@pytest.mark.asyncio
async def test_fetch_uses_probe_image_not_probe(
    provider: FluxComfyUIProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`media.probe()` raises on a file with no video stream, which every still
    image is -- this asserts the image path never calls it."""
    job = ImageJob(
        provider="flux", job_id="prompt-1", shot_id="job1", state=JobState.COMPLETED,
        submitted_at=0.0, updated_at=1.0,
        raw={"output": {"filename": "out.png", "subfolder": "", "type": "output"}},
    )

    async def fake_download(output: ComfyOutput) -> bytes:
        assert output.filename == "out.png"
        return b"not-a-real-png-but-thats-fine-because-probe-image-is-mocked"

    def fake_probe_image(path: Path) -> media.ImageInfo:
        return media.ImageInfo(width=1024, height=1024, size_bytes=len(Path(path).read_bytes()))

    def fake_probe(*_a: Any, **_k: Any) -> None:
        raise AssertionError("fetch() must not call media.probe() for a still image")

    provider.client.download = fake_download  # type: ignore[method-assign]
    monkeypatch.setattr(flux_provider.media, "probe_image", fake_probe_image)
    monkeypatch.setattr(flux_provider.media, "probe", fake_probe)

    asset = await provider.fetch(job, tmp_path / "out.png")

    assert asset.width == 1024
    assert asset.height == 1024
    assert asset.format == "png"
    for absent in ("duration_s", "fps", "has_audio"):
        assert not hasattr(asset, absent)


@pytest.mark.asyncio
async def test_cancel_swallows_provider_errors(provider: FluxComfyUIProvider) -> None:
    from ai_studio.core.errors import ProviderError

    async def fake_interrupt() -> None:
        raise ProviderError("nothing running")

    provider.client.interrupt = fake_interrupt  # type: ignore[method-assign]

    job = ImageJob(
        provider="flux", job_id="prompt-1", shot_id="job1", state=JobState.RUNNING,
        submitted_at=0.0, updated_at=0.0,
    )
    await provider.cancel(job)  # must not raise
