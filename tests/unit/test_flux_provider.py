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
from ai_studio.core.errors import ProviderSubmitError
from ai_studio.core.image_provider_spec import ImageJob, ImageRequest
from ai_studio.providers import flux as flux_provider
from ai_studio.providers.flux import FluxComfyUIProvider, flux_capabilities

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / "workflows" / "flux_dev.json"


def test_flux_capabilities_has_no_clip_length_concept() -> None:
    caps = flux_capabilities(hourly_usd=0.74)
    assert caps.modes == frozenset({GenMode.T2I, GenMode.I2I})
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
async def test_steps_are_bound_from_the_provider_when_the_request_leaves_them_unset() -> None:
    """The queue path never sets `ImageRequest.steps`. Before this, that left
    the `steps` binding unexercised and the JSON's own value silently in
    charge, so DEFAULT_STEPS was a number nothing read."""
    captured: dict[str, Any] = {}

    async def fake_queue_prompt(graph: dict[str, Any]) -> str:
        captured["graph"] = graph
        return "prompt-1"

    provider = FluxComfyUIProvider(WORKFLOW, base_url="http://fake-pod:8188", steps=12)
    provider.client.queue_prompt = fake_queue_prompt  # type: ignore[method-assign]

    await provider.submit(ImageRequest(shot_id="job1", prompt="a cat", width=1024, height=1024))
    assert captured["graph"]["10"]["inputs"]["steps"] == 12

    await provider.submit(
        ImageRequest(shot_id="job2", prompt="a cat", width=1024, height=1024, steps=20)
    )
    assert captured["graph"]["10"]["inputs"]["steps"] == 20


def test_flux_capabilities_refuses_a_steps_argument() -> None:
    """It used to accept `steps` and ignore it. Accepted-and-ignored is the
    silent kind of wrong this codebase refuses; now it is a TypeError."""
    with pytest.raises(TypeError):
        flux_capabilities(hourly_usd=0.74, steps=10)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_poll_reports_completion_from_history(provider: FluxComfyUIProvider) -> None:
    async def fake_history(prompt_id: str) -> dict[str, Any]:
        return {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {"13": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}},
        }

    async def fake_system_stats() -> dict[str, Any]:
        return {"devices": [{"type": "cuda", "vram_total": 25_000_000_000, "vram_free": 5_000_000_000}]}

    provider.client.history = fake_history  # type: ignore[method-assign]
    provider.client.system_stats = fake_system_stats  # type: ignore[method-assign]

    job = ImageJob(
        provider="flux", job_id="prompt-1", shot_id="job1", state=JobState.QUEUED,
        submitted_at=0.0, updated_at=0.0,
    )
    updated = await provider.poll(job)
    assert updated.state is JobState.COMPLETED
    assert updated.raw["output"]["filename"] == "out.png"
    # `poll_job` samples VRAM via /system_stats on every poll (comfy/jobs.py).
    assert updated.raw["peak_vram_bytes"] == 20_000_000_000


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


# ------------------------------------------------------------ image-to-image


def test_the_i2i_sibling_is_found_next_to_the_base_workflow(provider: FluxComfyUIProvider) -> None:
    assert provider._i2i_workflow is not None
    for binding in ("source_image", "denoise", "width", "height", "lora_strength"):
        assert binding in provider._i2i_workflow.bindings
    assert provider.capabilities().supports(GenMode.I2I)


def test_the_i2i_sibling_keeps_every_model_consumer_behind_the_lora(
    provider: FluxComfyUIProvider,
) -> None:
    """The same half-rewire trap `test_graph_validate` guards on the base
    graph. A sibling copied by hand is exactly where it would come back."""
    graph = provider._i2i_workflow.graph  # type: ignore[union-attr]
    unet = next(n for n, node in graph.items() if node["class_type"] == "UNETLoader")
    lora = next(n for n, node in graph.items() if node["class_type"] == "LoraLoaderModelOnly")
    direct = [
        node_id
        for node_id, node in graph.items()
        if node_id != lora
        for value in node.get("inputs", {}).values()
        if isinstance(value, list) and len(value) == 2 and str(value[0]) == unet
    ]
    assert direct == []
    assert not any(node["class_type"] == "EmptySD3LatentImage" for node in graph.values())
    assert any(node["class_type"] == "VAEEncode" for node in graph.values())


@pytest.mark.asyncio
async def test_submit_with_a_source_image_uploads_and_switches_workflow(
    provider: FluxComfyUIProvider, tmp_path: Path
) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"jpeg-bytes")
    captured: dict[str, Any] = {}

    async def fake_upload(data: bytes, filename: str) -> str:
        captured["upload"] = (data, filename)
        return "uploaded-photo.jpg"

    async def fake_queue_prompt(graph: dict[str, Any]) -> str:
        captured["graph"] = graph
        return "prompt-i2i"

    provider.client.upload_image = fake_upload  # type: ignore[method-assign]
    provider.client.queue_prompt = fake_queue_prompt  # type: ignore[method-assign]

    await provider.submit(
        ImageRequest(
            shot_id="job9", mode=GenMode.I2I, prompt="as an oil painting",
            width=1024, height=1024, seed=3, source_image_path=str(source),
        )
    )

    assert captured["upload"] == (b"jpeg-bytes", "photo.jpg")
    graph = captured["graph"]
    load = next(n for n in graph.values() if n["class_type"] == "LoadImage")
    assert load["inputs"]["image"] == "uploaded-photo.jpg"
    scheduler = next(n for n in graph.values() if n["class_type"] == "BasicScheduler")
    assert scheduler["inputs"]["denoise"] == flux_provider.DEFAULT_I2I_DENOISE
    assert scheduler["inputs"]["steps"] == flux_provider.DEFAULT_STEPS
    lora = next(n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert lora["inputs"]["strength_model"] == flux_provider.DEFAULT_LORA_STRENGTH


@pytest.mark.asyncio
async def test_submit_without_a_source_image_never_touches_upload(
    provider: FluxComfyUIProvider,
) -> None:
    async def fake_upload(data: bytes, filename: str) -> str:
        raise AssertionError("upload must not be called for text-to-image")

    async def fake_queue_prompt(graph: dict[str, Any]) -> str:
        return "prompt-t2i"

    provider.client.upload_image = fake_upload  # type: ignore[method-assign]
    provider.client.queue_prompt = fake_queue_prompt  # type: ignore[method-assign]
    await provider.submit(ImageRequest(shot_id="j", prompt="a cat", width=1024, height=1024))


@pytest.mark.asyncio
async def test_a_source_image_with_no_i2i_sibling_raises_clearly(tmp_path: Path) -> None:
    lonely = tmp_path / "custom.json"
    lonely.write_text(WORKFLOW.read_text(encoding="utf-8"), encoding="utf-8")
    provider = FluxComfyUIProvider(lonely, base_url="http://fake-pod:8188")
    assert provider._i2i_workflow is None

    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")
    with pytest.raises(ProviderSubmitError, match="no image-to-image sibling"):
        await provider.submit(
            ImageRequest(shot_id="j", prompt="p", width=1, height=1, source_image_path=str(source))
        )


@pytest.mark.asyncio
async def test_a_missing_source_file_raises_a_provider_error(
    provider: FluxComfyUIProvider,
) -> None:
    with pytest.raises(ProviderSubmitError, match="could not read"):
        await provider.submit(
            ImageRequest(
                shot_id="j", prompt="p", width=1, height=1,
                source_image_path="/does/not/exist.jpg",
            )
        )
