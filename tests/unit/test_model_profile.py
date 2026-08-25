"""Model profiles, and the conformance between them and the shipped graphs.

Before this file, **not one numeric value in either workflow was pinned by any
test.** `validate_graph` checks wiring and class names; `test_ui_workflow.py`
pins only the two values that differ between the H3 pair. Resolution, `length`,
fps, steps, LoRA strength, the scheduler, the weight filenames — all of them
could be changed in either JSON and the whole suite stayed green.

That mattered because the same constant lived in up to four unlinked places.
`124` was a literal in both workflows, `124 / 24` in `convert_worker`, and a
`max(frames, 124)` floor in `drain` — while the rule that produced it (`17k+5`)
existed only in a docstring, and `clip_duration_quantum`, the field designed to
carry exactly that rule, was set to `None`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_studio.comfy.graph import IMAGE_REQUIRED_BINDINGS, Workflow
from ai_studio.core.errors import UnknownKeyError
from ai_studio.core.model_profile import (
    FLUX_1_DEV,
    MINIMAX_H3,
    PROFILES,
    FrameGrid,
    get_profile,
)

REPO = Path(__file__).resolve().parents[2]
H3_WORKFLOWS = ("h3_fl2va_turbo.json", "h3_fl2va_turbo_fp8.json")


def _load(name: str) -> Workflow:
    path = REPO / "workflows" / name
    if name.startswith("flux"):
        return Workflow.load(path, required_bindings=IMAGE_REQUIRED_BINDINGS)
    return Workflow.load(path)


def _bound(workflow: Workflow, name: str) -> object:
    """The literal currently sitting in the graph at a named binding."""
    node_id, key = workflow.bindings[name]
    return workflow.graph[node_id]["inputs"][key]


# ------------------------------------------------------------------ the grid


def test_the_legal_frame_counts_are_the_ones_comfyui_accepts() -> None:
    """`17k+5`, read out of ComfyUI's own `nodes_minimax_h3.py`."""
    grid = MINIMAX_H3.frame_grid
    assert grid is not None
    assert [n for n in range(1, 150) if grid.is_valid(n)] == [
        5, 22, 39, 56, 73, 90, 107, 124, 141
    ]


def test_the_cli_default_of_five_seconds_is_not_a_legal_length() -> None:
    """The bug this profile closes. `--seconds 5.0` at 24fps is 120 frames,
    which is off-grid; ComfyUI would snap it up to 124 silently, so the clip
    was never the length anyone asked for."""
    grid = MINIMAX_H3.frame_grid
    assert grid is not None

    assert grid.is_valid(120) is False
    assert grid.snap(120) == 124
    assert grid.neighbours(120) == (107, 124)


def test_snapping_goes_up_because_that_is_what_comfyui_does() -> None:
    """A caller that snaps deliberately must land where the pod would have."""
    grid = MINIMAX_H3.frame_grid
    assert grid is not None

    for frames in (108, 115, 123):
        assert grid.snap(frames) == 124
    assert grid.snap(124) == 124, "an already-legal length is left alone"


def test_one_two_four_is_the_loras_floor_not_the_models() -> None:
    """This project's docs called 124 "the validated floor". ComfyUI's own
    minimum is 5; 124 is the turbo LoRA's trained lower bound. Recorded
    separately so the two can never be conflated again."""
    grid = MINIMAX_H3.frame_grid
    assert grid is not None

    assert grid.minimum == 5
    assert grid.recommended_minimum == 124
    assert grid.is_valid(5) is True, "the model accepts 5; only the adapter wants 124"


def test_a_length_past_the_maximum_raises_rather_than_wrapping() -> None:
    grid = FrameGrid(base=5, step=17, minimum=5, maximum=124)
    with pytest.raises(ValueError, match="past the 124 maximum"):
        grid.snap(125)


def test_the_shortest_useful_duration_is_the_one_everyone_asks_for() -> None:
    """`124` used to be spelled out in four places. Now there is one."""
    assert MINIMAX_H3.shortest_useful_duration_s == pytest.approx(124 / 24)
    assert MINIMAX_H3.frames_for(MINIMAX_H3.shortest_useful_duration_s) == 124


@pytest.mark.parametrize("seconds", [1.0, 3.3, 5.0, 5.1667, 7.5, 12.0])
def test_frames_for_always_returns_a_legal_length(seconds: float) -> None:
    grid = MINIMAX_H3.frame_grid
    assert grid is not None
    assert grid.is_valid(MINIMAX_H3.frames_for(seconds))


# --------------------------------------------------------------- canvases


def test_864x480_is_not_a_native_canvas() -> None:
    """It is legal — both dimensions are multiples of 32 — but H3's native
    short edge is 768, not 480. Stated as native in this repo's docs for a
    while; a quality question about it is not a bug report."""
    canvas = MINIMAX_H3.canvas(864, 480)

    assert canvas is not None
    assert canvas.native is False
    assert MINIMAX_H3.native_canvas.label == "1344x768"


def test_an_unknown_canvas_raises_instead_of_borrowing_a_latency() -> None:
    """The old `MEASURED_LATENCY_S.get((w, h), 300.0)` quietly handed an
    off-table canvas 1344x768's timing, so its cost estimate and its job
    timeout were both someone else's numbers."""
    with pytest.raises(UnknownKeyError, match="1024x576"):
        MINIMAX_H3.require_canvas(1024, 576)


def test_every_canvas_is_a_legal_grid_size() -> None:
    for profile in PROFILES.values():
        for canvas in profile.canvases:
            assert canvas.width % profile.dimension_multiple == 0, canvas.label
            assert canvas.height % profile.dimension_multiple == 0, canvas.label


def test_no_canvas_exceeds_the_models_pixel_ceiling() -> None:
    assert MINIMAX_H3.max_pixels is not None
    for canvas in MINIMAX_H3.canvases:
        assert canvas.pixels <= MINIMAX_H3.max_pixels, canvas.label


def test_an_unknown_profile_raises() -> None:
    with pytest.raises(UnknownKeyError, match="model profile"):
        get_profile("stable-diffusion-1.5")


# ------------------------------------------- the graphs agree with the profile

# The load-bearing tests in this file. A workflow whose literals drift from the
# profile is a workflow that generates something other than what the rest of the
# code believes it generates — and nothing else in the repo would notice.


@pytest.mark.parametrize("name", H3_WORKFLOWS)
def test_the_h3_graphs_ask_for_a_length_the_model_can_produce(name: str) -> None:
    grid = MINIMAX_H3.frame_grid
    assert grid is not None
    length = _bound(_load(name), "length")

    assert isinstance(length, int)
    assert grid.is_valid(length), f"{name} asks for {length} frames, which is off-grid"


@pytest.mark.parametrize("name", H3_WORKFLOWS)
def test_the_h3_graphs_use_a_canvas_the_profile_knows(name: str) -> None:
    workflow = _load(name)
    width, height = _bound(workflow, "width"), _bound(workflow, "height")

    assert isinstance(width, int) and isinstance(height, int)
    canvas = MINIMAX_H3.require_canvas(width, height)
    assert canvas.label == f"{width}x{height}"


@pytest.mark.parametrize("name", H3_WORKFLOWS)
def test_the_h3_graphs_render_at_the_profiles_fps(name: str) -> None:
    """fps is what turns `length` into a duration. If the graph and the profile
    disagree, every duration the pipeline computes is wrong."""
    assert _bound(_load(name), "fps") == MINIMAX_H3.fps


@pytest.mark.parametrize("name", H3_WORKFLOWS)
def test_the_h3_graphs_use_the_profiles_step_count(name: str) -> None:
    assert _bound(_load(name), "steps") == MINIMAX_H3.default_steps


@pytest.mark.parametrize("name", H3_WORKFLOWS)
def test_the_h3_graphs_load_the_lora_the_profile_names(name: str) -> None:
    graph = _load(name).graph
    spec = MINIMAX_H3.lora
    assert spec is not None

    loaders = [n for n in graph.values() if n["class_type"] == "MiniMaxH3TurboLoRA"]
    assert len(loaders) == 1, f"{name} has {len(loaders)} turbo LoRA nodes"
    assert loaders[0]["inputs"]["lora_name"] == spec.filename
    assert loaders[0]["inputs"]["strength"] == spec.strength


@pytest.mark.parametrize("name", H3_WORKFLOWS)
def test_the_h3_graphs_load_a_weight_file_the_profile_knows(name: str) -> None:
    """Quantisation is a property of the card, not a preference — so the DiT
    filename has to be one the profile can name, not whatever was pasted in."""
    graph = _load(name).graph
    loaders = [n for n in graph.values() if n["class_type"] == "UNETLoader"]

    assert len(loaders) == 1
    assert loaders[0]["inputs"]["unet_name"] in MINIMAX_H3.weights.values()


def test_the_two_h3_graphs_differ_only_in_quantisation_and_lora_mode() -> None:
    """The pair exists to serve a 24GB and a 48GB rung. Any *other* difference
    between them is a divergence nobody decided on."""
    low, high = (json.loads((REPO / "workflows" / n).read_text(encoding="utf-8"))
                 for n in H3_WORKFLOWS)

    differing = {
        node_id for node_id in low
        if not node_id.startswith("_") and low[node_id] != high.get(node_id)
    }
    assert differing == {"127", "134"}, (
        f"the two H3 graphs differ at {sorted(differing)}; expected only the "
        "UNETLoader (quantisation) and the turbo LoRA (low_vram)"
    )


def test_the_flux_graph_uses_the_profiles_canvas_and_lora() -> None:
    workflow = _load("flux_dev.json")
    spec = FLUX_1_DEV.lora
    assert spec is not None

    FLUX_1_DEV.require_canvas(_bound(workflow, "width"), _bound(workflow, "height"))
    loaders = [n for n in workflow.graph.values()
               if n["class_type"] == "LoraLoaderModelOnly"]
    assert len(loaders) == 1
    assert loaders[0]["inputs"]["lora_name"] == spec.filename


def test_the_flux_graph_uses_the_profiles_step_count() -> None:
    assert _bound(_load("flux_dev.json"), "steps") == FLUX_1_DEV.default_steps


# ----------------------------------------- capabilities come from the profile


def test_h3_capabilities_are_derived_not_retyped() -> None:
    from ai_studio.providers.comfyui import h3_capabilities

    caps = h3_capabilities()
    grid = MINIMAX_H3.frame_grid
    assert grid is not None and MINIMAX_H3.fps is not None

    assert caps.native_fps == MINIMAX_H3.fps
    assert caps.has_native_audio is MINIMAX_H3.has_native_audio
    assert caps.max_prompt_chars == MINIMAX_H3.max_prompt_chars
    assert caps.clip_duration_quantum == pytest.approx(grid.step / MINIMAX_H3.fps)


def test_the_duration_quantum_is_no_longer_switched_off() -> None:
    """It was `None`, which said "this model accepts any duration". It does
    not, and that claim is what let 120 frames through."""
    from ai_studio.providers.comfyui import h3_capabilities

    assert h3_capabilities().clip_duration_quantum is not None


def test_capabilities_for_an_unknown_canvas_raise() -> None:
    from ai_studio.providers.comfyui import h3_capabilities

    with pytest.raises(UnknownKeyError):
        h3_capabilities(1024, 576)


# ------------------------------------------------- the guard at the boundary

# `ClipRequest` carries `duration_s` and `fps`, never frames, so `submit` is
# where a provider-agnostic request becomes an H3 submission. That makes it the
# only place that can catch a length H3 will quietly reinterpret — and it must
# refuse rather than snap, because a program that submitted 120 frames did not
# mean 124.


class _RecordingClient:
    def __init__(self) -> None:
        self.submitted: list[dict] = []

    async def queue_prompt(self, graph: dict) -> str:
        self.submitted.append(graph)
        return "prompt-1"


def _provider(tmp_path: Path):
    from ai_studio.providers.comfyui import ComfyUIProvider

    provider = ComfyUIProvider(REPO / "workflows" / "h3_fl2va_turbo.json")
    provider.client = _RecordingClient()  # type: ignore[assignment]
    return provider


def _request(**kw):
    from ai_studio.core.enums import GenMode
    from ai_studio.core.provider_spec import ClipRequest

    base = dict(
        shot_id="s1", mode=GenMode.T2V, prompt="a cat",
        width=864, height=480, duration_s=124 / 24, fps=24,
    )
    base.update(kw)
    return ClipRequest(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_legal_request_is_submitted_with_the_frame_count_it_implies(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)

    await provider.submit(_request())

    node_id, key = provider.workflow.bindings["length"]
    assert provider.client.submitted[0][node_id]["inputs"][key] == 124


@pytest.mark.asyncio
async def test_an_off_grid_duration_is_refused_before_any_gpu_second(
    tmp_path: Path,
) -> None:
    """5.0s is 120 frames. Nothing on the pod would have complained."""
    from ai_studio.core.errors import ProviderSubmitError

    provider = _provider(tmp_path)

    with pytest.raises(ProviderSubmitError, match="120 frames"):
        await provider.submit(_request(duration_s=5.0))

    assert provider.client.submitted == [], "it reached ComfyUI anyway"


@pytest.mark.asyncio
async def test_an_unknown_canvas_is_refused_before_any_gpu_second(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)

    with pytest.raises(UnknownKeyError, match="1024x576"):
        await provider.submit(_request(width=1024, height=576))

    assert provider.client.submitted == []
