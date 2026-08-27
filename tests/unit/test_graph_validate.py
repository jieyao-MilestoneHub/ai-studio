"""The turbo trap must be impossible to submit.

This is the highest-value test in the repo right now. The failure it guards
against is silent, plausible-looking, and *faster* than the correct path — so
it survives casual review and shows up as a win on a benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_studio.comfy.graph import IMAGE_REQUIRED_BINDINGS, REQUIRED_BINDINGS, Workflow
from ai_studio.comfy.validate import uses_turbo_lora, validate_graph
from ai_studio.core.errors import GraphValidationError, UnknownKeyError


def _turbo_graph(lora_node: str = "MiniMaxH3TurboLoRA", sampler: str = "MiniMaxH3TurboSampler"):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "h3_fl2va.safetensors"}},
        "2": {"class_type": lora_node, "inputs": {"lora_name": "h3_turbo_4step.safetensors", "model": ["1", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "4": {"class_type": sampler, "inputs": {"steps": 12, "model": ["2", 0], "positive": ["3", 0]}},
        "5": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "ai_studio", "images": ["4", 0]}},
    }


def _base_graph() -> dict[str, Any]:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "h3_fl2va.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "4": {"class_type": "KSampler", "inputs": {"steps": 20, "model": ["1", 0]}},
        "5": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "ai_studio", "images": ["4", 0]}},
    }


# ------------------------------------------------------------- the turbo trap


def test_correctly_wired_turbo_graph_passes() -> None:
    validate_graph(_turbo_graph(), expect_turbo=True)


def test_turbo_lora_through_the_stock_loader_is_rejected() -> None:
    """The exact mistake: LoraLoaderModelOnly driving the turbo LoRA."""
    with pytest.raises(GraphValidationError) as exc:
        validate_graph(_turbo_graph(lora_node="LoraLoaderModelOnly"))
    message = str(exc.value)
    assert "MiniMaxH3TurboLoRA" in message
    assert "comb artifacts" in message  # the message explains *why*, not just *what*


def test_turbo_graph_sampling_through_ksamplerselect_is_rejected() -> None:
    with pytest.raises(GraphValidationError, match="MiniMaxH3TurboSampler"):
        validate_graph(_turbo_graph(sampler="KSamplerSelect"))


def test_turbo_is_detected_from_the_lora_filename_not_just_the_node_class() -> None:
    """A stock loader pointed at a turbo file is the trap, however it is spelled."""
    assert uses_turbo_lora(_turbo_graph(lora_node="LoraLoader")) is True
    assert uses_turbo_lora(_base_graph()) is False


def test_expecting_turbo_but_finding_none_is_an_error() -> None:
    """A workflow that lost its LoRA would silently render at base cost."""
    with pytest.raises(GraphValidationError, match=r"1\.7x the cost"):
        validate_graph(_base_graph(), expect_turbo=True)


def test_a_plain_base_graph_is_fine_when_turbo_is_not_expected() -> None:
    validate_graph(_base_graph())


def test_a_clean_non_h3_graph_with_stock_nodes_passes_untouched() -> None:
    """A Flux graph has no MiniMaxH3* nodes and no turbo/lightning hint strings,
    so the turbo trap must not fire on it at all — including its stock
    KSamplerSelect, which is only a problem when it drives a detected turbo LoRA."""
    flux_like = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "3": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    assert uses_turbo_lora(flux_like) is False
    validate_graph(flux_like)  # must not raise


# --------------------------------------------------------- image bindings


def _image_workflow_dict() -> dict[str, Any]:
    return {
        "_ai_studio": {
            "bindings": {
                "prompt": ["2", "text"],
                "width": ["3", "width"],
                "height": ["3", "height"],
            },
        },
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "3": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }


def test_an_image_graph_with_no_length_binding_loads_with_image_required_bindings() -> None:
    wf = Workflow(_image_workflow_dict(), required_bindings=IMAGE_REQUIRED_BINDINGS)
    assert "length" not in wf.bindings


def test_the_default_required_bindings_still_reject_a_graph_missing_length() -> None:
    """Regression: adding IMAGE_REQUIRED_BINDINGS must not weaken the H3 path."""
    with pytest.raises(GraphValidationError, match="missing required bindings"):
        Workflow(_image_workflow_dict())


def test_image_required_bindings_have_no_length_requirement() -> None:
    assert "length" in REQUIRED_BINDINGS
    assert "length" not in IMAGE_REQUIRED_BINDINGS


# -------------------------------------------------------------- structure


def test_dangling_link_is_rejected() -> None:
    graph = _base_graph()
    graph["4"]["inputs"]["model"] = ["99", 0]
    with pytest.raises(GraphValidationError, match="does not exist"):
        validate_graph(graph)


def test_empty_graph_is_rejected() -> None:
    with pytest.raises(GraphValidationError, match="no nodes"):
        validate_graph({})


# --------------------------------------------------------------- bindings


def _workflow_dict() -> dict[str, Any]:
    graph = _turbo_graph()
    graph["_ai_studio"] = {
        "expect_turbo": True,
        "bindings": {
            "prompt": ["3", "text"],
            "width": ["5", "filename_prefix"],
            "height": ["5", "filename_prefix"],
            "length": ["4", "steps"],
        },
    }
    return graph


def test_bindings_inject_and_do_not_mutate_the_source() -> None:
    wf = Workflow(_workflow_dict())
    out = wf.with_values({"prompt": "a baker at dawn"})
    assert out["3"]["inputs"]["text"] == "a baker at dawn"
    assert wf.graph["3"]["inputs"]["text"] == ""  # original untouched


def test_metadata_block_is_stripped_before_submission() -> None:
    out = Workflow(_workflow_dict()).with_values({})
    assert "_ai_studio" not in out


def test_a_value_with_no_binding_raises_rather_than_being_dropped() -> None:
    """A silently ignored seed is a run you cannot reproduce and will not notice."""
    with pytest.raises(UnknownKeyError):
        Workflow(_workflow_dict()).with_values({"seed": 42})


def test_binding_to_a_missing_node_fails_at_load_not_at_submit() -> None:
    graph = _workflow_dict()
    graph["_ai_studio"]["bindings"]["prompt"] = ["404", "text"]
    with pytest.raises(GraphValidationError, match="does not exist"):
        Workflow(graph)


def test_missing_required_bindings_are_reported() -> None:
    graph = _workflow_dict()
    del graph["_ai_studio"]["bindings"]["width"]
    with pytest.raises(GraphValidationError, match="missing required bindings"):
        Workflow(graph)


def test_load_reports_bad_json_with_the_path(tmp_path: Path) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(GraphValidationError, match="invalid JSON"):
        Workflow.load(bad)


def test_load_roundtrips_a_real_file(tmp_path: Path) -> None:
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(_workflow_dict()), encoding="utf-8")
    wf = Workflow.load(path)
    assert wf.expect_turbo is True
    assert "prompt" in wf.bindings


# ------------------------------------------------------ the Flux LoRA wiring

# `UNETLoader` feeds TWO consumers in the Flux graph: BasicGuider ("8") and
# BasicScheduler ("10"). The LoRA node ("14") goes between, and BOTH have to be
# rewired to it. Rewire only "10" and the LoRA has **no effect at all and
# raises no error** — the file loads, the graph validates, the picture comes
# out fine, and the adapter is simply not in the path that guides sampling.
#
# That is why this is a structural assertion and not a comment. The live check
# (same seed, lora_strength 1.0 vs 0.0, two images that must differ) costs
# GPU-seconds and only happens once, in Phase 7.1.

FLUX_WORKFLOW = Path("workflows/flux_dev.json")
LORA_NODE = "14"


def _flux() -> Workflow:
    return Workflow.load(FLUX_WORKFLOW, required_bindings=IMAGE_REQUIRED_BINDINGS)


def test_the_flux_workflow_loads_and_validates_with_the_lora_in_it() -> None:
    """`STOCK_LORA_NODES` contains `LoraLoaderModelOnly`, which reads like a
    tripwire for this graph. It is not: the H3 ban is scoped by
    `uses_turbo_lora`, and an ordinary LoRA is fine. Asserted rather than left
    to a comment in `validate.py` that says so."""
    workflow = _flux()

    validate_graph(workflow.graph)
    assert workflow.graph[LORA_NODE]["class_type"] == "LoraLoaderModelOnly"
    assert uses_turbo_lora(workflow.graph) is False


def test_every_consumer_of_the_base_model_goes_through_the_lora() -> None:
    """The one that catches a half-rewire. Any node still reading ["1", 0] is
    a path the LoRA does not affect."""
    graph = _flux().graph
    unet = next(n for n, node in graph.items() if node["class_type"] == "UNETLoader")

    direct = [
        node_id
        for node_id, node in graph.items()
        if node_id != LORA_NODE
        for value in node.get("inputs", {}).values()
        if isinstance(value, list) and len(value) == 2 and str(value[0]) == unet
    ]

    assert direct == [], f"nodes still bypassing the LoRA: {direct}"
    assert graph["8"]["inputs"]["model"] == [LORA_NODE, 0]
    assert graph["10"]["inputs"]["model"] == [LORA_NODE, 0]
    assert graph[LORA_NODE]["inputs"]["model"] == [unet, 0]


def test_the_lora_filename_is_the_one_the_pod_actually_writes() -> None:
    """`hf download` keeps the remote name (`lora.safetensors`);
    `deploy/pod_setup.sh` renames it. These two strings have to be the same
    string or ComfyUI loads nothing and says so only in a log."""
    lora_name = _flux().graph[LORA_NODE]["inputs"]["lora_name"]
    setup = Path("deploy/pod_setup.sh").read_text(encoding="utf-8")

    assert lora_name == "flux_nsfw_uncensored_v1.safetensors"
    assert lora_name in setup, "pod_setup.sh does not produce this filename"
    assert "Heartsync/Flux-NSFW-uncensored" in setup


def test_lora_strength_is_bindable_so_the_ab_test_is_a_parameter() -> None:
    """Phase 7.1 renders the same seed at 1.0 and 0.0. If those two images are
    identical, the wiring above is wrong — and it fails silently, so the check
    has to be cheap enough to actually run."""
    workflow = _flux()

    assert "lora_strength" in workflow.bindings
    off = workflow.with_values(
        {"prompt": "a fox", "width": 1024, "height": 1024, "lora_strength": 0.0}
    )
    on = workflow.with_values(
        {"prompt": "a fox", "width": 1024, "height": 1024, "lora_strength": 1.0}
    )

    assert off[LORA_NODE]["inputs"]["strength_model"] == 0.0
    assert on[LORA_NODE]["inputs"]["strength_model"] == 1.0


# ------------------------------------------------- the /短劇 face-repair sibling

FACE_WORKFLOW = Path("workflows/flux_dev_i2i_face.json")


def test_the_face_sibling_is_the_i2i_graph_plus_a_detailer_before_save() -> None:
    """`Workflow.sibling` finds it by name; it must load with the i2i bindings
    and differ from `flux_dev_i2i.json` only downstream of the decode."""
    face = Workflow.load(
        FACE_WORKFLOW, required_bindings=IMAGE_REQUIRED_BINDINGS | {"source_image", "denoise"}
    )
    i2i = Workflow.load(
        Path("workflows/flux_dev_i2i.json"),
        required_bindings=IMAGE_REQUIRED_BINDINGS | {"source_image", "denoise"},
    )
    validate_graph(face.graph)
    assert uses_turbo_lora(face.graph) is False

    for node_id, node in i2i.graph.items():
        if node_id == "13":  # SaveImage now reads the detailer's output
            continue
        assert face.graph[node_id] == node, f"node {node_id} drifted from the i2i graph"
    assert face.graph["13"]["inputs"]["images"] == ["19", 0]
    assert face.graph["19"]["class_type"] == "FaceDetailer"
    assert face.graph["19"]["inputs"]["image"] == ["12", 0], "detailer works on the decoded i2i picture"
    assert face.graph["19"]["inputs"]["model"] == [LORA_NODE, 0], "the detail pass goes through the LoRA too"
    assert face.graph["19"]["inputs"]["bbox_detector"] == ["18", 0]
    assert face.graph["18"]["class_type"] == "UltralyticsDetectorProvider"
    assert face.graph["18"]["inputs"]["model_name"] == "bbox/face_yolov8m.pt"


def test_every_consumer_of_the_base_model_goes_through_the_lora_in_the_face_sibling() -> None:
    graph = Workflow.load(
        FACE_WORKFLOW, required_bindings=IMAGE_REQUIRED_BINDINGS | {"source_image", "denoise"}
    ).graph
    unet = next(n for n, node in graph.items() if node["class_type"] == "UNETLoader")
    direct = [
        node_id
        for node_id, node in graph.items()
        if node_id != LORA_NODE
        for value in node.get("inputs", {}).values()
        if isinstance(value, list) and len(value) == 2 and str(value[0]) == unet
    ]
    assert direct == [], f"nodes still bypassing the LoRA: {direct}"
