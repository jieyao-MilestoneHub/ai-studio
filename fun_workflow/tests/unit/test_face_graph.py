"""The /短劇 face-repair graph: ai-studio's Flux i2i graph plus an
Impact-Pack FaceDetailer, shipped by this package and handed to
`FluxComfyUIProvider(i2i_face_workflow=...)` by the composition root."""

from __future__ import annotations

from ai_studio import paths
from ai_studio.comfy.graph import IMAGE_REQUIRED_BINDINGS, Workflow
from ai_studio.comfy.validate import uses_turbo_lora, validate_graph

from fun_workflow import paths as fun_paths

FACE_WORKFLOW = fun_paths.workflow("flux_dev_i2i_face.json")
LORA_NODE = "14"


def test_the_face_sibling_is_the_i2i_graph_plus_a_detailer_before_save() -> None:
    """Passed to `FluxComfyUIProvider(i2i_face_workflow=...)` by the caller
    that wants face repair; it must load with the i2i bindings and differ
    from `flux_dev_i2i.json` only downstream of the decode."""
    face = Workflow.load(
        FACE_WORKFLOW, required_bindings=IMAGE_REQUIRED_BINDINGS | {"source_image", "denoise"}
    )
    i2i = Workflow.load(
        paths.workflow("flux_dev_i2i.json"),
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
