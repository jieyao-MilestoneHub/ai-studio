"""UI -> API workflow conversion.

The regression that motivated most of these: treating INT/FLOAT as link-only
types silently swallowed every numeric widget (seed, steps, strength, fps),
producing a graph that ComfyUI accepts and runs with default values.
"""

from __future__ import annotations

from typing import Any

import pytest

from videogen.comfy.ui_workflow import (
    prune_unreachable,
    set_literal,
    ui_to_api,
)

OBJECT_INFO: dict[str, Any] = {
    "UNETLoader": {"required": [["unet_name", [["a.safetensors"]]], ["weight_dtype", [["default"]]]], "optional": []},
    "RandomNoise": {"required": [["noise_seed", ["INT", {}]]], "optional": []},
    "BasicScheduler": {
        "required": [["model", ["MODEL"]], ["scheduler", [["simple"]]], ["steps", ["INT", {}]], ["denoise", ["FLOAT", {}]]],
        "optional": [],
    },
    "TurboLoRA": {
        "required": [["model", ["MODEL"]], ["lora_name", [["l.safetensors"]]], ["strength", ["FLOAT", {}]], ["low_vram", ["BOOLEAN", {}]]],
        "optional": [],
    },
    "MathNode": {"required": [["expression", ["STRING", {}]]], "optional": []},
    "SaveVideo": {
        "required": [["video", ["VIDEO"]], ["sigmas", ["SIGMAS"]], ["filename_prefix", ["STRING", {}]]],
        "optional": [],
    },
    "MarkdownNote": {"required": [], "optional": []},
}


def _ui() -> dict[str, Any]:
    """A small graph where `steps` is wired but `denoise` is a widget."""
    return {
        "nodes": [
            {"id": 1, "type": "UNETLoader", "widgets_values": ["model.safetensors", "default"]},
            {"id": 2, "type": "RandomNoise", "widgets_values": [12345]},
            {
                "id": 3,
                "type": "BasicScheduler",
                "inputs": [{"name": "model", "link": 10}, {"name": "steps", "link": 11}],
                "widgets_values": ["simple", 1.0],
            },
            {
                "id": 4,
                "type": "TurboLoRA",
                "inputs": [{"name": "model", "link": 12}],
                "widgets_values": ["turbo.safetensors", 1.0, False],
            },
            {"id": 5, "type": "MathNode", "widgets_values": ["a * 24"]},
            {"id": 6, "type": "SaveVideo",
             "inputs": [{"name": "video", "link": 13}, {"name": "sigmas", "link": 14}],
             "widgets_values": ["out/x"]},
            {"id": 99, "type": "MarkdownNote", "widgets_values": ["a comment"]},
        ],
        "links": [
            [10, 1, 0, 3, 0, "MODEL"],
            [11, 5, 0, 3, 1, "INT"],
            [12, 1, 0, 4, 0, "MODEL"],
            [13, 4, 0, 6, 0, "VIDEO"],
            [14, 3, 0, 6, 1, "SIGMAS"],
        ],
    }


# ------------------------------------------------------------------ conversion


def test_numeric_widgets_survive_conversion() -> None:
    """The whole point: INT/FLOAT widgets are not links."""
    api = ui_to_api(_ui(), OBJECT_INFO)
    assert api["2"]["inputs"]["noise_seed"] == 12345
    assert api["4"]["inputs"]["strength"] == 1.0
    assert api["4"]["inputs"]["low_vram"] is False


def test_a_wired_int_becomes_a_link_not_a_widget() -> None:
    """`steps` is wired here, so the widget array starts at `denoise`."""
    api = ui_to_api(_ui(), OBJECT_INFO)
    scheduler = api["3"]["inputs"]
    assert scheduler["steps"] == ["5", 0]
    assert scheduler["scheduler"] == "simple"
    assert scheduler["denoise"] == 1.0


def test_links_are_resolved_to_source_and_slot() -> None:
    api = ui_to_api(_ui(), OBJECT_INFO)
    assert api["3"]["inputs"]["model"] == ["1", 0]
    assert api["6"]["inputs"]["video"] == ["4", 0]


def test_notes_are_dropped() -> None:
    assert "99" not in ui_to_api(_ui(), OBJECT_INFO)


def test_unknown_node_class_raises_and_names_it() -> None:
    ui = {"nodes": [{"id": 1, "type": "SomeMissingPack", "widgets_values": []}], "links": []}
    with pytest.raises(KeyError, match="SomeMissingPack"):
        ui_to_api(ui, OBJECT_INFO)


# --------------------------------------------------------------------- pruning


def test_replacing_a_wired_input_orphans_the_helper_which_pruning_removes() -> None:
    """The real workflow: swap a maths node for a literal, then drop the maths node."""
    api = ui_to_api(_ui(), OBJECT_INFO)
    assert "5" in api  # MathNode reachable via the steps link

    api = set_literal(api, "BasicScheduler", "steps", 6)
    api = prune_unreachable(api)

    assert api["3"]["inputs"]["steps"] == 6
    assert "5" not in api, "the maths node should have been pruned"
    assert {"1", "3", "4", "6"} <= set(api)


def test_pruning_keeps_everything_the_output_depends_on() -> None:
    api = prune_unreachable(ui_to_api(_ui(), OBJECT_INFO))
    assert {"1", "4", "6"} <= set(api)


def test_set_literal_does_not_mutate_the_input() -> None:
    api = ui_to_api(_ui(), OBJECT_INFO)
    set_literal(api, "SaveVideo", "filename_prefix", "changed")
    assert api["6"]["inputs"]["filename_prefix"] == "out/x"


def test_pruning_without_an_output_node_raises() -> None:
    with pytest.raises(ValueError, match="no output node"):
        prune_unreachable({"1": {"class_type": "UNETLoader", "inputs": {}}})
