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

from videogen.comfy.graph import Workflow
from videogen.comfy.validate import uses_turbo_lora, validate_graph
from videogen.core.errors import GraphValidationError, UnknownKeyError


def _turbo_graph(lora_node: str = "MiniMaxH3TurboLoRA", sampler: str = "MiniMaxH3TurboSampler"):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "h3_fl2va.safetensors"}},
        "2": {"class_type": lora_node, "inputs": {"lora_name": "h3_turbo_4step.safetensors", "model": ["1", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "4": {"class_type": sampler, "inputs": {"steps": 12, "model": ["2", 0], "positive": ["3", 0]}},
        "5": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "videogen", "images": ["4", 0]}},
    }


def _base_graph() -> dict[str, Any]:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "h3_fl2va.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "4": {"class_type": "KSampler", "inputs": {"steps": 20, "model": ["1", 0]}},
        "5": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "videogen", "images": ["4", 0]}},
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
    graph["_videogen"] = {
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
    assert "_videogen" not in out


def test_a_value_with_no_binding_raises_rather_than_being_dropped() -> None:
    """A silently ignored seed is a run you cannot reproduce and will not notice."""
    with pytest.raises(UnknownKeyError):
        Workflow(_workflow_dict()).with_values({"seed": 42})


def test_binding_to_a_missing_node_fails_at_load_not_at_submit() -> None:
    graph = _workflow_dict()
    graph["_videogen"]["bindings"]["prompt"] = ["404", "text"]
    with pytest.raises(GraphValidationError, match="does not exist"):
        Workflow(graph)


def test_missing_required_bindings_are_reported() -> None:
    graph = _workflow_dict()
    del graph["_videogen"]["bindings"]["width"]
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
