"""Tool-name masking. SPEC.md §5.3/D16."""

from __future__ import annotations

import random

from twin.core.trajectory import ActionStep, NoActionStep, ToolCallStep
from twin.train.masking import mask_tool_names


def test_masking_renames_tool_call_steps_consistently() -> None:
    steps = [
        ToolCallStep(tool="recall", args={"q": "1"}, result_digest="a"),
        ToolCallStep(tool="recall", args={"q": "2"}, result_digest="b"),
        ToolCallStep(tool="web_search", args={}, result_digest="c"),
    ]
    masked_steps, _ = mask_tool_names(steps, available_tools=["recall", "web_search", "reply"], rng=random.Random(0))

    # Both `recall` calls MUST map to the same replacement name.
    first, second = masked_steps[0], masked_steps[1]
    assert isinstance(first, ToolCallStep)
    assert isinstance(second, ToolCallStep)
    assert first.tool == second.tool


def test_masking_is_a_permutation_not_a_collapse() -> None:
    steps = [ToolCallStep(tool="recall", args={}, result_digest="a")]
    _, masked_tools = mask_tool_names(steps, available_tools=["recall", "web_search", "reply"], rng=random.Random(1))
    assert len(set(masked_tools)) == len(set(["recall", "web_search", "reply"]))


def test_masking_is_deterministic_given_the_same_seed() -> None:
    steps = [ToolCallStep(tool="recall", args={}, result_digest="a")]
    tools = ["recall", "web_search", "reply"]
    _, first = mask_tool_names(steps, available_tools=tools, rng=random.Random(42))
    _, second = mask_tool_names(steps, available_tools=tools, rng=random.Random(42))
    assert first == second


def test_masking_leaves_non_tool_call_steps_unchanged() -> None:
    steps = [ActionStep(surface="line", content="hi"), NoActionStep(reason="quiet")]
    masked_steps, _ = mask_tool_names(steps, available_tools=["recall"], rng=random.Random(0))
    assert masked_steps == steps


def test_masking_renames_available_tools_consistently_with_steps() -> None:
    steps = [ToolCallStep(tool="recall", args={}, result_digest="a")]
    masked_steps, masked_tools = mask_tool_names(steps, available_tools=["recall", "web_search"], rng=random.Random(7))
    assert isinstance(masked_steps[0], ToolCallStep)
    assert masked_steps[0].tool in masked_tools
