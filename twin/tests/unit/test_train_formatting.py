"""Trajectory -> SFTTrainer chat examples. SPEC.md §4.10, §5.3/D16."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import datasets
import pytest

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split
from twin.core.trajectory import (
    ActionStep,
    Exposure,
    NoActionStep,
    ReflectionStep,
    ToolCallStep,
    Trajectory,
)
from twin.ingest.store import write_trajectories_jsonl
from twin.train.formatting import build_sft_dataset, trajectory_to_messages


def _trajectory(**overrides: object) -> Trajectory:
    defaults: dict[str, object] = dict(
        principal_id="default",
        context_time=datetime(2026, 1, 1, tzinfo=UTC),
        split=Split.TRAIN,
        exposure=Exposure(occurred=True, stimulus="msg", evidence=ExposureEvidence.READ_RECEIPT),
        observation="hello",
        available_tools=["recall", "web_search", "reply"],
        steps=[ActionStep(surface="line", content="hi there")],
        negative_class=NegativeClass.NONE,
        ground_truth_source=GroundTruthSource.OBSERVED,
    )
    defaults.update(overrides)
    return Trajectory(**defaults)  # type: ignore[arg-type]


def test_trajectory_to_messages_is_deterministic_given_the_same_seed() -> None:
    trajectory = _trajectory(steps=[ToolCallStep(tool="recall", args={"q": "x"}, result_digest="d")])
    first = trajectory_to_messages(trajectory, seed=7)
    second = trajectory_to_messages(trajectory, seed=7)
    assert first == second


def test_trajectory_to_messages_masks_tool_names_consistently() -> None:
    trajectory = _trajectory(
        steps=[
            ToolCallStep(tool="recall", args={"q": "x"}, result_digest="d1"),
            ToolCallStep(tool="recall", args={"q": "y"}, result_digest="d2"),
        ]
    )
    messages = trajectory_to_messages(trajectory, seed=1)
    tool_call_messages = [m for m in messages if m.get("tool_calls")]
    names = {m["tool_calls"][0]["function"]["name"] for m in tool_call_messages}
    # Both calls were to the same original tool, so they MUST still map to the same masked name.
    assert len(names) == 1
    # And the masked name must not silently be the literal original name space collapsing to nothing.
    assert names != {""}


def test_trajectory_to_messages_renders_action_step_as_a_masked_tool_call() -> None:
    """SPEC.md §11-A/D28/D29: reply is an ordinary (masked) tool call, not a
    special assistant-content path — otherwise the model would learn "reply"
    as a hardcoded literal instead of one tool among several.

    Not asserted here: that the masked name differs from the literal "reply"
    string. §5.3/D16 masking is a random permutation over the trajectory's
    tool vocabulary, and a permutation CAN have a fixed point — "reply" maps
    to itself with real (not negligible) probability depending on seed and
    vocabulary size, so asserting inequality for one arbitrary seed would be
    flaky. `test_trajectory_to_messages_masks_reply_consistently_with_other_tools`
    below tests the invariant that actually matters and holds unconditionally:
    two *distinct* original tools always get *distinct* masked names.
    """
    trajectory = _trajectory(steps=[ActionStep(surface="line", content="a reply")])
    messages = trajectory_to_messages(trajectory, seed=1)
    tool_call_messages = [m for m in messages if m.get("tool_calls")]
    assert len(tool_call_messages) == 1
    call = tool_call_messages[0]["tool_calls"][0]["function"]
    assert json.loads(call["arguments"]) == {"surface": "line", "content": "a reply"}
    # A plain, unmasked assistant-content message MUST NOT also appear.
    assert not any(m["role"] == "assistant" and m.get("content") == "a reply" for m in messages)


def test_trajectory_to_messages_masks_reply_consistently_with_other_tools() -> None:
    """`reply` MUST enter the same §5.3/D16 name-permutation pool as every
    other tool — a distinct masked name per distinct original tool, and the
    same masked name every time the same original tool recurs."""
    trajectory = _trajectory(
        steps=[
            ToolCallStep(tool="recall", args={"q": "x"}, result_digest="d1"),
            ActionStep(surface="line", content="a reply"),
            ToolCallStep(tool="recall", args={"q": "y"}, result_digest="d2"),
        ]
    )
    messages = trajectory_to_messages(trajectory, seed=1)
    tool_call_messages = [m for m in messages if m.get("tool_calls")]
    names = [m["tool_calls"][0]["function"]["name"] for m in tool_call_messages]
    assert len(names) == 3
    # Both `recall` calls (1st and 3rd) map to the same masked name.
    assert names[0] == names[2]
    # The reply call (2nd) masks to something distinct from `recall`'s masked name.
    assert names[1] != names[0]


def test_trajectory_to_messages_renders_no_action_step_without_reason_text() -> None:
    trajectory = _trajectory(
        steps=[NoActionStep(reason="not sure this needs a response")],
        negative_class=NegativeClass.TRIVIAL,
        exposure=Exposure(occurred=False, stimulus="msg", evidence=ExposureEvidence.ABSENT),
    )
    messages = trajectory_to_messages(trajectory, seed=1)
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    assert assistant_messages == [{"role": "assistant", "content": "", "tool_calls": []}]
    # The reason MUST NOT leak into the trained token sequence (see formatting.py's docstring).
    assert "not sure this needs a response" not in str(messages)


def test_trajectory_to_messages_raises_on_reflection_step() -> None:
    trajectory = _trajectory(steps=[ReflectionStep(content="I felt rushed")])
    with pytest.raises(NotImplementedError):
        trajectory_to_messages(trajectory, seed=1)


def test_build_sft_dataset_is_map_style_not_streaming(tmp_path: Path) -> None:
    trajectories = [_trajectory() for _ in range(3)]
    uri = f"file://{tmp_path}/trajectories.jsonl"
    write_trajectories_jsonl(trajectories, uri)

    dataset, trajectory_ids = build_sft_dataset(uri, seed=1)

    assert isinstance(dataset, datasets.Dataset)
    assert not isinstance(dataset, datasets.IterableDataset)
    assert len(dataset) == 3
    assert trajectory_ids == [t.trajectory_id for t in trajectories]


def test_build_sft_dataset_excludes_non_train_split(tmp_path: Path) -> None:
    train = _trajectory(split=Split.TRAIN)
    heldout = _trajectory(split=Split.HELDOUT)
    uri = f"file://{tmp_path}/trajectories.jsonl"
    write_trajectories_jsonl([train, heldout], uri)

    _dataset, trajectory_ids = build_sft_dataset(uri, seed=1)

    assert trajectory_ids == [train.trajectory_id]


def test_tool_call_arguments_keep_cjk_unescaped() -> None:
    """The training target must contain the principal's actual characters,
    not `\\uXXXX` escapes — the first real adapter learned to emit escapes
    because the default `json.dumps` produced them (2026-08-30 smoke test)."""
    trajectory = _trajectory(steps=[ActionStep(surface="line", content="好啊，晚點聊")])
    messages = trajectory_to_messages(trajectory, seed=0)
    arguments = next(m for m in messages if m["role"] == "assistant" and m.get("tool_calls"))["tool_calls"][0]["function"]["arguments"]
    assert "好啊，晚點聊" in arguments
    assert "\\u" not in arguments
