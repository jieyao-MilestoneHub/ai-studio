"""Pre-training dataset gate. Each rule is a defect a real run surfaced first."""

from __future__ import annotations

import datasets
import pytest

from twin.train.preflight import assert_dataset_trainable, inspect_dataset


def _reply(content: str, surface: str = "line") -> dict:
    import json

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": json.dumps({"surface": surface, "content": content}, ensure_ascii=False)}}],
    }


def _ds(assistants: list[dict]) -> datasets.Dataset:
    return datasets.Dataset.from_list(
        [{"trajectory_id": str(i), "messages": [{"role": "user", "content": "q"}, a]} for i, a in enumerate(assistants)]
    )


def test_counts_replies_no_action_and_self_report() -> None:
    ds = _ds([_reply("好"), _reply("我在台北長大", "interview"), {"role": "assistant", "content": "", "tool_calls": []}])
    r = inspect_dataset(ds)
    assert (r.reply_targets, r.no_action_targets, r.self_report_targets) == (2, 1, 1)
    assert r.escaped_targets == 0 and r.empty_targets == 0


def test_rejects_escaped_targets() -> None:
    bad = {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": '{"surface": "line", "content": "\\u5c0d\\u5440"}'}}]}
    with pytest.raises(ValueError, match="escapes"):
        assert_dataset_trainable(_ds([bad, _reply("ok", "interview")]), require_self_report=False)


def test_rejects_empty_targets_and_missing_self_report() -> None:
    with pytest.raises(ValueError, match="empty"):
        assert_dataset_trainable(_ds([_reply("  "), _reply("ok", "interview")]), require_self_report=False)
    with pytest.raises(ValueError, match="D19"):
        assert_dataset_trainable(_ds([_reply("ok")] * 200 + [_reply("i", "interview")]), require_self_report=True)
    assert_dataset_trainable(_ds([_reply("ok")] * 200 + [_reply("i", "interview")]), require_self_report=False)


def test_passes_a_healthy_set() -> None:
    ds = _ds([_reply("ok")] * 50 + [_reply("我在台北長大", "interview")] * 2)
    assert assert_dataset_trainable(ds, require_self_report=True).self_report_share == pytest.approx(2 / 52)
