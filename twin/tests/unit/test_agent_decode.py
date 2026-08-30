"""Tool-call decoding. SPEC.md §6.2/D28."""

from __future__ import annotations

from twin.agent.decode import decode_tool_calls, reply_content


def test_decodes_a_well_formed_qwen3_tool_call() -> None:
    text = '<tool_call>\n{"name": "reply", "arguments": {"surface": "line", "content": "好啊"}}\n</tool_call>'
    calls = decode_tool_calls(text)
    assert [(c.name, c.arguments["content"]) for c in calls] == [("reply", "好啊")]
    assert reply_content(text) == "好啊"


def test_truncated_block_and_string_arguments_still_decode() -> None:
    text = '<tool_call>\n{"name": "lineNotify", "arguments": "{\\"surface\\": \\"line\\", \\"content\\": \\"晚點聊\\"}"}'
    assert reply_content(text) == "晚點聊"


def test_masked_tool_name_does_not_matter_for_reply_content() -> None:
    text = '<tool_call>\n{"name": "surface", "arguments": {"surface": "line", "content": "對呀"}}\n</tool_call>'
    assert reply_content(text) == "對呀"


def test_no_tool_call_or_no_content_is_none() -> None:
    assert reply_content("我覺得第一個選項比較好。") is None
    assert reply_content('<tool_call>\n{"name": "recall", "arguments": {"query": "去年"}}\n</tool_call>') is None
    assert decode_tool_calls("<tool_call>\n{not json") == []


def test_content_cut_off_by_max_new_tokens_is_salvaged() -> None:
    """T v1 emitted `\\uXXXX`-escaped Chinese, so 256 tokens ended mid-string
    for 41/70 items (2026-08-30 run) — the words already emitted are the answer."""
    text = '<tool_call>\n{"name": "reply", "arguments": {"surface": "line", "content": "\\u5c0d\\u5440 \\u6211\\u6703\\u50'
    calls = decode_tool_calls(text)
    assert calls and calls[0].arguments["truncated"] is True
    assert reply_content(text) == "對呀 我會"
