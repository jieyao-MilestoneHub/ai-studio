"""Decode a base-model completion into tool calls. L4 (SPEC.md §6.2, D28): the
model's turn is zero or more tool calls; text outside a `<tool_call>` block
is not an action. Qwen3's chat template renders each call as
`<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>`; this is the
inverse, tolerant of a missing closing tag (max_new_tokens truncation) and of
`arguments` arriving as a JSON string instead of an object.

Lives in `twin.agent`, not `twin.train` (C2/C3: serving must not depend on
trainer internals) — the harness's T inference and the future tick loop
share this one decoder.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


class DecodedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any]


def _load_json_prefix(text: str) -> dict[str, Any] | None:
    """`json.loads` the first complete object in `text`; None if none parses."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


_TRUNCATED_CALL = re.compile(r'"name"\s*:\s*"([^"]+)".*?"content"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)


def _salvage_truncated(block: str) -> dict[str, Any] | None:
    """A call cut off mid-`content` by max_new_tokens: recover the name and
    whatever of the content string was emitted. Drops a trailing partial
    escape (`\\u12`) so the fragment still decodes as JSON."""
    match = _TRUNCATED_CALL.search(block)
    if match is None:
        return None
    raw = match.group(2)
    raw = re.sub(r"\\(u[0-9a-fA-F]{0,3}|$)$", "", raw)
    try:
        content = json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return None
    return {"name": match.group(1), "arguments": {"content": content, "truncated": True}}


def decode_tool_calls(completion: str) -> list[DecodedToolCall]:
    calls: list[DecodedToolCall] = []
    for match in _BLOCK.finditer(completion):
        obj = _load_json_prefix(match.group(1)) or _salvage_truncated(match.group(1))
        if obj is None or not isinstance(obj.get("name"), str):
            continue
        arguments = obj.get("arguments", {})
        if isinstance(arguments, str):
            parsed = _load_json_prefix(arguments)
            arguments = parsed if parsed is not None else {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"raw": arguments}
        calls.append(DecodedToolCall(name=obj["name"], arguments=arguments))
    return calls


def reply_content(completion: str) -> str | None:
    """The `content` argument of the first tool call that carries one — what
    the principal would have said. Tool *names* are unreliable at inference
    (they were masked at train time, SPEC.md §5.3), so this keys on the
    argument shape, not on the name being "reply". None when the model
    called no tool with a `content` (i.e. no_action, or a non-reply tool)."""
    for call in decode_tool_calls(completion):
        content = call.arguments.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None
