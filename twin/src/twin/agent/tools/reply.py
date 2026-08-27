"""The `reply` tool. SPEC.md §6.3/§6.5, D29: replying is an ordinary tool call
("突然想回" IS the twin choosing to call `reply`) — but calling this tool only
*stages* an intent. It never sends. Send/hold/block is `agent.gate`'s job,
intercepted at the runtime layer, specifically so upgrading or downgrading
the send gate never requires touching this tool's definition or a retrain
(D29's whole point).
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict

from twin.agent.tools.base import ToolResult


class ReplyIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    surface: str
    content: str


class ReplyTool:
    name = "reply"

    def schema(self) -> dict[str, Any]:
        return {
            "name": "reply",
            "description": "Draft a reply on a surface. Sending is decided by the send gate, not this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["surface", "content"],
            },
        }

    def call(self, *, surface: str, content: str) -> ToolResult:
        intent = ReplyIntent(surface=surface, content=content)
        digest = hashlib.sha256(intent.model_dump_json().encode("utf-8")).hexdigest()[:16]
        return ToolResult(ok=True, content=content, result_digest=digest)
