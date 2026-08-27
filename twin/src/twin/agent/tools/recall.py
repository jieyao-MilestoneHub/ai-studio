"""The `recall` tool. SPEC.md §3.1/C4 — behind the exact same `Tool`
interface as everything else, wrapping the naive `memory.retrieve` built in
this same pass (Phase 9's coarse-to-fine retrieval replaces the backend
later; this tool's shape does not change when that happens)."""

from __future__ import annotations

import hashlib
from typing import Any

from twin.agent.tools.base import ToolResult
from twin.ingest.store import read_fragments_jsonl
from twin.memory.retrieve import retrieve


class RecallTool:
    name = "recall"

    def __init__(self, fragments_uri: str) -> None:
        self._fragments_uri = fragments_uri

    def schema(self) -> dict[str, Any]:
        return {
            "name": "recall",
            "description": "Search the principal's own memory for fragments matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "time_hint": {"type": ["string", "null"], "default": None},
                },
                "required": ["query"],
            },
        }

    def call(self, *, query: str, time_hint: str | None = None) -> ToolResult:
        fragments = read_fragments_jsonl(self._fragments_uri)
        hits = retrieve(query, fragments, time_hint=time_hint)
        content = "\n".join(fragment.content for fragment in hits)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return ToolResult(ok=True, content=content, result_digest=digest)
