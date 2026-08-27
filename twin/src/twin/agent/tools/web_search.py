"""The `web_search` tool. SPEC.md §6.3 names it, but no backend is
spec-mandated — which provider/API to call is a genuinely open choice, not a
spec gap, so this stays thin until that choice is made."""

from __future__ import annotations

from typing import Any

from twin.agent.tools.base import ToolResult


class WebSearchTool:
    name = "web_search"

    def schema(self) -> dict[str, Any]:
        return {
            "name": "web_search",
            "description": "External lookup.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }

    def call(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError(
            "web_search has no backend decision yet — no SPEC.md section "
            "mandates a specific search provider; this is an open "
            "implementation choice, not a deferred spec requirement."
        )
