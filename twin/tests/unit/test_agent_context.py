"""Tick context assembly. SPEC.md §6.1/§6.2."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from twin.agent.context import TickContext, build_context
from twin.agent.tools.base import ToolResult


class _StubTool:
    name = "stub"

    def schema(self) -> dict[str, Any]:
        return {"name": "stub"}

    def call(self, **kwargs: Any) -> ToolResult:
        return ToolResult(ok=True, content="", result_digest="")


def test_build_context_collects_tool_schemas_and_metadata() -> None:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    context = build_context(tools=[_StubTool()], inbound_events=[{"kind": "message"}], now=now)

    assert context.now == now
    assert context.inbound_events == [{"kind": "message"}]
    assert context.tool_schemas == [{"name": "stub"}]


def test_build_context_carries_no_memory_content() -> None:
    """SPEC.md §6.2: memory is not pre-loaded into context — the twin calls
    `recall` when it wants memory. TickContext's fields are exhaustive: if
    this ever gains a field carrying fragment/memory content directly, that
    would be the violation this test is meant to catch."""
    build_context(tools=[], inbound_events=[], now=datetime(2026, 1, 1, tzinfo=UTC))
    assert set(TickContext.model_fields.keys()) == {"now", "inbound_events", "tool_schemas"}
