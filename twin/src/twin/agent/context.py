"""Tick context assembly. SPEC.md §6.1 (tool schemas injected at runtime,
the C2 boundary) and §6.2 (each tick provides new messages, time, environment
events). Deliberately narrow: no memory content is pre-loaded here — the
tick loop calls `recall` when it wants memory (§6.2's own model), so this
function's whole job is tool schemas plus tick metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from twin.agent.tools.base import Tool


class TickContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    now: datetime
    inbound_events: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]


def build_context(*, tools: Sequence[Tool], inbound_events: Sequence[dict[str, Any]], now: datetime) -> TickContext:
    return TickContext(
        now=now,
        inbound_events=list(inbound_events),
        tool_schemas=[tool.schema() for tool in tools],
    )
