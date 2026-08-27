"""The tool interface. SPEC.md §3.1/C2 (tool schema/list MUST be injected by
L4 at inference time, MUST NOT enter L3 weights) and C4 (`recall()` MUST be
an ordinary tool, sharing this exact interface with everything else — no
bespoke memory subsystem). This is the concrete mechanism both boundaries
depend on: every tool, including `recall`, implements `Tool`.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class ToolResult(BaseModel):
    """`result_digest` matches `core.trajectory.ToolCallStep.result_digest` —
    what a tool actually returned to the twin is what gets recorded as the
    training signal, not a separate summary invented at ingest time."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    content: str
    result_digest: str


class Tool(Protocol):
    name: str

    def schema(self) -> dict[str, Any]:
        """The JSON-schema-shaped description injected into context at
        inference time (C2) — this MUST NOT be baked into L3 weights."""
        ...

    def call(self, **kwargs: Any) -> ToolResult: ...
