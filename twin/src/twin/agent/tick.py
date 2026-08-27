"""The tick loop. SPEC.md §6.2/D28: runtime MUST be a tick loop, not
request-response. Each tick, the twin chooses zero or more tool calls — an
empty list of tool calls *is* `no_action`, with no special branch.

`Decider` is the swappable-inference-backend interface: whatever actually
powers tool-call decisions — local HF+PEFT running the trained adapter, a
served vLLM endpoint, Ollama, a hosted API — implements `decide()`. This loop
never knows or cares which; the real, inference-backed `Decider` doesn't
exist until Phase 4/11 produce a trained adapter to run, so only the
plumbing (this dispatch) and a scripted fake (for tests) exist yet.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from twin.agent.context import TickContext


class TickResult(BaseModel):
    """An empty `tool_calls` list IS no_action (D28) — there is deliberately
    no separate `no_action: bool` field to keep in sync with it."""

    model_config = ConfigDict(frozen=True)

    tool_calls: list[dict[str, Any]]


class Decider(Protocol):
    def decide(self, context: TickContext) -> TickResult: ...


def run_tick(context: TickContext, decider: Decider) -> TickResult:
    return decider.decide(context)
