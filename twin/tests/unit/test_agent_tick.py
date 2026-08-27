"""The tick loop's plumbing. SPEC.md §6.2/D28 — a scripted fake Decider
exercises the "empty tool_calls IS no_action, no special branch" invariant
today, even though the real inference-backed Decider doesn't exist until
Phase 4/11 (mirrors ScriptedLlmClient/GeminiTeacher's fake-client precedent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from twin.agent.context import TickContext
from twin.agent.tick import TickResult, run_tick


@dataclass
class _ScriptedDecider:
    """Returns canned TickResults in sequence, same shape as
    teacher.gemini's tests' fake client — for tests and offline development."""

    scripted: list[TickResult] = field(default_factory=list)

    def decide(self, context: TickContext) -> TickResult:
        return self.scripted.pop(0)


def _context() -> TickContext:
    return TickContext(now=datetime(2026, 8, 27, tzinfo=UTC), inbound_events=[], tool_schemas=[])


def test_run_tick_dispatches_to_the_decider() -> None:
    expected = TickResult(tool_calls=[{"tool": "reply", "args": {"surface": "line", "content": "hi"}}])
    decider = _ScriptedDecider(scripted=[expected])
    assert run_tick(_context(), decider) == expected


def test_empty_tool_calls_is_no_action_with_no_special_branch() -> None:
    """D28: calling zero tools IS no_action. There is no separate no_action
    flag anywhere in TickResult to check instead — the empty list itself
    is the signal, and this pins that exact shape."""
    decider = _ScriptedDecider(scripted=[TickResult(tool_calls=[])])
    result = run_tick(_context(), decider)
    assert result.tool_calls == []
    assert set(TickResult.model_fields.keys()) == {"tool_calls"}
