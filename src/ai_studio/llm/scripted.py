"""A canned `LlmClient` for tests and offline runs.

The real rewriter is gpt-oss-20b on the pod (`pipeline.pod_llm.PodLlmClient`);
this one answers from a script so prompt conversion can be exercised with no
GPU and no network.
"""

from __future__ import annotations


class LlmError(Exception):
    """The client refused, failed, or ran out of scripted replies."""


class ScriptedLlmClient:
    """Returns canned replies in order; raises when the script runs out."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
        self.calls.append((system, user))
        if not self._replies:
            raise LlmError("no scripted replies left")
        return self._replies.pop(0)
