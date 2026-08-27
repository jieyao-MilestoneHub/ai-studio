"""Naive recall — a keyword + time-window filter over Fragments.

SPEC.md §3.1/C4: `recall()` MUST be an ordinary tool, behind the same
interface as any other plugin. Building this now (rather than leaving it as
an interface stub) is what makes `agent.tools.recall` a real, testable tool
call instead of an untestable one — Phase 1 already has real Fragments to
run it against, so there is no reason to defer that much.

This is deliberately a plain function, not a `Protocol`: unlike `Teacher` or
`Surface`, there is exactly one realistic implementation path for the
foreseeable future — Phase 9's coarse-to-fine retrieval (§4.5) *replaces*
this function, it doesn't coexist alongside it as an alternate backend.
Nothing here does entity clustering, period-level drill-down, salience
weighting, or conflict-aware version selection — those need a real corpus
(Phase 9) to mean anything.
"""

from __future__ import annotations

from collections.abc import Iterable

from twin.core.fragment import Fragment


def retrieve(
    query: str,
    fragments: Iterable[Fragment],
    *,
    time_hint: str | None = None,
    limit: int = 10,
) -> list[Fragment]:
    """Case-insensitive substring match on `content`, optionally narrowed to
    fragments whose `event_time.value` starts with `time_hint` (e.g. "2026-06"
    matches any precision at or under that prefix). Most-recent-first."""
    query_lower = query.lower()
    matches = [
        fragment
        for fragment in fragments
        if query_lower in fragment.content.lower()
        and (time_hint is None or fragment.event_time.value.startswith(time_hint))
    ]
    matches.sort(key=lambda f: f.event_time.value, reverse=True)
    return matches[:limit]
