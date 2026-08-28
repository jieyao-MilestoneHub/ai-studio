"""Interview transcript post-processing. INTERVIEW.md §6.2's four ordered
steps, re-runnable by construction: "詞表更新後能重新產生逐字稿" — this module
takes a raw transcript plus a glossary and returns a corrected one, pure, so
re-running it after a glossary update is just calling it again.
"""

from __future__ import annotations


def apply_correction_glossary(text: str, *, glossary: dict[str, str]) -> str:
    """Unifies INTERVIEW.md §6.2 steps 1 (proper-noun correction) and 2
    (code-switch restoration): both reduce to "replace mis-transcribed
    forms via a term list." The glossary is caller-supplied (built from
    contacts/past messages, per §6.2's own text, e.g.
    `ingest.entities.build_glossary_from_contacts`) — this function does not
    attempt to detect ASR errors on its own.

    Step 3 ("口語保留") has no function here at all: it is a MUST NOT (don't
    smooth disfluencies), satisfied by this function only ever touching
    glossary-matched terms, never general phrasing, fillers, or repetition.
    """
    corrected = text
    for wrong, right in glossary.items():
        corrected = corrected.replace(wrong, right)
    return corrected


def mark_unclear_spans(text: str, *, low_confidence_ranges: list[tuple[int, int]]) -> str:
    """INTERVIEW.md §6.2 step 4: `[unclear]` MUST replace undeterminable
    spans, MUST NOT be guess-filled. `low_confidence_ranges` is an external
    interface point — whatever produces the raw transcript (an open Phase
    3-B question, see twin/PLAN.md) supplies these as `(start, end)` byte
    offsets into `text`; this function only consumes them, it does not
    invent a confidence heuristic of its own."""
    if not low_confidence_ranges:
        return text
    # Apply back-to-front so replacing a later range doesn't shift the
    # indices an earlier range still needs to be valid against.
    ordered = sorted(low_confidence_ranges, key=lambda r: r[0], reverse=True)
    result = text
    for start, end in ordered:
        if start < 0 or end > len(result) or start >= end:
            raise ValueError(f"invalid low_confidence_range ({start}, {end}) for text of length {len(result)}")
        result = result[:start] + "[unclear]" + result[end:]
    return result


def run_postprocessing_pipeline(
    raw_transcript: str, *, correction_glossary: dict[str, str], low_confidence_ranges: list[tuple[int, int]]
) -> str:
    """Composed and pure — INTERVIEW.md §6.2's "MUST 可重跑" requirement
    falls out of this for free: the same raw transcript plus an updated
    glossary simply produces a different corrected transcript, no extra
    machinery needed. Unclear-marking runs first, against `raw_transcript`'s
    own indices, before glossary correction — correcting a mis-transcribed
    term must not shift the ranges a caller computed against the raw text."""
    marked = mark_unclear_spans(raw_transcript, low_confidence_ranges=low_confidence_ranges)
    return apply_correction_glossary(marked, glossary=correction_glossary)
