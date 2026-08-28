"""Third-party span extraction, generalized beyond LINE's whole-message
heuristic (`ingest.sources.line`'s own docstring names this exact gap: "the
principal's own messages *mentioning* a third party — that's entity
extraction, ingest.entities, a later phase's job, not this one's"). SPEC.md
§4.9/D23.

Rule-based v1, not Teacher-driven: INTERVIEW.md §7's Q8 is a hard blocker on
interview-transcript fragments entering the memory layer, so gating it
behind a live network call that can fail on RPD exhaustion (D9's named
failure symptom, "配額耗盡，資料工廠停擺") is a bad dependency for something
that MUST run before every transcript fragment. Under-tagging is
recoverable later — SPEC.md §4.9/D23 already treats disposition (and, by the
same logic, re-extraction) as a re-runnable step, never one-shot.
"""

from __future__ import annotations

from collections.abc import Iterable

from twin.core.fragment import ThirdPartySpan


def build_glossary_from_contacts(contact_names: Iterable[str], *, relationship_terms: Iterable[str] = ()) -> list[str]:
    """Merges known contact names with a small supplementary relationship-
    term list (媽/爸/老闆/男友/女友/...) — the two sources
    `extract_third_party_spans` can realistically draw on without a live
    Teacher call."""
    names = {name.strip() for name in contact_names if name.strip()}
    terms = {term.strip() for term in relationship_terms if term.strip()}
    return sorted(names | terms)


def extract_third_party_spans(text: str, *, known_parties: list[str]) -> list[ThirdPartySpan]:
    """Substring matching over `text` — one span per match, reusing
    `core.fragment.ThirdPartySpan` (never a new type). Documented
    limitation: misses relationship mentions with no resolvable name/term
    (e.g. an unnamed "那個朋友"). A future Teacher-driven "pass 2" over
    already-ingested fragments is a possible follow-up, out of scope here."""
    spans: list[ThirdPartySpan] = []
    for party in known_parties:
        if not party:
            continue
        start = 0
        while True:
            index = text.find(party, start)
            if index == -1:
                break
            spans.append(ThirdPartySpan(start=index, end=index + len(party), party_ref=party))
            start = index + len(party)
    return spans
