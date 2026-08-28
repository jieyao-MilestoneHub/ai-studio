"""Third-party span extraction, generalized beyond LINE's whole-message
heuristic. SPEC.md §4.9/D23.
"""

from __future__ import annotations

from twin.ingest.entities import build_glossary_from_contacts, extract_third_party_spans


def test_extract_third_party_spans_finds_a_known_contact_name() -> None:
    text = "昨天跟小美一起吃飯"
    spans = extract_third_party_spans(text, known_parties=["小美"])

    assert len(spans) == 1
    assert spans[0].party_ref == "小美"
    assert text[spans[0].start : spans[0].end] == "小美"


def test_extract_third_party_spans_returns_empty_list_when_principal_only_speaks_of_self() -> None:
    text = "我昨天自己去散步"
    assert extract_third_party_spans(text, known_parties=["小美", "阿明"]) == []


def test_extract_third_party_spans_handles_overlapping_or_repeated_mentions() -> None:
    text = "小美說她要找小美一起去"
    spans = extract_third_party_spans(text, known_parties=["小美"])
    assert len(spans) == 2
    assert all(text[span.start : span.end] == "小美" for span in spans)


def test_build_glossary_from_contacts_merges_relationship_terms() -> None:
    glossary = build_glossary_from_contacts(["小美", "阿明", ""], relationship_terms=["媽", "老闆", " "])
    assert glossary == ["媽", "小美", "老闆", "阿明"]
