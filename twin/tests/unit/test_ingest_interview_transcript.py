"""Interview transcript ingest. INTERVIEW.md §4, §6, §7 Q8; SPEC.md §4.9/D23,
D26.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from twin.core.enums import Modality, Precision, SourceClass
from twin.ingest.sources import interview_transcript as it_module
from twin.ingest.sources.interview_transcript import fragments_from_interview_transcript

TRAIN_CUTOFF = datetime(2020, 1, 1)
SEALED_CUTOFF = datetime(2030, 1, 1)
SESSION_STARTED_AT = datetime(2026, 8, 28, 10, 0)

# A real interview transcript always mentions someone — see
# test_fragments_from_interview_transcript_rejects_empty_known_parties for
# the case this default deliberately steers every other test away from.
DEFAULT_KNOWN_PARTIES = ["小美"]


def _fragments(blocks: dict[str, str], known_parties: list[str] | None = None) -> list:
    return list(
        fragments_from_interview_transcript(
            blocks,
            principal_id="p1",
            session_started_at=SESSION_STARTED_AT,
            known_parties=DEFAULT_KNOWN_PARTIES if known_parties is None else known_parties,
            train_cutoff=TRAIN_CUTOFF,
            sealed_cutoff=SEALED_CUTOFF,
        )
    )


ALL_BLOCKS = {
    "A": "我出生在台北，後來搬到台中。",
    "B": "有一次小美傳訊息給我我沒回。",
    "C": "大概去年夏天我去爬山。",
    "D": "希望它不要亂猜我的意思。",
}


def test_fragments_from_interview_transcript_produces_one_fragment_per_block() -> None:
    fragments = _fragments(ALL_BLOCKS)
    assert len(fragments) == 4
    assert [f.content for f in fragments] == [ALL_BLOCKS[label] for label in ("A", "B", "C", "D")]


def test_fragments_are_self_report_and_text_modality() -> None:
    fragments = _fragments(ALL_BLOCKS)
    assert all(f.source_class == SourceClass.SELF_REPORT for f in fragments)
    assert all(f.modality == Modality.TEXT for f in fragments)
    assert all(f.event_time.precision == Precision.MINUTE for f in fragments)


def test_fragments_round_trip_to_the_original_transcript_text() -> None:
    fragments = _fragments(ALL_BLOCKS)
    reconstructed = "".join(f.content for f in fragments)
    assert reconstructed == "".join(ALL_BLOCKS[label] for label in ("A", "B", "C", "D"))


def test_missing_blocks_are_simply_skipped() -> None:
    fragments = _fragments({"A": ALL_BLOCKS["A"], "C": ALL_BLOCKS["C"]})
    assert len(fragments) == 2
    assert fragments[0].content == ALL_BLOCKS["A"]
    assert fragments[1].content == ALL_BLOCKS["C"]


def test_event_time_reflects_when_spoken_not_content_recalled() -> None:
    """Block C's own content recalls "去年夏天" — a fuzzy date that must stay
    untouched verbatim in `content` (INTERVIEW.md §6.2 step 3 / C2) — while
    `event_time` reflects the session's actual clock, offset by the
    cumulative duration of the blocks spoken before it (§4)."""
    fragments = _fragments(ALL_BLOCKS)
    by_label = dict(zip(("A", "B", "C", "D"), fragments, strict=True))

    assert by_label["A"].event_time.value == "2026-08-28T10:00"
    assert by_label["B"].event_time.value == "2026-08-28T10:42"  # +42 min, block A's duration
    assert by_label["C"].event_time.value == "2026-08-28T11:18"  # +42+36 min
    assert "去年夏天" in by_label["C"].content


def test_extract_third_party_spans_is_called_for_every_block(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = it_module.extract_third_party_spans

    def _counting_wrapper(text: str, *, known_parties: list[str]):
        calls.append(text)
        return real(text, known_parties=known_parties)

    monkeypatch.setattr(it_module, "extract_third_party_spans", _counting_wrapper)

    _fragments(ALL_BLOCKS, known_parties=["小美"])

    assert calls == [ALL_BLOCKS[label] for label in ("A", "B", "C", "D")]


def test_a_blocks_own_message_mentioning_a_known_party_is_tagged() -> None:
    fragments = _fragments(ALL_BLOCKS, known_parties=["小美"])
    block_b = fragments[1]
    assert len(block_b.third_party_spans) == 1
    assert block_b.third_party_spans[0].party_ref == "小美"


def test_event_time_confidence_is_honestly_below_one() -> None:
    """The per-block offset is a derived estimate (INTERVIEW.md §4's minutes
    are a suggested, not guaranteed, allocation) — SPEC.md §4.4 forbids
    claiming full confidence in a value that isn't a platform timestamp."""
    fragments = _fragments(ALL_BLOCKS)
    assert all(f.event_time.confidence < 1.0 for f in fragments)


def test_rejects_an_unrecognised_block_label() -> None:
    with pytest.raises(ValueError, match="unrecognised interview block label"):
        _fragments({"A": ALL_BLOCKS["A"], "B1": "typo'd label"})


def test_rejects_empty_known_parties() -> None:
    with pytest.raises(ValueError, match="known_parties is empty"):
        _fragments(ALL_BLOCKS, known_parties=[])
