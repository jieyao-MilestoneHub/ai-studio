"""LINE export parsing and Fragment assembly. Phase 1's acceptance criteria
(twin/PLAN.md §2 Phase 1): fragments produced by ingest MUST cover 100% of
MUST fields, none MUST be missing event_time, and the heldout window's time
MUST actually be later than train's."""

from __future__ import annotations

from datetime import datetime

import pytest

from twin.core.enums import Modality, SourceClass, Split
from twin.ingest.fragment import fragments_from_line_export
from twin.ingest.sources.line import parse_line_export

SAMPLE_EXPORT = """\
[LINE] Test Chat のトーク履歴
保存日時：2026/08/27 10:00

2026/01/15(木)
09:00\tAlice\t早安
09:01\tBob\t早

2026/06/20(土)
14:30\tAlice\t今天要不要出門
14:32\tBob\t好啊
14:33\tBob\t幾點
"""

DEFAULT_TRAIN_CUTOFF = datetime(2026, 6, 1)


def test_parse_line_export_yields_all_messages() -> None:
    messages = list(parse_line_export(SAMPLE_EXPORT))
    assert len(messages) == 5
    assert messages[0].sender == "Alice"
    assert messages[0].content == "早安"
    assert messages[0].sent_at == datetime(2026, 1, 15, 9, 0)
    assert messages[-1].sent_at == datetime(2026, 6, 20, 14, 33)


def test_parse_line_export_folds_multiline_messages() -> None:
    text = SAMPLE_EXPORT.replace("14:33\tBob\t幾點", "14:33\tBob\t幾點\n見面")
    messages = list(parse_line_export(text))
    assert messages[-1].content == "幾點\n見面"


def test_parse_line_export_raises_on_unrecognised_line() -> None:
    text = "2026/01/15(木)\nnot a valid message line\n"
    with pytest.raises(ValueError, match="unrecognised line"):
        list(parse_line_export(text))


class TestFragmentsFromLineExport:
    """SPEC.md §4.4/§4.8 — the actual "minimal ingest that produces a real
    heldout window" Phase 1 asks for."""

    def _assemble(self, train_cutoff: datetime = DEFAULT_TRAIN_CUTOFF) -> list:
        return list(
            fragments_from_line_export(
                SAMPLE_EXPORT,
                principal_id="p1",
                principal_display_name="Alice",
                train_cutoff=train_cutoff,
                sealed_cutoff=train_cutoff.replace(year=2027),
            )
        )

    def test_all_fragments_cover_every_must_field(self) -> None:
        fragments = self._assemble()
        assert fragments
        for fragment in fragments:
            assert fragment.fragment_id
            assert fragment.principal_id == "p1"
            assert fragment.source_class == SourceClass.BEHAVIOR
            assert fragment.modality == Modality.MESSAGE
            assert fragment.content
            assert fragment.event_time is not None
            assert fragment.ingest_time is not None
            assert fragment.split in (Split.TRAIN, Split.HELDOUT, Split.SEALED)
            assert isinstance(fragment.third_party_spans, list)

    def test_principals_own_messages_carry_no_third_party_span(self) -> None:
        """SPEC.md §4.9/§8 guardrail 1: tagging MUST happen at ingest — Alice
        (the principal here) is not a third party in her own export."""
        fragments = self._assemble()
        alice_fragments = [f for f in fragments if f.content.startswith("Alice:")]
        assert alice_fragments
        assert all(f.third_party_spans == [] for f in alice_fragments)

    def test_other_senders_messages_are_tagged_as_third_party(self) -> None:
        """Every message from anyone but the principal is, in full, third-party
        content (SPEC.md §2.1) — the whole fragment content MUST be spanned,
        not left untagged just because no NLP extraction has run yet."""
        fragments = self._assemble()
        bob_fragments = [f for f in fragments if f.content.startswith("Bob:")]
        assert bob_fragments
        for fragment in bob_fragments:
            assert len(fragment.third_party_spans) == 1
            span = fragment.third_party_spans[0]
            assert span.party_ref == "Bob"
            assert span.start == 0
            assert span.end == len(fragment.content)

    def test_no_fragment_is_missing_event_time(self) -> None:
        # event_time is a required field on Fragment (see core/fragment.py), so
        # a script check here is really asserting the model can't be bypassed —
        # this is the "腳本檢查零筆缺 event_time" acceptance line from PLAN.md.
        fragments = self._assemble()
        assert all(f.event_time.value for f in fragments)

    def test_heldout_time_is_later_than_train_time(self) -> None:
        fragments = self._assemble()
        train_times = [f.event_time.value for f in fragments if f.split == Split.TRAIN]
        heldout_times = [f.event_time.value for f in fragments if f.split == Split.HELDOUT]
        assert train_times and heldout_times
        assert max(train_times) < min(heldout_times)

    def test_january_messages_are_train_june_messages_are_heldout(self) -> None:
        fragments = self._assemble()
        splits_by_month = {f.event_time.value[:7]: f.split for f in fragments}
        assert splits_by_month["2026-01"] == Split.TRAIN
        assert splits_by_month["2026-06"] == Split.HELDOUT
