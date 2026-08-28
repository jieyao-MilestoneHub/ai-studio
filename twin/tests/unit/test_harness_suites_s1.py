"""S1 item-bank generation. EVAL.md §3.1-§3.2, §9 anti-pattern #7,
twin/PLAN.md Phase 2. `Teacher` is injected (a fake stands in) — nothing
here needs a real Gemini call."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from twin.core.enums import Modality, SourceClass, Split
from twin.core.fragment import EventTime, Fragment
from twin.harness.schema import HarnessError
from twin.harness.suites.s1 import (
    MAX_ITEMS,
    MIN_ITEMS,
    _GeneratedItem,
    _ItemBankResponse,
    build_item_bank,
)
from twin.ingest.store import write_fragments_jsonl
from twin.teacher.base import TeacherError


def _fragment(fragment_id: str, *, split: Split = Split.HELDOUT, content: str = "x") -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        principal_id="p1",
        source_class=SourceClass.SELF_REPORT,
        modality=Modality.TEXT,
        content=content,
        event_time=EventTime(value="2024-06", precision="month", confidence=0.8),
        ingest_time=datetime(2026, 8, 27, tzinfo=UTC),
        split=split,
    )


def _write_store(tmp_path: Path, fragments: list[Fragment]) -> str:
    uri = f"file://{tmp_path}/fragments.jsonl"
    write_fragments_jsonl(fragments, uri)
    return uri


def _generated(*, prompt: str, source_fragment_ids: list[str], options: list[str] | None = None) -> _GeneratedItem:
    return _GeneratedItem(prompt=prompt, options=options or ["a", "b"], source_fragment_ids=source_fragment_ids)


def _response(
    *,
    value_tradeoff: int = 0,
    preference: int = 0,
    reaction_tendency: int = 0,
    recall: int = 0,
    source_fragment_ids: list[str],
) -> _ItemBankResponse:
    def make(n: int, label: str) -> list[_GeneratedItem]:
        return [_generated(prompt=f"{label}-{i}", source_fragment_ids=source_fragment_ids) for i in range(n)]

    return _ItemBankResponse(
        value_tradeoff=make(value_tradeoff, "vt"),
        preference=make(preference, "pref"),
        reaction_tendency=make(reaction_tendency, "reac"),
        recall=make(recall, "recall"),
    )


@dataclass
class _FakeTeacher:
    response: object
    calls: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, response_schema: type[Any]) -> Any:
        self.calls.append(prompt)
        return self.response


def test_build_item_bank_returns_unique_content_derived_ids(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1"), _fragment("frag-2")])
    ids = ["frag-1", "frag-2"]
    response = _response(value_tradeoff=21, preference=18, reaction_tendency=17, recall=14, source_fragment_ids=ids)
    teacher = _FakeTeacher(response=response)

    bank = build_item_bank(held_out_fragment_ids=ids, teacher=teacher, fragment_store_uri=uri)

    assert len(bank) == 70
    assert MIN_ITEMS <= len(bank) <= MAX_ITEMS
    assert all(item.source_fragment_ids == ids for item in bank)
    assert len({item.item_id for item in bank}) == len(bank)
    assert all(len(item.item_id) == 16 for item in bank)


def test_build_item_bank_calls_teacher_exactly_once(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1")])
    response = _response(value_tradeoff=60, source_fragment_ids=["frag-1"])
    teacher = _FakeTeacher(response=response)

    build_item_bank(held_out_fragment_ids=["frag-1"], teacher=teacher, fragment_store_uri=uri)

    assert len(teacher.calls) == 1


def test_build_item_bank_rejects_empty_fragment_ids_without_calling_teacher(tmp_path: Path) -> None:
    teacher = _FakeTeacher(response=_response(source_fragment_ids=[]))
    uri = f"file://{tmp_path}/fragments.jsonl"

    with pytest.raises(HarnessError, match="empty"):
        build_item_bank(held_out_fragment_ids=[], teacher=teacher, fragment_store_uri=uri)
    assert teacher.calls == []


def test_build_item_bank_rejects_a_missing_fragment_id(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1")])
    teacher = _FakeTeacher(response=_response(source_fragment_ids=["frag-1"]))

    with pytest.raises(HarnessError, match="not found"):
        build_item_bank(held_out_fragment_ids=["frag-missing"], teacher=teacher, fragment_store_uri=uri)
    assert teacher.calls == []


def test_build_item_bank_rejects_non_heldout_fragments(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1", split=Split.TRAIN)])
    teacher = _FakeTeacher(response=_response(source_fragment_ids=["frag-1"]))

    with pytest.raises(HarnessError, match="HELDOUT"):
        build_item_bank(held_out_fragment_ids=["frag-1"], teacher=teacher, fragment_store_uri=uri)
    assert teacher.calls == []


def test_build_item_bank_rejects_a_citation_outside_the_held_out_set(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1")])
    response = _response(value_tradeoff=60, source_fragment_ids=["frag-not-real"])
    teacher = _FakeTeacher(response=response)

    with pytest.raises(TeacherError, match="outside"):
        build_item_bank(held_out_fragment_ids=["frag-1"], teacher=teacher, fragment_store_uri=uri)


def test_build_item_bank_rejects_too_few_items(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1")])
    response = _response(value_tradeoff=5, source_fragment_ids=["frag-1"])
    teacher = _FakeTeacher(response=response)

    with pytest.raises(TeacherError, match="60-80"):
        build_item_bank(held_out_fragment_ids=["frag-1"], teacher=teacher, fragment_store_uri=uri)


def test_build_item_bank_rejects_too_many_items(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1")])
    response = _response(value_tradeoff=90, source_fragment_ids=["frag-1"])
    teacher = _FakeTeacher(response=response)

    with pytest.raises(TeacherError, match="60-80"):
        build_item_bank(held_out_fragment_ids=["frag-1"], teacher=teacher, fragment_store_uri=uri)


def test_build_item_bank_rejects_duplicate_generated_items(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1")])
    duplicate = _generated(prompt="same", source_fragment_ids=["frag-1"])
    response = _ItemBankResponse(
        value_tradeoff=[duplicate, duplicate] + [_generated(prompt=f"vt-{i}", source_fragment_ids=["frag-1"]) for i in range(58)],
        preference=[],
        reaction_tendency=[],
        recall=[],
    )
    teacher = _FakeTeacher(response=response)

    with pytest.raises(TeacherError, match="duplicate"):
        build_item_bank(held_out_fragment_ids=["frag-1"], teacher=teacher, fragment_store_uri=uri)


def test_build_item_bank_rejects_an_oversized_prompt_without_calling_teacher(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1", content="x" * 100_000)])
    teacher = _FakeTeacher(response=_response(source_fragment_ids=["frag-1"]))

    with pytest.raises(HarnessError, match="char"):
        build_item_bank(held_out_fragment_ids=["frag-1"], teacher=teacher, fragment_store_uri=uri)
    assert teacher.calls == []


def test_item_id_is_deterministic_and_content_derived(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1")])
    response = _response(value_tradeoff=60, source_fragment_ids=["frag-1"])

    bank_a = build_item_bank(held_out_fragment_ids=["frag-1"], teacher=_FakeTeacher(response=response), fragment_store_uri=uri)
    bank_b = build_item_bank(held_out_fragment_ids=["frag-1"], teacher=_FakeTeacher(response=response), fragment_store_uri=uri)

    assert [item.item_id for item in bank_a] == [item.item_id for item in bank_b]


def test_prompt_contains_fragment_ids_and_content(tmp_path: Path) -> None:
    uri = _write_store(tmp_path, [_fragment("frag-1", content="the actual content")])
    response = _response(value_tradeoff=60, source_fragment_ids=["frag-1"])
    teacher = _FakeTeacher(response=response)

    build_item_bank(held_out_fragment_ids=["frag-1"], teacher=teacher, fragment_store_uri=uri)

    assert "frag-1" in teacher.calls[0]
    assert "the actual content" in teacher.calls[0]
