"""S1 item-bank freeze/write-once. twin/PLAN.md Phase 2's own acceptance
criterion: R1-vs-R2 self-consistency depends on nothing being able to
silently edit a question after freezing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import fsspec
import pytest

from twin.harness.item_bank import (
    S1BankManifest,
    bank_hash,
    read_and_verify_item_bank,
    write_item_bank_once,
)
from twin.harness.schema import HarnessError, S1Item


def _item(
    item_id: str,
    *,
    item_type: Literal["value_tradeoff", "preference", "reaction_tendency", "recall"] = "value_tradeoff",
) -> S1Item:
    return S1Item(
        item_id=item_id,
        item_type=item_type,
        prompt=f"prompt-{item_id}",
        options=["a", "b"],
        source_fragment_ids=["frag-1"],
    )


def _manifest_for(bank: list[S1Item]) -> S1BankManifest:
    return S1BankManifest(
        bank_hash=bank_hash(bank),
        item_count=len(bank),
        source_fragment_dataset_hash="x" * 64,
        teacher_model="gemini-test",
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    bank = [_item("a"), _item("b")]
    bank_uri = f"file://{tmp_path}/bank.jsonl"
    manifest_uri = f"file://{tmp_path}/manifest.json"

    write_item_bank_once(bank, bank_uri=bank_uri, manifest=_manifest_for(bank), manifest_uri=manifest_uri)
    read_back, manifest = read_and_verify_item_bank(bank_uri=bank_uri, manifest_uri=manifest_uri)

    assert [item.item_id for item in read_back] == ["a", "b"]
    assert manifest.item_count == 2
    assert manifest.bank_hash == bank_hash(bank)


def test_refuses_to_overwrite_an_existing_bank(tmp_path: Path) -> None:
    bank = [_item("a")]
    bank_uri = f"file://{tmp_path}/bank.jsonl"
    write_item_bank_once(
        bank, bank_uri=bank_uri, manifest=_manifest_for(bank), manifest_uri=f"file://{tmp_path}/manifest.json"
    )

    with pytest.raises(HarnessError, match="already exists"):
        write_item_bank_once(
            [_item("b")],
            bank_uri=bank_uri,
            manifest=_manifest_for([_item("b")]),
            manifest_uri=f"file://{tmp_path}/other_manifest.json",
        )


def test_refuses_when_only_the_manifest_path_is_already_taken(tmp_path: Path) -> None:
    """Fail-fast, no partial writes: the manifest path being occupied refuses
    the whole write — including the bank file, which must not appear either."""
    bank_uri = f"file://{tmp_path}/bank.jsonl"
    manifest_uri = f"file://{tmp_path}/manifest.json"
    with fsspec.open(manifest_uri, "w", encoding="utf-8") as f:
        f.write("unrelated")

    with pytest.raises(HarnessError, match="already exists"):
        write_item_bank_once(
            [_item("a")], bank_uri=bank_uri, manifest=_manifest_for([_item("a")]), manifest_uri=manifest_uri
        )

    fs, path = fsspec.core.url_to_fs(bank_uri)
    assert not fs.exists(path)


def test_read_and_verify_detects_a_changed_item_id(tmp_path: Path) -> None:
    """This layer's guarantee is over the *set* of item_ids the manifest
    froze, not each item's content in isolation — id-to-content binding is
    s1.py's job (item_id is itself content-derived there; see
    test_harness_suites_s1.py's determinism test). Swapping in a same-count
    bank with a different id is exactly what bank_hash is designed to catch."""
    bank = [_item("a"), _item("b")]
    bank_uri = f"file://{tmp_path}/bank.jsonl"
    manifest_uri = f"file://{tmp_path}/manifest.json"
    write_item_bank_once(bank, bank_uri=bank_uri, manifest=_manifest_for(bank), manifest_uri=manifest_uri)

    swapped = _item("a").model_dump_json() + "\n" + _item("c").model_dump_json() + "\n"
    with fsspec.open(bank_uri, "w", encoding="utf-8") as f:
        f.write(swapped)

    with pytest.raises(HarnessError, match="does not match"):
        read_and_verify_item_bank(bank_uri=bank_uri, manifest_uri=manifest_uri)


def test_read_and_verify_detects_a_truncated_bank(tmp_path: Path) -> None:
    bank = [_item("a"), _item("b")]
    bank_uri = f"file://{tmp_path}/bank.jsonl"
    manifest_uri = f"file://{tmp_path}/manifest.json"
    write_item_bank_once(bank, bank_uri=bank_uri, manifest=_manifest_for(bank), manifest_uri=manifest_uri)

    with fsspec.open(bank_uri, "w", encoding="utf-8") as f:
        f.write(_item("a").model_dump_json() + "\n")

    with pytest.raises(HarnessError, match="its manifest recorded"):
        read_and_verify_item_bank(bank_uri=bank_uri, manifest_uri=manifest_uri)


def test_bank_hash_is_order_independent_but_id_sensitive() -> None:
    a, b = _item("a"), _item("b")
    assert bank_hash([a, b]) == bank_hash([b, a])
    assert bank_hash([a, b]) != bank_hash([a])
    assert bank_hash([a]) != bank_hash([_item("a-different")])
