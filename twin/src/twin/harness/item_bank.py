"""S1 item bank freezing. twin/PLAN.md Phase 2's own acceptance criterion —
not literally stated in EVAL.md/SPEC.md — is that the 60-80 items used for
Wave 1 (R1) MUST be provably identical to what's used 14 days later for
Wave 2 (R2), and again for A/B0/B1/B2 judging (Phase 6+). `RunManifest`
(`harness.manifest`) already establishes the write-once, hash-verified
pattern this reuses for a new artifact type it doesn't itself cover: a raw
item bank, not a judge/eval run.

`S1WaveManifest.completed_at` is the durable "project day 0" artifact —
EVAL.md §1.2's 14-day clock starts the instant Wave 1 finishes, not when
generation happens, so that timestamp lives here, not in `S1BankManifest`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import fsspec
from pydantic import BaseModel, ConfigDict

from twin.core.hashing import dataset_hash
from twin.harness.schema import HarnessError, S1Item


class S1BankManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    bank_hash: str
    item_count: int
    source_fragment_dataset_hash: str
    teacher_model: str
    created_at: datetime


class S1WaveManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    wave: Literal[1, 2]
    bank_hash: str
    item_count: int
    completed_at: datetime


def bank_hash(bank: list[S1Item]) -> str:
    """Content-sensitive for free: `S1Item.item_id` is itself a content hash
    (see `harness.suites.s1._item_id`), so this transitively changes if any
    item's content changes, not just if the set of ids changes."""
    return dataset_hash(item.item_id for item in bank)


def _ensure_parent(uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)


def write_item_bank_once(bank: list[S1Item], *, bank_uri: str, manifest: S1BankManifest, manifest_uri: str) -> None:
    """Checks both `bank_uri` and `manifest_uri` are absent before writing
    either — a bank frozen without its manifest (or vice versa) is a state
    `read_and_verify_item_bank` could not detect as incomplete."""
    bank_fs, bank_path = fsspec.core.url_to_fs(bank_uri)
    manifest_fs, manifest_path = fsspec.core.url_to_fs(manifest_uri)
    if bank_fs.exists(bank_path):
        raise HarnessError(
            f"an item bank already exists at {bank_uri} — frozen once written; "
            f"build and freeze a new bank rather than overwriting this one"
        )
    if manifest_fs.exists(manifest_path):
        raise HarnessError(
            f"a bank manifest already exists at {manifest_uri} — frozen once written; "
            f"build and freeze a new bank rather than overwriting this one"
        )
    _ensure_parent(bank_uri)
    _ensure_parent(manifest_uri)
    with fsspec.open(bank_uri, "w", encoding="utf-8") as f:
        for item in bank:
            f.write(item.model_dump_json())
            f.write("\n")
    with fsspec.open(manifest_uri, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json())


def read_and_verify_item_bank(*, bank_uri: str, manifest_uri: str) -> tuple[list[S1Item], S1BankManifest]:
    """Refuses to hand back a bank that doesn't match its own frozen hash —
    this is what protects R1-vs-R2 self-consistency from a silently-edited
    question between Wave 1 and Wave 2."""
    with fsspec.open(manifest_uri, "r", encoding="utf-8") as f:
        manifest = S1BankManifest.model_validate_json(f.read())

    items: list[S1Item] = []
    with fsspec.open(bank_uri, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                items.append(S1Item.model_validate_json(stripped))

    if len(items) != manifest.item_count:
        raise HarnessError(
            f"item bank at {bank_uri} has {len(items)} items, its manifest recorded "
            f"{manifest.item_count} — the bank was edited after freezing"
        )
    actual_hash = bank_hash(items)
    if actual_hash != manifest.bank_hash:
        raise HarnessError(
            f"item bank at {bank_uri} does not match its frozen manifest hash "
            f"({actual_hash} != {manifest.bank_hash}) — the bank was edited after freezing"
        )
    return items, manifest


def write_wave_manifest_once(manifest: S1WaveManifest, uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        raise HarnessError(
            f"a wave {manifest.wave} manifest already exists at {uri} — MUST NOT be "
            f"rewritten once written (its completed_at timestamp is, for wave 1, the "
            f"project's day-0 record); investigate rather than overwrite"
        )
    _ensure_parent(uri)
    with fsspec.open(uri, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json())
