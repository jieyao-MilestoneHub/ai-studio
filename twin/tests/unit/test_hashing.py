"""Deterministic hashes. SPEC.md §7.5, data-contract skill rule 8: the
algorithm MUST be fixed and MUST NOT depend on ordering or timestamps."""

from __future__ import annotations

from pathlib import Path

from twin.core.hashing import adapter_hash, config_hash, dataset_hash


def test_dataset_hash_is_order_independent() -> None:
    assert dataset_hash(["b", "a", "c"]) == dataset_hash(["a", "b", "c"]) == dataset_hash(["c", "a", "b"])


def test_dataset_hash_changes_with_content() -> None:
    assert dataset_hash(["a", "b"]) != dataset_hash(["a", "b", "c"])


def test_config_hash_is_key_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_content() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_adapter_hash_is_deterministic_over_content(tmp_path: Path) -> None:
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"identical weights")
    path_b.write_bytes(b"identical weights")
    assert adapter_hash(f"file://{path_a}") == adapter_hash(f"file://{path_b}")


def test_adapter_hash_changes_with_content(tmp_path: Path) -> None:
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"weights v1")
    path_b.write_bytes(b"weights v2")
    assert adapter_hash(f"file://{path_a}") != adapter_hash(f"file://{path_b}")
