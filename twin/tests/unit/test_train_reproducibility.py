"""`run_id` binding. SPEC.md §7.5."""

from __future__ import annotations

from twin.train.reproducibility import derive_run_id


def test_derive_run_id_is_deterministic() -> None:
    first = derive_run_id(seed=1, dataset_hash="a" * 64, config_hash="b" * 64)
    second = derive_run_id(seed=1, dataset_hash="a" * 64, config_hash="b" * 64)
    assert first == second


def test_derive_run_id_changes_with_seed() -> None:
    a = derive_run_id(seed=1, dataset_hash="a" * 64, config_hash="b" * 64)
    b = derive_run_id(seed=2, dataset_hash="a" * 64, config_hash="b" * 64)
    assert a != b


def test_derive_run_id_changes_with_dataset_hash() -> None:
    a = derive_run_id(seed=1, dataset_hash="a" * 64, config_hash="b" * 64)
    b = derive_run_id(seed=1, dataset_hash="c" * 64, config_hash="b" * 64)
    assert a != b


def test_derive_run_id_changes_with_config_hash() -> None:
    a = derive_run_id(seed=1, dataset_hash="a" * 64, config_hash="b" * 64)
    b = derive_run_id(seed=1, dataset_hash="a" * 64, config_hash="d" * 64)
    assert a != b


def test_derive_run_id_has_a_stable_prefix() -> None:
    run_id = derive_run_id(seed=1, dataset_hash="a" * 64, config_hash="b" * 64)
    assert run_id.startswith("run_")
