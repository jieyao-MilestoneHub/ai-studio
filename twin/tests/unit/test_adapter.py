"""AdapterManifest — the checkpoint contract train writes and agent reads.
SPEC.md §7.4/§7.5."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from twin.core.adapter import AdapterManifest, ModelSpec


def _manifest(**overrides: object) -> AdapterManifest:
    defaults: dict[str, object] = dict(
        run_id="run-001",
        adapter_uri="file:///data/adapters/run-001/adapter.safetensors",
        model_spec=ModelSpec(base_model_id="some-org/some-8b-model", base_model_revision="main"),
        seed=42,
        dataset_hash="a" * 64,
        config_hash="b" * 64,
        global_step=1000,
        created_at=datetime(2026, 8, 27, tzinfo=UTC),
    )
    defaults.update(overrides)
    return AdapterManifest(**defaults)  # type: ignore[arg-type]


def test_adapter_manifest_constructs() -> None:
    manifest = _manifest()
    assert manifest.run_id == "run-001"
    assert manifest.model_spec.base_model_id == "some-org/some-8b-model"


def test_adapter_manifest_is_frozen() -> None:
    manifest = _manifest()
    with pytest.raises(ValidationError):
        manifest.global_step = 2000  # type: ignore[misc]
