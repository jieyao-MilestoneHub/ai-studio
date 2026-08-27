"""The checkpoint contract. SPEC.md §7.4/§7.5 — train (L3) writes this, agent
(L4) reads it to know which weights to load, and never needs to import
`twin.train` internals to do so. This is the dependency-inversion artifact
PLAN.md §3.3 names but doesn't spell out; it exists specifically so the
existing "C2/C3 - serving does not depend on trainer internals" import-linter
contract has something concrete to hold agent to.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelSpec(BaseModel):
    """SPEC.md §5.1: base model MUST be open-weight, permissive-license, and
    shared across every twin — individual differences live only in the
    adapter. This records *which* base model an adapter was trained against,
    not a choice of its own (§11 item G leaves the size/identity open)."""

    model_config = ConfigDict(frozen=True)

    base_model_id: str
    base_model_revision: str


class AdapterManifest(BaseModel):
    """SPEC.md §7.4's checkpoint contract (adapter weights, optimizer state, LR
    schedule, RNG state, global_step, dataloader cursor) plus §7.5's
    reproducibility binding (seed/dataset_hash/config_hash). Everything past
    `adapter_uri` and `model_spec` is metadata *about* the checkpoint that
    train.py's actual checkpoint directory holds — this manifest is what
    agent reads to find and identify the checkpoint, not the checkpoint
    itself. See core.hashing for how dataset_hash/config_hash are computed."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    adapter_uri: str  # fsspec URI, SPEC.md §7.2
    model_spec: ModelSpec
    seed: int
    dataset_hash: str
    config_hash: str
    global_step: int
    created_at: datetime
