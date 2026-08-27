"""Time-based split decision. SPEC.md §4.8/D21 — the only place `split` is decided.

`Fragment.split` is written once here, at ingest, and is read-only after that
(`core.fragment.Fragment` is frozen). This module MUST NOT be imported by
`twin.train` — moving this logic to train time is undetectable time leakage:
every metric still looks normal (SPEC.md §4.8's own stated failure symptom).
"""

from __future__ import annotations

from datetime import datetime

from twin.core.enums import Split


def decide_split(event_time: datetime, *, train_cutoff: datetime, sealed_cutoff: datetime) -> Split:
    """Everything before `train_cutoff` is train; from there up to (not including)
    `sealed_cutoff` is heldout; from `sealed_cutoff` onward is sealed.

    SPEC.md §2.5: sealed is carved out of heldout, opened only at final acceptance
    (EVAL.md §9's 20%-reserved-split row) — so it is deliberately the *latest*
    slice, the one furthest from train and hardest to have leaked into it.
    """
    if sealed_cutoff < train_cutoff:
        raise ValueError(
            f"sealed_cutoff ({sealed_cutoff.isoformat()}) is before train_cutoff "
            f"({train_cutoff.isoformat()}) — sealed MUST be carved out of heldout, "
            f"not overlap train (SPEC.md §2.5)"
        )
    if event_time < train_cutoff:
        return Split.TRAIN
    if event_time < sealed_cutoff:
        return Split.HELDOUT
    return Split.SEALED


def sealed_cutoff_for(
    *, train_cutoff: datetime, now: datetime, sealed_fraction: float = 0.2
) -> datetime:
    """The boundary that makes the most recent `sealed_fraction` of the heldout
    window (train_cutoff..now) sealed. EVAL.md §9: "eval set MUST 有 20% 保留分割"."""
    if not 0.0 <= sealed_fraction < 1.0:
        raise ValueError(f"sealed_fraction must be in [0, 1), got {sealed_fraction}")
    if now < train_cutoff:
        raise ValueError(
            f"now ({now.isoformat()}) is before train_cutoff ({train_cutoff.isoformat()})"
        )
    heldout_span = now - train_cutoff
    return now - heldout_span * sealed_fraction
