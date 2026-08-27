"""Load training trajectories. SPEC.md §4.8/D21 — the hard filter PLAN.md
§3.5 names as the target of an un-skippable CI test: "訓練管線 MUST 於載入時
硬性過濾 split != train 的樣本，且此過濾 MUST 有測試覆蓋."

`twin.train` MUST NOT decide `split` itself — see `ingest.split.decide_split`,
the only place that happens. This function's entire job is refusing anything
that isn't already labeled `train`. Reading is delegated to `ingest.store`
(train legitimately imports ingest — trajectories are constructed there,
this is expected, not a layering violation; see twin/PLAN.md §3.2's own note
on this exact question).
"""

from __future__ import annotations

from collections.abc import Iterator

from twin.core.enums import Split
from twin.core.trajectory import Trajectory
from twin.ingest.store import read_trajectories_jsonl


def load_training_examples(trajectories_uri: str) -> Iterator[Trajectory]:
    """Yields only `split == train` trajectories from `trajectories_uri`.
    Anything `heldout` or `sealed` MUST NOT reach this point — SPEC.md §4.8's
    named failure mode if it does is silent, undetectable time leakage."""
    for trajectory in read_trajectories_jsonl(trajectories_uri):
        if trajectory.split == Split.TRAIN:
            yield trajectory
