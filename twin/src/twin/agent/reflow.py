"""Veto-to-hard-negative reflow. SPEC.md §6.5: "每一次本人的『不送出』MUST 被記錄
為硬負例並回流資料集" — every time the principal declines to send a drafted
reply, that MUST become a new hard-negative trajectory.

This only ever *constructs* a new `Trajectory` via the normal
`ingest.split.decide_split` path — it MUST NOT decide `split` itself, and
MUST NOT mutate any existing record (data-contract skill rule 5): if L4 could
write `split` or edit history, "today's runtime" would silently change
tomorrow's train/test boundary, undetectably.
"""

from __future__ import annotations

from datetime import datetime

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass
from twin.core.trajectory import Exposure, NoActionStep, Trajectory
from twin.ingest.split import decide_split


def reflow_veto(
    *,
    principal_id: str,
    context_time: datetime,
    observation: str,
    available_tools: list[str],
    train_cutoff: datetime,
    sealed_cutoff: datetime,
) -> Trajectory:
    """The drafted reply's own content is deliberately not stored anywhere on
    this record: it was a rejected candidate, not something that happened —
    keeping it (e.g. as an `action` step) would misrepresent a corrected
    `no_action` ground truth as containing a positive action example.

    `evidence=read_receipt` here: a veto means the principal directly
    reviewed the draft in the L0 review UI and declined it — stronger, more
    direct confirmation of exposure to the triggering observation than a
    platform read-receipt would even be, and `read_receipt` is the closest
    category SPEC.md §4.3's vocabulary offers for "confirmed by direct
    interaction" rather than inferred after the fact."""
    split = decide_split(context_time, train_cutoff=train_cutoff, sealed_cutoff=sealed_cutoff)
    return Trajectory(
        principal_id=principal_id,
        context_time=context_time,
        split=split,
        exposure=Exposure(occurred=True, stimulus=observation, evidence=ExposureEvidence.READ_RECEIPT),
        observation=observation,
        available_tools=available_tools,
        steps=[NoActionStep(reason="principal declined to send the drafted reply")],
        negative_class=NegativeClass.HARD,
        ground_truth_source=GroundTruthSource.OBSERVED,
    )
