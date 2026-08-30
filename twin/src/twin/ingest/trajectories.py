"""LINE chat history -> SPEC.md §4.10 `Trajectory` records (Phase 4's slim
trajectory set, PLAN.md Phase 4 "仍待辦 2").

One trajectory per *counterpart burst*: consecutive messages from the other
party with no gap over `burst_gap` (empirically p90 of same-sender gaps is
2 min on the real data; 5 min is the guard). That burst is the stimulus
(§4.3 `exposure.stimulus`) and, with the preceding `context_messages` turns,
the `observation`. Then:

- the principal's own next burst starts within `reply_window` and before the
  counterpart writes again -> one `ActionStep(surface="line")` whose content
  is that burst (§11-A/D28: `train.formatting` later folds it into a `reply`
  tool call);
- otherwise -> one explicit `NoActionStep` (§4.10/D6 — silence is a sample,
  never data absence). `reply_window` is chosen from the observed latency
  distribution by the caller (p90 = 120 min on the real data), not guessed
  here, and is recorded in the build manifest.

Exposure evidence (§4.3, §4.11, D7/D20): a LINE export carries no read
receipt (SPEC.md §11 open item H), so the only honest values are
`inferred` — the principal sent *something* in this room later, which on
LINE requires opening the room where the unanswered burst is visible — and
`absent` when they never did. `absent` MUST be `negative_class=trivial`
(validator in `core.trajectory`); `inferred` no-replies are labelled `hard`
**at the lower confidence §4.3 explicitly grants historical data** ("歷史時段
的 S3 負例只能以較低信心使用，並於報告中標示") — this whole corpus is for
training only; per §4.3 LINE MUST NOT be an S3 evaluation source until
item H is verified, and every report built on it must say so.

`split` is decided here, at build time, from the *same* cutoffs the fragment
store was ingested under (§4.8/D21) — the caller passes them and they are
written to the manifest so drift is detectable. `ground_truth_source` is
`observed` throughout (D25: nothing here is teacher-synthesized). No
`ReflectionStep` is ever emitted (§5.4 is a Phase 10 decision;
`train.formatting` raises on one).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass
from twin.core.trajectory import ActionStep, Exposure, NoActionStep, Step, Trajectory
from twin.ingest.sources.line import LineMessage
from twin.ingest.split import decide_split

V1_TOOLS: tuple[str, ...] = ("recall", "web_search", "reply")  # SPEC.md §6.3, in table order


@dataclass(frozen=True)
class TrajectoryBuildParams:
    train_cutoff: datetime
    sealed_cutoff: datetime
    burst_gap: timedelta = timedelta(minutes=5)
    reply_window: timedelta = timedelta(minutes=120)
    context_messages: int = 8
    # `inferred` exposure only counts when the principal's next activity in
    # this room falls within this horizon after the stimulus; later than that
    # we cannot honestly say they had seen it at decision time (§2.3: exposure
    # is the *precondition*), so the no-reply drops to `absent`/`trivial`.
    exposure_horizon: timedelta = timedelta(hours=24)
    # A reply that arrives after `reply_window` is, at the stimulus tick, a
    # no_action (tick-loop semantics, §6.2/D28) — or the stimulus can be
    # skipped entirely if "late reply" is judged to be neither class (§2.3
    # does not decide it). Recorded in the build manifest either way.
    late_reply: Literal["no_action", "skip"] = "no_action"


def _bursts(messages: list[LineMessage], gap: timedelta) -> list[list[LineMessage]]:
    out: list[list[LineMessage]] = []
    for m in messages:
        if out and out[-1][-1].sender == m.sender and m.sent_at - out[-1][-1].sent_at <= gap:
            out[-1].append(m)
        else:
            out.append([m])
    return out


def _coalesce_counterpart(
    bursts: list[list[LineMessage]], principal: str, window: timedelta
) -> list[list[LineMessage]]:
    """Adjacent counterpart bursts with no principal message between them and
    a gap within `window` are ONE stimulus: "ping ... ping again 40 min later
    ... reply" is a single decision, not a no_action followed by an action
    (data-hygiene 2026-08-29: ~22% of hard negatives were this shape)."""
    out: list[list[LineMessage]] = []
    for b in bursts:
        if out and out[-1][0].sender != principal and b[0].sender != principal and b[0].sent_at - out[-1][-1].sent_at <= window:
            out[-1] = out[-1] + b
        else:
            out.append(b)
    return out


# LINE's export writes a media message as the bare word 貼圖/圖片/影片. Left as-is,
# the principal's sticker replies become literal text targets and the adapter
# learns to *say* "圖片" (T v1, 2026-08-30: "圖片\n圖片" answers). Marked the
# same way the parser already marks recalled messages (`[已收回訊息]`): the
# brackets separate "sent a sticker" from the word, cost 3-4 Qwen3 tokens
# (vs 6 for `[media:sticker]`), and stay Chinese-native. SPEC.md §4.2
# (multimodal reduced to text). Applied here, not in `sources.line`, so the
# fragment store — whose fragment_ids the frozen S1 bank references — is
# untouched; only trajectories (rebuilt 2026-08-30) carry the marker.
MEDIA_PLACEHOLDERS: dict[str, str] = {"貼圖": "[貼圖]", "圖片": "[圖片]", "影片": "[影片]"}


def _media_marked(content: str) -> str:
    return MEDIA_PLACEHOLDERS.get(content.strip(), content)


def _render(messages: list[LineMessage], *, principal: str) -> str:
    return "\n".join(f"{'我' if m.sender == principal else m.sender}: {_media_marked(m.content)}" for m in messages)


def trajectories_from_line_messages(
    messages: list[LineMessage],
    *,
    principal_id: str,
    principal_display_name: str,
    params: TrajectoryBuildParams,
) -> Iterator[Trajectory]:
    ordered = sorted(messages, key=lambda m: m.sent_at)
    bursts = _coalesce_counterpart(_bursts(ordered, params.burst_gap), principal_display_name, params.reply_window)
    principal_times = sorted(m.sent_at for m in ordered if m.sender == principal_display_name)

    for index, burst in enumerate(bursts):
        if burst[0].sender == principal_display_name:
            continue
        stimulus_end = burst[-1].sent_at
        # Preceding context: the last `context_messages` messages before this burst.
        prior = [m for b in bursts[:index] for m in b][-params.context_messages :]
        observation = _render(prior + burst, principal=principal_display_name)
        stimulus = _render(burst, principal=principal_display_name)

        nxt = bursts[index + 1] if index + 1 < len(bursts) else None
        replied = (
            nxt is not None
            and nxt[0].sender == principal_display_name
            and nxt[0].sent_at - stimulus_end <= params.reply_window
        )
        steps: list[Step]
        if replied:
            assert nxt is not None
            steps = [ActionStep(surface="line", content="\n".join(_media_marked(m.content) for m in nxt))]
            evidence = ExposureEvidence.INFERRED
            negative = NegativeClass.NONE
        else:
            late_reply = nxt is not None and nxt[0].sender == principal_display_name
            if late_reply and params.late_reply == "skip":
                continue
            next_activity = next((t for t in principal_times if t > stimulus_end), None)
            seen_within_horizon = next_activity is not None and next_activity - stimulus_end <= params.exposure_horizon
            evidence = ExposureEvidence.INFERRED if seen_within_horizon else ExposureEvidence.ABSENT
            negative = NegativeClass.HARD if seen_within_horizon else NegativeClass.TRIVIAL
            reason = (
                f"no reply within {int(params.reply_window.total_seconds() // 60)} min; "
                + (
                    f"principal active in this room within {int(params.exposure_horizon.total_seconds() // 3600)}h (exposure inferred)"
                    if seen_within_horizon
                    else "no principal activity in this room within the exposure horizon"
                )
            )
            steps = [NoActionStep(reason=reason)]

        yield Trajectory(
            principal_id=principal_id,
            context_time=stimulus_end,
            split=decide_split(stimulus_end, train_cutoff=params.train_cutoff, sealed_cutoff=params.sealed_cutoff),
            exposure=Exposure(occurred=True, stimulus=stimulus, evidence=evidence),
            observation=observation,
            available_tools=list(V1_TOOLS),
            steps=steps,
            negative_class=negative,
            ground_truth_source=GroundTruthSource.OBSERVED,
        )
