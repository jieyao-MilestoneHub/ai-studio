"""Interview transcript -> training trajectories. SPEC.md §4.10 (trajectory
schema), D19 (self-report is the dominant persona-fidelity source, so it MUST
reach the LoRA), D37 (self-report split = train), D39 (effect first: full,
un-redacted content, 2026-08-30).

Shape: every interviewer question the principal answered becomes one
trajectory — observation = the exchange so far on that coverage point plus
the question, stimulus = the question, one `ActionStep(surface="interview")`
carrying the answer verbatim. This is the same "stimulus -> the principal's
own words" shape `ingest.trajectories` derives from LINE, so
`train.formatting.trajectory_to_messages` needs no special case: the answer
becomes a masked `reply` tool call exactly like a LINE reply does.

Exposure is `READ_RECEIPT`-grade certainty (the principal typed the answer
to the question on screen) — the only place in the corpus where exposure is
not inferred. There are no negatives here: an empty answer means the
principal skipped the question, which is not "chose not to act on a
stimulus they saw in the wild" (SPEC.md §2.3) — it is dropped, never
labelled `no_action`.
"""

from __future__ import annotations

from collections.abc import Iterator

from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass
from twin.core.trajectory import ActionStep, Exposure, Trajectory
from twin.ingest.interviewer import InterviewTranscript, Turn
from twin.ingest.split import decide_self_report_split
from twin.ingest.trajectories import V1_TOOLS

INTERVIEW_SURFACE = "interview"


def _render(turns: list[Turn]) -> str:
    return "\n".join(("訪談員：" if t.speaker == "interviewer" else "本人：") + t.text for t in turns)


def trajectories_from_interview(transcript: InterviewTranscript, *, context_turns: int = 6) -> Iterator[Trajectory]:
    """One trajectory per answered interviewer turn. `context_turns` prior
    turns from the same block are kept as observation context (mirrors
    `TrajectoryBuildParams.context_messages` for LINE)."""
    turns = transcript.turns
    for index, turn in enumerate(turns):
        if turn.speaker != "interviewer" or index + 1 >= len(turns):
            continue
        answer = turns[index + 1]
        if answer.speaker != "respondent" or not answer.text.strip():
            continue
        prior = [t for t in turns[max(0, index - context_turns) : index] if t.block == turn.block]
        yield Trajectory(
            principal_id=transcript.principal_id,
            context_time=turn.at,
            split=decide_self_report_split(),
            exposure=Exposure(occurred=True, stimulus=turn.text, evidence=ExposureEvidence.READ_RECEIPT),
            observation=_render([*prior, turn]),
            available_tools=list(V1_TOOLS),
            steps=[ActionStep(surface=INTERVIEW_SURFACE, content=answer.text)],
            negative_class=NegativeClass.NONE,
            ground_truth_source=GroundTruthSource.OBSERVED,
        )
