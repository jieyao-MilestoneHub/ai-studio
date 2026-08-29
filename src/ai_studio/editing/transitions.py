"""Transitions by meaning, not by effect (editing grammar section 3).

Authors write a `TransitionReason`; this module picks the `TransitionKind`.
That one indirection is what stops transitions accumulating as decoration:
there is no way to *ask* for a whip pan, only to say a topic changed, and the
table decides what a topic change looks like.

Frequency caps (section 3.2) live here, in the module that applies them:

- hard cuts are >= 90% of all splices and have no cap;
- `wipe` <= 3 per video, `zoom_punch` <= 2; the cap turns the excess back
  into hard cuts rather than refusing the plan;
- a transition that needs shot-pair evidence (section 3.3: whip needs real
  whip motion in both shots, a wipe needs >70% foreground occlusion, a zoom
  needs a detail target) and has none is *downgraded* to a hard cut and says
  so in `downgraded_from`. With generated clips there is no evidence source
  yet, so in practice only `dissolve` survives the table today.

Durations are the upstream kit's, quoted as `[reported]`; they land on whole
frames at 24 and 30 fps, which is not an accident. The audio dip at a hard
cut is ours, `[speculative]`: three frames each side is enough to hide the
ambience jump between two independently generated soundtracks without
reading as a fade.

Pure: no I/O, no ffmpeg. `render` turns a `Transition` into a filter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai_studio.core.enums import Severity, TransitionKind, TransitionReason
from ai_studio.core.errors import UnknownKeyError
from ai_studio.core.models import GateFinding

SOURCE_URL = "https://github.com/Hao0321/video-autopilot-kit"

WHIP_S = 0.30
WIPE_S = 0.50
ZOOM_PUNCH_S = 0.33
DISSOLVE_S = 0.50
"""[reported] section 3.2. 0.5 s is 12 frames at 24 fps, 15 at 30."""

AUDIO_DIP_S = 0.125
"""[speculative] Audio crossfade across a hard cut between two separately
generated clips: 3 frames at 24 fps. The picture still cuts hard."""

MAX_WIPES = 3
MAX_ZOOM_PUNCH = 2
MIN_HARD_CUT_RATIO = 0.90

DURATION_S: dict[TransitionKind, float] = {
    TransitionKind.HARD_CUT: 0.0,
    TransitionKind.WHIP: WHIP_S,
    TransitionKind.WIPE: WIPE_S,
    TransitionKind.ZOOM_PUNCH: ZOOM_PUNCH_S,
    TransitionKind.DISSOLVE: DISSOLVE_S,
}

SEMANTIC_TABLE: dict[TransitionReason, TransitionKind] = {
    TransitionReason.TOPIC_CHANGE: TransitionKind.WIPE,
    TransitionReason.TIME_PASSING: TransitionKind.DISSOLVE,
    TransitionReason.DRILL_DOWN: TransitionKind.ZOOM_PUNCH,
    TransitionReason.DEFAULT: TransitionKind.HARD_CUT,
}
"""Section 3.1. The only place a reason becomes a kind."""


@dataclass(frozen=True)
class Evidence:
    """What the shot pair around a splice can be shown to contain.

    Nothing measures these yet for generated clips; every field defaults to
    False, which is what makes the downgrade rule bite.
    """

    whip_motion: bool = False
    occlusion: bool = False
    detail_target: bool = False


_NEEDS: dict[TransitionKind, str] = {
    TransitionKind.WHIP: "whip_motion",
    TransitionKind.WIPE: "occlusion",
    TransitionKind.ZOOM_PUNCH: "detail_target",
}


@dataclass(frozen=True)
class Transition:
    """One splice, decided. `overlap_s` is how much the two clips overlap on
    the output timeline (a dissolve shortens the piece by that much); a hard
    cut overlaps nothing and only dips the audio."""

    kind: TransitionKind
    reason: TransitionReason
    overlap_s: float
    audio_fade_s: float
    downgraded_from: TransitionKind | None = None

    @property
    def is_hard_cut(self) -> bool:
        return self.kind is TransitionKind.HARD_CUT


def hard_cut(reason: TransitionReason = TransitionReason.DEFAULT, *, downgraded_from: TransitionKind | None = None,
             audio_fade_s: float = AUDIO_DIP_S) -> Transition:
    return Transition(
        kind=TransitionKind.HARD_CUT, reason=reason, overlap_s=0.0,
        audio_fade_s=audio_fade_s, downgraded_from=downgraded_from,
    )


NO_EVIDENCE = Evidence()


def choose(reason: TransitionReason, evidence: Evidence = NO_EVIDENCE) -> Transition:
    """The table, then the evidence rule. Unknown reasons raise."""
    try:
        kind = SEMANTIC_TABLE[TransitionReason(reason)]
    except (KeyError, ValueError):
        raise UnknownKeyError("transition reason", reason, [r.value for r in TransitionReason]) from None
    needs = _NEEDS.get(kind)
    if needs is not None and not getattr(evidence, needs):
        return hard_cut(reason, downgraded_from=kind)
    if kind is TransitionKind.HARD_CUT:
        return hard_cut(reason)
    overlap = DURATION_S[kind]
    return Transition(kind=kind, reason=reason, overlap_s=overlap, audio_fade_s=overlap)


def plan(reasons: Sequence[TransitionReason], evidence: Sequence[Evidence] | None = None) -> list[Transition]:
    """One `Transition` per splice, in order, with the caps applied.

    The fourth wipe and the third zoom punch become hard cuts (`downgraded_from`
    set) -- the plan is kept, the decoration is not.
    """
    if evidence is not None and len(evidence) != len(reasons):
        raise ValueError(f"{len(reasons)} splices but {len(evidence)} evidence entries")
    caps = {TransitionKind.WIPE: MAX_WIPES, TransitionKind.ZOOM_PUNCH: MAX_ZOOM_PUNCH}
    used: dict[TransitionKind, int] = {k: 0 for k in caps}
    out: list[Transition] = []
    for i, reason in enumerate(reasons):
        t = choose(reason, evidence[i] if evidence is not None else NO_EVIDENCE)
        if t.kind in caps:
            if used[t.kind] >= caps[t.kind]:
                t = hard_cut(reason, downgraded_from=t.kind)
            else:
                used[t.kind] += 1
        out.append(t)
    return out


def check(transitions: Sequence[Transition]) -> list[GateFinding]:
    """Section 3.2 as findings: caps (fail) and the hard-cut ratio (warn)."""
    findings: list[GateFinding] = []
    kinds = [t.kind for t in transitions]
    for kind, cap in ((TransitionKind.WIPE, MAX_WIPES), (TransitionKind.ZOOM_PUNCH, MAX_ZOOM_PUNCH)):
        n = kinds.count(kind)
        if n > cap:
            findings.append(GateFinding(
                rule_id=f"T-CAP-{kind.value.upper()}", severity=Severity.FAIL,
                message=f"{n} {kind.value} transitions, cap is {cap}",
                observed=str(n), expected=f"<= {cap}", source_url=SOURCE_URL,
            ))
    if kinds:
        # "Only a chapter boundary earns a transition": one is always allowed,
        # then the 90% floor applies -- nine splices with one dissolve is the
        # rule working, not a 88.9% violation of it.
        non_hard = len(kinds) - kinds.count(TransitionKind.HARD_CUT)
        allowed = max(1, int(len(kinds) * (1 - MIN_HARD_CUT_RATIO)))
        if non_hard > allowed:
            ratio = 1 - non_hard / len(kinds)
            findings.append(GateFinding(
                rule_id="T-RATIO", severity=Severity.WARN,
                message=f"hard cuts are {ratio:.0%} of {len(kinds)} splices, grammar wants >= {MIN_HARD_CUT_RATIO:.0%}",
                observed=f"{non_hard} non-hard", expected=f"<= {allowed}", source_url=SOURCE_URL,
            ))
    return findings
