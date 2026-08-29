"""The `/短劇` data model: a screenplay in, a resumable render state on disk.

Three disciplines carry over from the rest of `core`:

1. **Authoring carries indices and frame counts, never time.** A `DramaShot`
   has an `index` and a `frames` count from the beat template; where it
   starts in the finished minute exists only once `render.timeline` lays the
   segments out, and that number lives in `offsets.json`, not here. Same rule
   as `CaptionCue`.
2. **The anchor rule is a validator, not a hope.** The whole point of the
   feature is that the lead's face survives six independent generations. The
   appearance string is therefore *required verbatim* in every keyframe prompt
   -- a screenplay that drops or paraphrases it does not validate, so a model
   that "improves" the description on shot four fails at conversion instead
   of at the fourth H3 clip, twenty GPU-minutes later.
3. **The rhythm is a template, not a model choice.** Six equal ten-second
   shots is what the first dramas were, and it reads as a slideshow: no
   hook, five cuts a minute, every cut the same weight. `BEAT_TEMPLATE`
   fixes six beats of unequal length, and the shots that are long enough
   carry a second sub-shot the model cuts to itself -- H3's own multi-shot
   prompt keeps the person consistent across a cut far better than two
   separate generations do. The screenwriter fills the slots; it cannot
   change them.

`DramaState` is the per-job resume point (`runs/drama/<token>/state.json`):
every fetched artifact is recorded with its sha256 before the stage moves on,
so a lease end or a worker restart re-renders only what is missing. That is
the `clips.json` idea from docs/architecture.md, scoped to one job.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from ai_studio.core.enums import TransitionReason
from ai_studio.editing.rhythm import PacingPolicy
from pydantic import BaseModel, ConfigDict, Field, model_validator

FPS = 24
FRAME_GRID = 17
MIN_FRAMES = 124
MAX_FRAMES = 362
"""H3's frame grid: 17k+5, from 124 (the validated floor) to 362 (the
measured ceiling, `pipeline.convert_worker`). Restated here so the template
can be checked at import without a pipeline import (core sits below it)."""


class Beat(str, Enum):
    """The six-beat skeleton every drama is cut to. Named so the screenwriter
    is asked for a *turn*, not "beat 4"."""

    HOOK = "hook"
    SETUP = "setup"
    CONFLICT = "conflict"
    TURN = "turn"
    PAYOFF = "payoff"
    CLIFFHANGER = "cliffhanger"


class Framing(str, Enum):
    """Closed shot-size vocabulary. An unknown framing fails at build time --
    the alternation rule below is only worth anything over a closed set."""

    WIDE = "wide"
    MEDIUM = "medium"
    MEDIUM_CLOSE_UP = "medium close-up"
    CLOSE_UP = "close-up"
    OVER_THE_SHOULDER = "over-the-shoulder"
    TWO_SHOT = "two-shot"


WIDE_FRAMINGS = frozenset({Framing.WIDE, Framing.TWO_SHOT})
"""Framings whose keyframe must be allowed to leave the portrait behind."""


@dataclass(frozen=True)
class BeatSlot:
    beat: Beat
    frames: int
    internal_cut_frames: int | None
    """Where the model cuts to the second sub-shot, or None for one sub-shot."""

    @property
    def sub_shots(self) -> int:
        return 1 if self.internal_cut_frames is None else 2

    @property
    def duration_s(self) -> float:
        return self.frames / FPS


BEAT_TEMPLATE: tuple[BeatSlot, ...] = (
    BeatSlot(Beat.HOOK, 175, 60),          # 7.3 s, cut at 2.5 s: the first cut is the hook
    BeatSlot(Beat.SETUP, 243, 132),        # 10.1 s, cut at 5.5 s
    BeatSlot(Beat.CONFLICT, 209, None),    # 8.7 s, one held shot
    BeatSlot(Beat.TURN, 277, 144),         # 11.5 s, cut at 6.0 s
    BeatSlot(Beat.PAYOFF, 243, 108),       # 10.1 s, cut at 4.5 s
    BeatSlot(Beat.CLIFFHANGER, 192, None),  # 8.0 s, one held shot
)
"""[speculative] 1339 frames = 55.8 s; shot-length CV 0.155 (upstream's
metronome floor is 0.11); ten segments, so ~11 visual events a minute
before any caption change. All six lengths sit on the 17k+5 grid and under
the 294-frame step the community guide puts drift risk above -- except the
turn, which gets 277 because a reversal needs room to land."""

SHOT_COUNT = len(BEAT_TEMPLATE)
TOTAL_FRAMES = sum(s.frames for s in BEAT_TEMPLATE)
DURATION_BAND_S = (55.0, 65.0)
HOOK_CUT_MAX_S = 2.5

for _slot in BEAT_TEMPLATE:
    assert (_slot.frames - 5) % FRAME_GRID == 0, f"{_slot.beat}: {_slot.frames} is off the 17k+5 grid"
    assert MIN_FRAMES <= _slot.frames <= MAX_FRAMES, f"{_slot.beat}: {_slot.frames} frames out of range"
    assert _slot.internal_cut_frames is None or 0 < _slot.internal_cut_frames < _slot.frames
assert DURATION_BAND_S[0] * FPS <= TOTAL_FRAMES <= DURATION_BAND_S[1] * FPS

DRAMA_PACING = PacingPolicy(min_s=2.0, warn_s=8.0, fail_s=12.5, total_band_s=DURATION_BAND_S)
"""[speculative] The band the ten segments must sit in. Upstream's Shorts
numbers (4.0 / 6.5 s) are for talking-head explainers; a drama beat needs
longer to land, and the held conflict shot at 8.7 s is a slow shot by design
(one warning, never two in a row)."""


class CharacterAnchor(BaseModel):
    """The lead, described once, precisely, and never re-described."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    appearance: str = Field(
        min_length=12,
        description=(
            "Concrete, stable facial and hair features -- age, ethnicity, face "
            "shape, a distinguishing mark, hair colour/length/style. No mood "
            "adjectives. Pasted verbatim into every prompt."
        ),
    )
    wardrobe: str = Field(min_length=3, description="What they wear for the whole drama.")
    voice: str = Field(default="", description="How they sound, for dialogue identity.")


class WorldBible(BaseModel):
    """The place, fixed once. Pasted verbatim into every prompt, like the
    anchor: six generations that each re-imagine the light drift as surely
    as six that each re-imagine the face."""

    model_config = ConfigDict(frozen=True)

    location: str = Field(min_length=8, description="The setting and its layout, one sentence.")
    light: str = Field(min_length=4, description="Light direction and colour temperature.")
    signature_prop: str = Field(min_length=3, description="One object that is in every shot.")

    def prefix(self) -> str:
        return f"{self.location.rstrip('.')}, {self.light.rstrip('.')}, {self.signature_prop.rstrip('.')} in frame"


class DramaLine(BaseModel):
    """One spoken line. Mirrors `prompts.h3.Dialogue` without importing it
    (core sits below prompts); `prompts.drama` converts."""

    model_config = ConfigDict(frozen=True)

    speaker_id: str = Field(pattern=r"^S\d+$")
    identity: str
    language: str = "Mandarin Chinese"
    text: str = Field(min_length=1)


class SubShot(BaseModel):
    """One framing inside a generated clip. Two of them and H3 cuts between
    them itself, at the template's `internal_cut_frames`."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1, le=2)
    framing: Framing
    action: str = Field(min_length=1, description="What the lead visibly does, ending in a holdable pose.")
    camera: str | None = Field(default=None, description="A rendered `prompts.h3.camera_phrase`, or None.")
    dialogue: tuple[DramaLine, ...] = ()

    @model_validator(mode="after")
    def _one_line(self) -> SubShot:
        if len(self.dialogue) > 1:
            raise ValueError("at most one line per sub-shot")
        return self


class DramaShot(BaseModel):
    """One beat: one Flux keyframe, then one H3 clip of `frames` frames."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1, le=SHOT_COUNT)
    beat: Beat
    frames: int
    scene: str = Field(min_length=1, description="Setting, light, props -- what is around the lead.")
    sub_shots: tuple[SubShot, ...] = Field(min_length=1, max_length=2)
    keyframe_prompt: str = Field(
        min_length=1,
        description="The Flux prompt for the still first frame; must contain the anchor verbatim.",
    )
    cut_reason: TransitionReason = Field(
        default=TransitionReason.DEFAULT, description="What the cut *into* this shot means."
    )

    @property
    def slot(self) -> BeatSlot:
        return BEAT_TEMPLATE[self.index - 1]

    @property
    def duration_s(self) -> float:
        return self.frames / FPS

    @property
    def internal_cut_s(self) -> float | None:
        cut = self.slot.internal_cut_frames
        return None if cut is None or len(self.sub_shots) < 2 else cut / FPS

    def segment_frames(self) -> tuple[int, ...]:
        """How many frames each sub-shot holds the screen."""
        cut = self.slot.internal_cut_frames
        if cut is None or len(self.sub_shots) < 2:
            return (self.frames,)
        return (cut, self.frames - cut)

    # Delegates so the status page and older callers keep reading one framing.
    @property
    def framing(self) -> Framing:
        return self.sub_shots[0].framing

    @property
    def action(self) -> str:
        return self.sub_shots[0].action

    @property
    def dialogue(self) -> tuple[DramaLine, ...]:
        return tuple(line for s in self.sub_shots for line in s.dialogue)

    @model_validator(mode="after")
    def _check(self) -> DramaShot:
        slot = BEAT_TEMPLATE[self.index - 1]
        if self.beat is not slot.beat:
            raise ValueError(f"shot {self.index} is the {slot.beat.value}, not {self.beat.value}")
        if self.frames != slot.frames:
            raise ValueError(f"shot {self.index} must be {slot.frames} frames, got {self.frames}")
        if len(self.sub_shots) != slot.sub_shots:
            raise ValueError(f"shot {self.index} ({slot.beat.value}) has {slot.sub_shots} sub-shot(s), got {len(self.sub_shots)}")
        if [s.index for s in self.sub_shots] != list(range(1, len(self.sub_shots) + 1)):
            raise ValueError(f"shot {self.index}: sub-shot indices must be 1..n in order")
        return self


class Screenplay(BaseModel):
    """What the screenwriter produced and what the render consumes."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    logline: str = Field(min_length=1)
    style: str = Field(default="Live-action, cinematic")
    anchor: CharacterAnchor
    world: WorldBible
    shots: tuple[DramaShot, ...]
    overall_soundscape: str = Field(min_length=1)
    non_diegetic_music: str = "N/A"

    def sub_shots(self) -> list[tuple[DramaShot, SubShot]]:
        return [(shot, sub) for shot in self.shots for sub in shot.sub_shots]

    @model_validator(mode="after")
    def _check(self) -> Screenplay:
        if len(self.shots) != SHOT_COUNT:
            raise ValueError(f"a drama has exactly {SHOT_COUNT} shots, got {len(self.shots)}")
        indices = [s.index for s in self.shots]
        if indices != list(range(1, SHOT_COUNT + 1)):
            raise ValueError(f"shot indices must be 1..{SHOT_COUNT} in order: {indices}")
        for shot in self.shots:
            if self.anchor.appearance not in shot.keyframe_prompt:
                raise ValueError(
                    f"shot {shot.index}: keyframe prompt does not contain the anchor "
                    f"verbatim ({self.anchor.appearance!r}) -- the face would drift"
                )
            if self.world.prefix() not in shot.keyframe_prompt:
                raise ValueError(f"shot {shot.index}: keyframe prompt does not contain the world bible verbatim")
        flat = self.sub_shots()
        for (a_shot, a), (b_shot, b) in pairwise(flat):
            if a.framing is b.framing:
                raise ValueError(
                    f"shot {a_shot.index}/{a.index} and shot {b_shot.index}/{b.index} are both "
                    f"{a.framing.value}; consecutive framings must differ"
                )
        pushes = [shot.index for shot, sub in flat if sub.camera and "pushes in" in sub.camera]
        if len(pushes) > 1:
            raise ValueError(f"one push-in per drama, reserved for the turn; found on shots {pushes}")
        if pushes and BEAT_TEMPLATE[pushes[0] - 1].beat is not Beat.TURN:
            raise ValueError(f"the push-in belongs to the turn (shot {Beat.TURN.value}), not shot {pushes[0]}")
        return self


# ------------------------------------------------------------------ render state


class ArtifactRecord(BaseModel):
    """One fetched file. `sha256` is what lets a resume trust it."""

    path: str
    sha256: str
    cost_usd: float = Field(default=0.0, ge=0)
    job_id: str = ""
    created_at: str = ""
    """UTC ISO-8601 of the fetch. Default "" so a state.json written before
    2026-08-28 (which had no times at all) still loads and resumes."""


class StageTiming(BaseModel):
    """When one stage of a drama began and ended -- the only record of how a
    15-30 minute, multi-window render actually spent its time."""

    started_at: str = ""
    finished_at: str = ""


class DramaState(BaseModel):
    """Stage bookkeeping for one drama. Mutable; rewritten after every artifact.

    Keys are shot indices as strings (JSON object keys). `face_repair`
    records what actually happened on this pod -- "applied", "skipped: ...",
    or "failed: ..." -- so a result can be read back honestly.
    """

    character: dict[str, ArtifactRecord] = Field(default_factory=dict)
    """`"front"` and `"three_quarter"` sheet images."""

    keyframes: dict[str, ArtifactRecord] = Field(default_factory=dict)
    clips: dict[str, ArtifactRecord] = Field(default_factory=dict)
    leveled: dict[str, ArtifactRecord] = Field(default_factory=dict)
    output: ArtifactRecord | None = None

    face_repair: str = "pending"
    plan_gate: str = "pending"
    """"passed", "passed with N warning(s)" or "failed: ..." -- the PRE gate's
    verdict on the screenplay, taken before any GPU-second."""
    spent_usd: float = Field(default=0.0, ge=0)
    ffmpeg_argv: list[list[str]] = Field(default_factory=list)
    """Every ffmpeg invocation, literally -- the render manifest."""

    created_at: str = ""
    updated_at: str = ""
    stages: dict[str, StageTiming] = Field(default_factory=dict)
    """`character` | `keyframes` | `clips` | `level` | `assemble` -> when. A
    stage that resumes across windows keeps its first `started_at`. A state
    file from before 2026-08-29 says `concat` for the last one; it loads."""

    def stage_start(self, name: str, now: str) -> None:
        timing = self.stages.setdefault(name, StageTiming())
        if not timing.started_at:
            timing.started_at = now

    def stage_finish(self, name: str, now: str) -> None:
        self.stages.setdefault(name, StageTiming()).finished_at = now

    def add_cost(self, cost_usd: float) -> None:
        self.spent_usd = round(self.spent_usd + max(0.0, cost_usd), 6)
