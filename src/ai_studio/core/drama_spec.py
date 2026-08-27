"""The `/短劇` data model: a screenplay in, a resumable render state on disk.

Two disciplines carry over from the rest of `core`:

1. **Authoring carries indices, never time.** A `DramaShot` has an `index`;
   its start time exists only once the six clips are concatenated, and that
   number lives in the render manifest, not here. Same rule as `CaptionCue`.
2. **The anchor rule is a validator, not a hope.** The whole point of the
   feature is that the lead's face survives six independent generations. The
   appearance string is therefore *required verbatim* in every keyframe prompt
   -- a screenplay that drops or paraphrases it does not validate, so a model
   that "improves" the description on shot four fails at conversion instead
   of at the fourth H3 clip, twenty GPU-minutes later.

`DramaState` is the per-job resume point (`runs/drama/<token>/state.json`):
every fetched artifact is recorded with its sha256 before the stage moves on,
so a lease end or a worker restart re-renders only what is missing. That is
the `clips.json` idea from docs/architecture.md, scoped to one job.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_studio.core.enums import TransitionReason

SHOT_COUNT = 6
"""Six shots of the default clip length (~10 s) is the one-minute drama.

Fixed rather than chosen by the screenwriter: a shot longer than ~10 s is
where the community guide puts H3's drift risk, and a shot much shorter than
that pays the same text-encoder cost for less story. Six is what fits."""


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


class DramaLine(BaseModel):
    """One spoken line. Mirrors `prompts.h3.Dialogue` without importing it
    (core sits below prompts); `prompts.drama` converts."""

    model_config = ConfigDict(frozen=True)

    speaker_id: str = Field(pattern=r"^S\d+$")
    identity: str
    language: str = "Mandarin Chinese"
    text: str = Field(min_length=1)


class DramaShot(BaseModel):
    """One of the six shots. Rendered as one Flux keyframe, then one H3 clip."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1, le=SHOT_COUNT)
    scene: str = Field(min_length=1, description="Setting, light, props -- what is around the lead.")
    framing: str = Field(
        min_length=1, description="wide / medium / close-up / over-the-shoulder ...",
    )
    action: str = Field(min_length=1, description="The one thing that happens in this shot.")
    keyframe_prompt: str = Field(
        min_length=1,
        description="The Flux prompt for the still first frame; must contain the anchor verbatim.",
    )
    camera: str | None = Field(
        default=None, description="A rendered `prompts.h3.camera_phrase`, or None for static."
    )
    dialogue: tuple[DramaLine, ...] = ()
    cut_reason: TransitionReason = TransitionReason.DEFAULT


class Screenplay(BaseModel):
    """What the screenwriter produced and what the render consumes."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    logline: str = Field(min_length=1)
    style: str = Field(default="Live-action, cinematic")
    anchor: CharacterAnchor
    shots: tuple[DramaShot, ...]
    overall_soundscape: str = Field(min_length=1)
    non_diegetic_music: str = "N/A"

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
        return self


# ------------------------------------------------------------------ render state


class ArtifactRecord(BaseModel):
    """One fetched file. `sha256` is what lets a resume trust it."""

    path: str
    sha256: str
    cost_usd: float = Field(default=0.0, ge=0)
    job_id: str = ""


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
    spent_usd: float = Field(default=0.0, ge=0)
    ffmpeg_argv: list[list[str]] = Field(default_factory=list)
    """Every ffmpeg invocation, literally -- the render manifest."""

    def add_cost(self, cost_usd: float) -> None:
        self.spent_usd = round(self.spent_usd + max(0.0, cost_usd), 6)
