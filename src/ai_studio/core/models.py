"""The data model every layer agrees on.

Two disciplines are load-bearing here and both come from video-autopilot-kit:

1. **Authoring models carry `segment_id` references, never timestamps.**
   `CaptionCue` has no `start`/`end`. Absolute time is computed in exactly one
   place (`render.timeline.resolve_timeline`) and written to exactly one file
   (`offsets.json`). Upstream shipped captions that drifted 2-3 seconds out of
   sync because segments were split by hand; binding to an index instead of a
   time makes that class of bug structurally impossible rather than merely
   detectable.

2. **Semantic names, not effect names.** A `Scene` declares a
   `TransitionReason`, not a `TransitionKind`. The mapping is one table. You
   cannot write "put a wipe here" without first saying what the wipe means.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_studio.core.enums import (
    CaptionKind,
    FormatStrategy,
    GenMode,
    SceneMode,
    Severity,
    SourceKind,
    TransitionReason,
)

# ------------------------------------------------------------------ authoring


class VideoSpec(BaseModel):
    """What the user asked for. Written verbatim to `spec.json`."""

    model_config = ConfigDict(frozen=True)

    topic: str
    language: str = "zh-TW"
    target_platform: str = "yt_longform_1080p"
    target_duration_s: float = Field(default=30.0, gt=0)

    style_key: str | None = None
    provider: str = "stub"
    seed: int | None = None

    max_cost_usd: float = Field(
        default=5.0,
        ge=0,
        description="Hard ceiling checked before generation, not after.",
    )


class Shot(BaseModel):
    """One generated (or sourced) picture element.

    `subcuts` is how a fixed-length model clip participates in fast pacing: H3
    emits 5s with no cut inside it, so the planner marks in-clip cut offsets
    and the clip becomes several `Segment`s. See docs/editing-grammar.md,
    conflict 2.
    """

    model_config = ConfigDict(frozen=True)

    shot_id: str
    scene_id: str
    index: int = Field(ge=0)

    source_kind: SourceKind = SourceKind.GENERATED
    mode: GenMode = GenMode.T2V

    prompt: str = ""
    negative_prompt: str | None = None
    duration_s: float = Field(gt=0)

    motion_semantic: str | None = Field(
        default=None,
        description=(
            "Ken Burns / push-in semantic. Must stay None for GENERATED shots "
            "unless explicitly waived: model output already contains camera "
            "motion, and layering a push-in on top double-moves the frame."
        ),
    )

    subcuts: tuple[float, ...] = Field(
        default=(),
        description="In-clip cut offsets in seconds, relative to the shot start.",
    )

    ref_image_path: str | None = None

    @model_validator(mode="after")
    def _check_subcuts(self) -> Shot:
        if any(c <= 0 or c >= self.duration_s for c in self.subcuts):
            raise ValueError(
                f"subcuts must lie strictly inside (0, {self.duration_s}): {self.subcuts}"
            )
        if list(self.subcuts) != sorted(self.subcuts):
            raise ValueError(f"subcuts must be ascending: {self.subcuts}")
        if self.source_kind is SourceKind.GENERATED and self.motion_semantic is not None:
            raise ValueError(
                f"shot {self.shot_id}: motion_semantic on a GENERATED shot double-moves "
                "the frame. Use SourceKind.STILL, or record a waiver in the plan."
            )
        return self

    @property
    def segment_count(self) -> int:
        return len(self.subcuts) + 1


class Scene(BaseModel):
    """A narrative beat. Owns its shots and its rhythm mode."""

    model_config = ConfigDict(frozen=True)

    scene_id: str
    index: int = Field(ge=0)
    semantic: str
    mode: SceneMode = SceneMode.FAST

    intent: str = ""
    narration: str = ""
    chapter_boundary: bool = False
    transition_reason: TransitionReason = TransitionReason.DEFAULT

    shots: tuple[Shot, ...] = ()

    @property
    def duration_s(self) -> float:
        return sum(s.duration_s for s in self.shots)


class Segment(BaseModel):
    """The atomic timeline unit. One shot yields `len(subcuts) + 1` of them.

    Captions bind here, by id.
    """

    model_config = ConfigDict(frozen=True)

    segment_id: str
    shot_id: str
    scene_id: str
    subcut_index: int = Field(ge=0)
    intended_duration_s: float = Field(gt=0)


class CaptionCue(BaseModel):
    """A caption bound to a segment — deliberately with no time fields.

    `color_key` is a key, not a colour. Resolution happens against a registry
    that raises on an unknown key rather than falling back to white.
    """

    model_config = ConfigDict(frozen=True)

    cue_id: str
    segment_id: str
    text: str
    kind: CaptionKind = CaptionKind.MAIN
    color_key: str = "w"

    @property
    def char_count(self) -> int:
        return len(self.text)


# ------------------------------------------------------------------ derived


class FormatPlan(BaseModel):
    """How native output becomes the delivery canvas. Derived, never authored."""

    model_config = ConfigDict(frozen=True)

    strategy: FormatStrategy
    target_name: str

    native_width: int = Field(gt=0)
    native_height: int = Field(gt=0)
    target_width: int = Field(gt=0)
    target_height: int = Field(gt=0)

    crop_width: int = Field(gt=0)
    crop_height: int = Field(gt=0)
    scale_width: int = Field(gt=0)
    scale_height: int = Field(gt=0)
    crop_x: int = Field(ge=0)
    crop_y: int = Field(ge=0)

    upscale_factor: float = Field(gt=0)
    area_retained: float = Field(gt=0, le=1.0)
    backdrop_blur_sigma: int | None = None

    waived: bool = False
    waiver_reason: str | None = None


class GateFinding(BaseModel):
    """One rule result. Carries its own source link so a reader can check us."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    severity: Severity
    message: str
    where: str | None = None
    observed: str | None = None
    expected: str | None = None
    source_url: str | None = None


class GateReport(BaseModel):
    """Every gate returns this shape. Written to `runs/<id>/gates/<gate>.json`."""

    model_config = ConfigDict(frozen=True)

    gate: str
    findings: tuple[GateFinding, ...] = ()
    counters: dict[str, float] = Field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def failures(self) -> tuple[GateFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.FAIL)

    @property
    def warnings(self) -> tuple[GateFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARN)

    @property
    def passed(self) -> bool:
        return not self.failures


class RenderResult(BaseModel):
    """The delivered artifact. `result.json`.

    The delivery-facing fields (`output_uri`, `poster_uri`) are populated from
    day one so a delivery channel is additive.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    output_uri: str
    poster_uri: str | None = None

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    size_bytes: int = Field(ge=0)
    has_audio: bool = False

    measured_lufs: float | None = None
    measured_true_peak: float | None = None

    cost_usd: float = Field(default=0.0, ge=0)
    waivers: tuple[str, ...] = ()
