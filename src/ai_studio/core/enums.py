"""Closed vocabularies.

Note the naming discipline inherited from video-autopilot-kit: scene sheets and
shot plans accept **semantic** names, never effect names. `TransitionReason` is
what an author writes; `TransitionKind` is what the renderer picks. The mapping
lives in one table, which forces the author to think about meaning first and
makes "why is there a wipe here" answerable.
"""

from __future__ import annotations

from enum import Enum


class SceneMode(str, Enum):
    """Dual-mode rhythm wave.

    A video that is entirely one mode is flat. The pace gate enforces that no
    three consecutive scenes share a mode.
    """

    FAST = "fast"
    """Explanation mode: a visual event every 3-5s. A still frame >4s fails."""

    FOCUS = "focus"
    """Demo mode: may hold up to 40s without a cut. A hold >40s fails."""


class SourceKind(str, Enum):
    """Where a shot's picture comes from.

    This drives whether motion may be applied. AI-generated clips already
    contain camera movement; adding a push-in on top double-moves the frame.
    See docs/editing-grammar.md, conflict 1.
    """

    GENERATED = "generated"
    """Model output. Already has camera motion. Motion defaults to None."""

    STILL = "still"
    """A static image. The legitimate home for sub-pixel Ken Burns."""

    ASSET = "asset"
    """Pre-existing footage supplied by the user."""


class GenMode(str, Enum):
    """Generation modes a provider may support."""

    T2V = "t2v"
    I2V = "i2v"
    KEYFRAME = "keyframe"
    """First/last frame conditioning (MiniMax H3 fl2va)."""

    REF2V = "ref2v"
    """Reference-driven generation for character consistency (H3 ref2va)."""

    T2I = "t2i"
    I2I = "i2i"
    """Image-to-image: a source photo re-rendered under the prompt (Flux)."""
    """Text-to-image (Flux). No frame-count or fps concept applies."""


class MediaKind(str, Enum):
    """The job-dispatch discriminator: what a job's provider produces --
    video, image -- or consumes and describes -- a photo, an audio clip, a
    video clip. One shared FIFO queue and one `providers_for()` dict are
    dispatched on this single field, so every kind gets exactly one provider,
    prompt builder (generation kinds only), and asset shape. See
    `core/understanding_spec.py` for why the *request/job/asset* types for
    the three understanding kinds are still kept as siblings of
    `core/provider_spec.py`/`core/image_provider_spec.py` rather than merged
    into them -- a description has no width, height, fps, or duration."""

    VIDEO = "video"
    IMAGE = "image"
    IMAGE_UNDERSTAND = "image_understand"
    """/說圖: describe a photo. Backed by moondream3 (or Florence-2)."""

    AUDIO_UNDERSTAND = "audio_understand"
    """/說音: describe/transcribe an audio clip. Backed by
    Qwen3-Omni-Captioner -- no text prompt accepted, audio capped at
    `Settings.max_audio_understand_s`."""

    VIDEO_UNDERSTAND = "video_understand"
    """/說影: describe a video clip. Backed by Tarsier2."""

    CHAT = "chat"
    """/himonkey: a plain-text LLM reply, backed by gpt-oss-20b. Deliberately
    *not* in `_UNDERSTANDING_KINDS` -- `is_understanding` specifically means
    "describes attached media", and chat describes nothing. It shares the
    understanding side's GPU slot (see `pipeline.drain.make_room_for`), but
    that is a VRAM-residency fact, not a semantic one, so call sites that
    need it join with an explicit `or job_kind is MediaKind.CHAT` rather than
    being folded into this property."""

    @property
    def is_understanding(self) -> bool:
        return self in _UNDERSTANDING_KINDS


_UNDERSTANDING_KINDS = frozenset(
    {MediaKind.IMAGE_UNDERSTAND, MediaKind.AUDIO_UNDERSTAND, MediaKind.VIDEO_UNDERSTAND}
)


class JobState(str, Enum):
    """Clip-generation job lifecycle.

    Deliberately mirrors RunPod's serverless envelope so that a self-hosted
    ComfyUI job and a RunPod endpoint job are the same shape to the pipeline.
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES

    @property
    def is_success(self) -> bool:
        return self is JobState.COMPLETED


_TERMINAL_STATES = frozenset(
    {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMED_OUT}
)


class FormatStrategy(str, Enum):
    """How native model output is mapped onto a delivery canvas."""

    NATIVE = "native"
    """Passthrough. No resampling at all."""

    FILL_CROP = "fill_crop"
    """Scale up, then centre-crop. Scale first, crop second — never reversed."""

    HYBRID_PAD = "hybrid_pad"
    """Crop toward the target aspect, scale, then pad over a blurred backdrop.

    The escape hatch for aspect changes too large to crop through, e.g. a 1.8
    source onto a 0.5625 vertical canvas.
    """


class Severity(str, Enum):
    """Gate finding severity."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class TransitionReason(str, Enum):
    """What a transition *means*. Authors write these; renderers map them."""

    TOPIC_CHANGE = "topic_change"
    TIME_PASSING = "time_passing"
    DRILL_DOWN = "drill_down"
    DEFAULT = "default"


class TransitionKind(str, Enum):
    """What a transition *is*. Chosen by the semantic table, not by the author."""

    HARD_CUT = "hard_cut"
    """>=90% of all splices. The default, and the only one with no cap."""

    WHIP = "whip"
    WIPE = "wipe"
    ZOOM_PUNCH = "zoom_punch"
    DISSOLVE = "dissolve"


class CaptionKind(str, Enum):
    """Caption presentation modes. Kinetic kinds are capped at 40% of cues."""

    MAIN = "main"
    HOOK = "hook"
    SUB = "sub"
    IMPACT = "impact"
    RIBBON = "ribbon"
    FLOAT_LEFT = "float_left"
    FLOAT_RIGHT = "float_right"

    @property
    def is_kinetic(self) -> bool:
        return self in _KINETIC_KINDS


_KINETIC_KINDS = frozenset(
    {
        CaptionKind.IMPACT,
        CaptionKind.RIBBON,
        CaptionKind.FLOAT_LEFT,
        CaptionKind.FLOAT_RIGHT,
    }
)


class AudioLayer(str, Enum):
    """The four-layer dB ladder. Voice is always loudest; everything else moves."""

    VOICE = "voice"
    SFX = "sfx"
    AMBIENCE = "ambience"
    BGM = "bgm"
