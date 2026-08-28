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
    """What a model serves: the key a provider is registered under. A
    provider produces video or image, or consumes and describes a photo, an
    audio clip or a video clip, or answers a chat turn. Whoever submits work
    keys its own job types onto these; this package knows only the models.
    See `core/understanding_spec.py` for why the *request/job/asset* types
    for the three understanding kinds are siblings of
    `core/provider_spec.py`/`core/image_provider_spec.py` rather than merged
    -- a description has no width, height, fps, or duration."""

    VIDEO = "video"
    """MiniMax H3 through ComfyUI."""
    IMAGE = "image"
    """Flux.1-dev through ComfyUI."""
    IMAGE_UNDERSTAND = "image_understand"
    """Describe a photo: moondream3 on the inference server."""
    AUDIO_UNDERSTAND = "audio_understand"
    """Describe an audio clip: Qwen2-Audio-7B-Instruct on the inference server."""
    VIDEO_UNDERSTAND = "video_understand"
    """Describe a video clip: Qwen2.5-VL-7B-Instruct on the frames, then
    Qwen2-Audio on the extracted track -- two answers, see
    `understanding_spec.VideoSections`."""
    CHAT = "chat"
    """A plain-text reply: gpt-oss-20b on the inference server. The same
    model is the prompt rewriter (`pipeline.pod_llm`)."""


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
