"""The understanding-provider contract. Sibling to `provider_spec.py` and
`image_provider_spec.py`, not a merge into either.

A description has no width, height, fps, or duration to produce -- it
*consumes* a photo/audio/video and produces text. Forcing it through
`ClipRequest`/`ImageRequest` (which require output dimensions) would mean
inventing values that mean nothing, which is exactly the silent-degrade
pattern this codebase avoids everywhere else (see `core/errors.py`). These
types mirror their clip/image counterparts field-for-field except for
whatever is output-media-shaped.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_studio.core.enums import JobState, MediaKind


class UnderstandingCapabilities(BaseModel):
    """What an understanding backend can actually do.

    One instance per modality (`IMAGE_UNDERSTAND`/`AUDIO_UNDERSTAND`/
    `VIDEO_UNDERSTAND`), not one shared instance -- the three backing models
    differ in whether they accept a steering prompt and how long an input
    they tolerate, and neither should be guessed at a call site.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    model_id: str
    modality: MediaKind

    accepts_prompt: bool = False
    """Whether the backend takes a steering question at all. Enforced by
    `check_prompt`, not just documented: a request with a prompt against a
    capability that says False raises rather than silently dropping it."""

    max_input_seconds: float | None = Field(
        default=None,
        description="Longest input this backend accepts, in seconds. None "
        "for image (not applicable) or where no ceiling is known yet.",
    )
    max_output_chars: int = Field(
        default=1000,
        gt=0,
        description="Ceiling on each returned answer. The caller's delivery "
        "channel decides the number; 1000 is a conservative default.",
    )

    cost_per_call_usd: float = Field(default=0.0, ge=0)
    expected_latency_s: float = Field(default=30.0, gt=0)
    max_concurrent_jobs: int = Field(
        default=1,
        gt=0,
        description="Kept at 1: one 24GB card holds at most one of "
        "{ComfyUI's resident checkpoint, an understanding model} at a time.",
    )

    def check_prompt(self, prompt: str | None) -> None:
        """Raise if `prompt` is given but this backend cannot use one.

        Fail loudly rather than silently drop it: a dropped prompt is a
        request that ran but not the one the caller asked for.
        """
        if prompt and not self.accepts_prompt:
            raise ValueError(
                f"{self.provider} ({self.modality.value}) does not accept a prompt, got {prompt!r}"
            )


class UnderstandingRequest(BaseModel):
    """One description to produce. Provider-agnostic."""

    model_config = ConfigDict(frozen=True)

    shot_id: str
    modality: MediaKind
    input_media_path: str
    prompt: str | None = None
    """The question put to the model. The caller writes it -- neither this
    package nor the pod server holds a default. Required for audio and video;
    None for an image means the model's caption path (no question)."""
    audio_prompt: str | None = None
    """Video only: the question for the second model, which listens to the
    extracted audio track. Required for video, forbidden elsewhere."""

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific knobs. Kept out of the typed surface on purpose.",
    )

    @model_validator(mode="after")
    def _questions_match_the_modality(self) -> UnderstandingRequest:
        if self.modality is MediaKind.VIDEO_UNDERSTAND:
            if not self.prompt or not self.audio_prompt:
                raise ValueError("a video request needs both prompt and audio_prompt")
        elif self.modality is MediaKind.AUDIO_UNDERSTAND:
            if not self.prompt:
                raise ValueError("an audio request needs a prompt")
            if self.audio_prompt:
                raise ValueError("audio_prompt is only for video requests")
        elif self.audio_prompt:
            raise ValueError("audio_prompt is only for video requests")
        return self


class UnderstandingJob(BaseModel):
    """A submitted understanding job. Same lifecycle shape as `ClipJob`/`ImageJob`."""

    model_config = ConfigDict(frozen=True)

    provider: str
    job_id: str
    shot_id: str
    state: JobState = JobState.PENDING

    submitted_at: float
    updated_at: float
    queue_position: int | None = None

    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.updated_at - self.submitted_at)

    def with_state(self, state: JobState, *, now: float, **changes: Any) -> UnderstandingJob:
        return self.model_copy(update={"state": state, "updated_at": now, **changes})


class VideoSections(BaseModel):
    """What the video modality returns: one answer per model. Presenting
    them -- headings, the note for a silent clip -- is the caller's."""

    model_config = ConfigDict(frozen=True)

    visual: str
    audio: str | None
    """None when the clip carries no audio track (`has_audio_track` False),
    so the audio model was never run."""
    has_audio_track: bool


class UnderstandingAsset(BaseModel):
    """A produced description. No output file, so no `sha256`/`size_bytes`/
    dimensions -- the same reason `ImageProviderCapabilities` drops fps and
    duration for a still image.

    Exactly one of `result_text` (image, audio) and `sections` (video) is
    set."""

    model_config = ConfigDict(frozen=True)

    shot_id: str
    provider: str
    job_id: str
    modality: MediaKind
    result_text: str | None = None
    sections: VideoSections | None = None
    truncated: bool = False
    """An answer hit the server's or this side's character ceiling."""
    cost_usd: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _one_shape(self) -> UnderstandingAsset:
        if (self.result_text is None) == (self.sections is None):
            raise ValueError("exactly one of result_text and sections must be set")
        return self
