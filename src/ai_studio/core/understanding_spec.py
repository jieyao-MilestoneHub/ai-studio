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

from pydantic import BaseModel, ConfigDict, Field

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
    """False for Qwen3-Omni-Captioner (audio) -- it rejects a text prompt
    outright. Enforced by `UnderstandingRequest` validation, not just
    documented: a request with a prompt against a capability that says False
    raises rather than silently dropping the prompt."""

    max_input_seconds: float | None = Field(
        default=None,
        description="Longest input this backend accepts, in seconds. None "
        "for image (not applicable) or where no ceiling is known yet.",
    )
    max_output_chars: int = Field(
        default=1000,
        gt=0,
        description="Ceiling on the returned description, so it fits LINE's "
        "MAX_TEXT_CHARS (bots/line/push.py) with room for the wrapper text.",
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
    """The question put to the model. Built at conversion
    (`prompts.understanding`): the engineered default for a bare trigger, or
    the user's own trailing text rewritten into the model's best form. None
    means "no question" -- the server's caption path for a photo."""

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific knobs. Kept out of the typed surface on purpose.",
    )


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


class UnderstandingAsset(BaseModel):
    """A produced description. No output file, so no `sha256`/`size_bytes`/
    dimensions -- the same reason `ImageProviderCapabilities` drops fps and
    duration for a still image."""

    model_config = ConfigDict(frozen=True)

    shot_id: str
    provider: str
    job_id: str
    modality: MediaKind
    result_text: str
    cost_usd: float = Field(default=0.0, ge=0)
