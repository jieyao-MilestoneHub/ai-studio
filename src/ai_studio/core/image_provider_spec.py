"""The image-provider contract. Sibling to `provider_spec.py`, not a merge.

A still image has no frame-count, no fps, no duration — forcing it through
`ClipRequest`/`ClipAsset` (which require `fps`/`duration_s` > 0) would mean
inventing values that mean nothing, which is exactly the silent-degrade
pattern this codebase avoids everywhere else (see `core/errors.py`). These
types mirror their clip counterparts field-for-field except for whatever is
clip-length-shaped.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_studio.core.enums import GenMode, JobState


class ImageProviderCapabilities(BaseModel):
    """What an image backend can actually do."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model_id: str

    native_width: int = Field(gt=0)
    native_height: int = Field(gt=0)

    modes: frozenset[GenMode]
    supports_seed: bool = True
    supports_negative_prompt: bool = False
    max_prompt_chars: int = Field(default=2000, gt=0)

    output_format: str = "png"

    cost_per_image_usd: float = Field(default=0.0, ge=0)
    expected_latency_s: float = Field(default=30.0, gt=0)
    max_concurrent_jobs: int = Field(default=1, gt=0)

    @property
    def native_aspect(self) -> float:
        return self.native_width / self.native_height

    def supports(self, mode: GenMode) -> bool:
        return mode in self.modes


class ImageRequest(BaseModel):
    """One image to generate. Provider-agnostic."""

    model_config = ConfigDict(frozen=True)

    shot_id: str
    mode: GenMode = GenMode.T2I

    prompt: str
    negative_prompt: str | None = None

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    seed: int | None = None
    steps: int | None = None

    source_image_path: str | None = None
    """Local path of the photo to re-render (`GenMode.I2I`). The image
    counterpart of `ClipRequest.first_frame_path`; None is plain text-to-image."""

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific knobs. Kept out of the typed surface on purpose.",
    )


class ImageJob(BaseModel):
    """A submitted image-generation job. Same lifecycle shape as `ClipJob`."""

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

    def with_state(self, state: JobState, *, now: float, **changes: Any) -> ImageJob:
        return self.model_copy(update={"state": state, "updated_at": now, **changes})


class ImageAsset(BaseModel):
    """A generated image that has been fetched into our own storage."""

    model_config = ConfigDict(frozen=True)

    shot_id: str
    key: str
    sha256: str
    size_bytes: int = Field(ge=0)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str = "png"

    provider: str
    job_id: str
    cost_usd: float = Field(default=0.0, ge=0)

    @property
    def aspect(self) -> float:
        return self.width / self.height
