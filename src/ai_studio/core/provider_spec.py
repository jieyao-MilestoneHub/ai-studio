"""The clip-provider contract.

**This module lives in `core`, not in `providers`, and that placement is the
whole trick.** `editing.format_policy` and `planner` need to reason about the
model's native size, clip-length quantum, and supported modes — but if they
imported a provider to ask, swapping the model would ripple through the editing
layer. Instead a provider publishes a `ProviderCapabilities` snapshot into
`provider_manifest.json`, and everyone downstream reads the snapshot.

Dependency inverted: nothing above `core` needs to know a provider exists.

Shared shape across the four spec files (`provider_spec`,
`image_provider_spec`, `understanding_spec`, `chat_spec`), so a reader of
any one knows the others: a `*Capabilities` model with `native_aspect` /
`supports` where the output has a canvas; a `*Request`; a `*Job` with
`is_terminal` (state will not change again), `elapsed_s` (submit to last
poll) and `with_state` (an immutable copy with the new state and timestamp);
and a `*Asset` with `aspect` where there is an image. The methods are
undocumented in the other three files on purpose -- they mean the same
thing everywhere.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_studio.core.enums import GenMode, JobState


class ProviderCapabilities(BaseModel):
    """What a clip backend can actually do.

    Consumers, none of which import `providers`:

    - `editing.format_policy` reads `native_width/height` to derive the delivery
      transform and decide whether a target is reachable at all.
    - `planner` reads `min/max_clip_s` and `clip_duration_quantum` to refuse a
      7s shot against a 5s-quantum model before anything is submitted.
    - `pipeline` reads `cost_per_second_usd` to enforce the cost ceiling, and
      `expected_latency_s` / `max_concurrent_jobs` to size its poll loop.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    model_id: str

    native_width: int = Field(gt=0)
    native_height: int = Field(gt=0)
    native_fps: int = Field(gt=0)

    modes: frozenset[GenMode]
    min_clip_s: float = Field(gt=0)
    max_clip_s: float = Field(gt=0)
    clip_duration_quantum: float | None = Field(
        default=None,
        description="If set, the model only emits multiples of this length (H3: 5.0).",
    )

    has_native_audio: bool = False
    supports_seed: bool = True
    supports_negative_prompt: bool = True
    max_prompt_chars: int = Field(default=4000, gt=0)

    output_container: str = "mp4"
    output_codec: str = "h264"

    url_ttl_s: int | None = Field(
        default=None,
        description=(
            "Seconds before a returned output URL expires. RunPod public "
            "endpoints expire at 604800 (7 days); a self-hosted pod writes to "
            "local disk that dies with the pod, so treat None as 'fetch now'."
        ),
    )

    cost_per_second_usd: float = Field(default=0.0, ge=0)
    expected_latency_s: float = Field(default=180.0, gt=0)
    max_concurrent_jobs: int = Field(
        default=1,
        gt=0,
        description=(
            "Keep at 1 for GPU-bound generation. Running two diffusion jobs on "
            "one GPU time-slices the same silicon and risks OOM; scale out with "
            "more workers instead."
        ),
    )

    @model_validator(mode="after")
    def _check_clip_bounds(self) -> ProviderCapabilities:
        if self.min_clip_s > self.max_clip_s:
            raise ValueError(f"min_clip_s {self.min_clip_s} > max_clip_s {self.max_clip_s}")
        return self

    @property
    def native_aspect(self) -> float:
        return self.native_width / self.native_height

    def supports(self, mode: GenMode) -> bool:
        return mode in self.modes

    def estimated_cost_usd(self, clip_seconds: float) -> float:
        return round(self.cost_per_second_usd * clip_seconds, 4)


class ClipRequest(BaseModel):
    """One clip to generate. Provider-agnostic."""

    model_config = ConfigDict(frozen=True)

    shot_id: str
    mode: GenMode = GenMode.T2V

    prompt: str
    negative_prompt: str | None = None

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    fps: int = Field(gt=0)

    seed: int | None = None
    steps: int | None = None

    ref_image_path: str | None = None
    first_frame_path: str | None = None
    last_frame_path: str | None = None

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific knobs. Kept out of the typed surface on purpose.",
    )


class ClipJob(BaseModel):
    """A submitted generation job.

    Serialized into `clips.json` after every state change. That file is the
    resume point that matters: one H3 clip is 2-6 minutes of GPU time, so
    re-running generation because assembly crashed is the expensive failure
    mode. On resume the pipeline reattaches to jobs still in flight rather than
    paying for them twice.
    """

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

    def with_state(self, state: JobState, *, now: float, **changes: Any) -> ClipJob:
        return self.model_copy(update={"state": state, "updated_at": now, **changes})


class ClipAsset(BaseModel):
    """A generated clip that has been fetched into our own storage.

    `sha256` is what lets `resume` verify a previously generated clip is still
    intact before reusing it instead of regenerating.
    """

    model_config = ConfigDict(frozen=True)

    shot_id: str
    key: str
    sha256: str
    size_bytes: int = Field(ge=0)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    duration_s: float = Field(gt=0)
    has_audio: bool = False

    provider: str
    job_id: str
    cost_usd: float = Field(default=0.0, ge=0)
    peak_vram_gb: float | None = None
    """Peak VRAM ComfyUI's own `/system_stats` reported while this job was in
    flight (see `comfy/jobs.py::poll_job`). None when the backend never
    exposed it (the stub provider) or a sample failed -- a benchmark metric
    missing one point is fine; a render that fails because of one is not."""

    @property
    def aspect(self) -> float:
        return self.width / self.height
