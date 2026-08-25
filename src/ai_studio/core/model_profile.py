"""What each generation model can be asked for, in one place.

`ProviderCapabilities` describes *one configured instance* — this canvas, this
hourly rate, this expected latency. A `ModelProfile` describes the **model**:
which canvases exist at all, which frame counts are legal, which weight file
belongs to which quantisation. The distinction matters because the second kind
of fact was, until this module, scattered across up to four unlinked places.

Frame count was the worst case. `124` appeared as a literal in both H3 workflow
JSONs, again in `pipeline/convert_worker.py` as `124 / 24`, and a third time in
`pipeline/drain.py` as a `max()` floor — while
`ProviderCapabilities.clip_duration_quantum`, the field designed to express
exactly this rule, was set to `None`. The rule that produced 124 lived only in a
docstring. Nothing cross-checked any pair of them, and nothing does today: the
CLI's `--seconds 5.0` default yields 120 frames, which is **not a legal length**,
and it submits happily.

This lives in `core` for the same reason `ProviderCapabilities` does, and
`CLAUDE.md` says not to move that: it is what lets `editing`, `planner` and
`gates` reason about a model without importing a backend.

Every number here is graded. 📏 means we measured it; `[verified]` means it was
read out of the model's or ComfyUI's own source; `[reported]` means someone else
measured it; `[speculative]` means it was inferred. See `docs/attribution.md`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ai_studio.core.errors import UnknownKeyError


class FrameGrid(BaseModel):
    """The legal frame counts for a video model.

    MiniMax H3 accepts lengths where ``n % step == base`` — for H3 that is
    ``17k + 5``: 5, 22, 39, 56, 73, 90, 107, 124, 141 … `[verified]` by reading
    ComfyUI's `comfy_extras/nodes_minimax_h3.py`, which snaps an out-of-grid
    request *upward* to the next legal value.

    We do not rely on that snap. A request that needed snapping is a request
    whose duration is not what the caller thought it was, and the caller is the
    only one who can decide whether that matters.
    """

    model_config = ConfigDict(frozen=True)

    base: int = Field(ge=0)
    step: int = Field(gt=0)
    minimum: int = Field(gt=0)
    """The model's own floor. For H3 this is 5 `[verified]`, not 124."""

    maximum: int = Field(gt=0)

    recommended_minimum: int | None = None
    """Below this the output is legal but not necessarily good.

    For H3 the turbo LoRA's card gives a trained range of 124-362 frames
    `[reported]` — a property of *that adapter*, not of the model. Recorded
    separately from `minimum` so the two are never confused again.
    """

    def is_valid(self, frames: int) -> bool:
        return self.minimum <= frames <= self.maximum and frames % self.step == self.base

    def snap(self, frames: int) -> int:
        """The smallest legal length at or above `frames`.

        Snapping up rather than to-nearest matches what ComfyUI does, so a
        caller that snaps deliberately gets the same answer the pod would have
        arrived at silently.
        """
        candidate = max(frames, self.minimum)
        while candidate % self.step != self.base:
            candidate += 1
        if candidate > self.maximum:
            raise ValueError(
                f"{frames} frames snaps to {candidate}, past the {self.maximum} maximum"
            )
        return candidate

    def neighbours(self, frames: int) -> tuple[int, int]:
        """The legal lengths either side of `frames`, for error messages."""
        below = frames - ((frames - self.base) % self.step)
        while below > self.minimum and not self.is_valid(below):
            below -= self.step
        return max(below, self.minimum), self.snap(frames)


class Canvas(BaseModel):
    """One resolution a model will accept, and what it costs there."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    native: bool = False
    """True only for a canvas the model was actually trained at.

    864x480 is *legal* for H3 — both dimensions are multiples of 32 — but its
    short edge is 480 where H3's native short edge is 768 `[verified]`. It was
    described as a native canvas in this project's docs for some time; it is
    not, and a quality question about it is not a bug report.
    """

    latency_s: float | None = None
    """Seconds to generate at this canvas, or None when nobody has measured it.

    None on purpose rather than a plausible default: an unmeasured canvas
    silently inheriting another's timing is how a cost estimate becomes fiction.
    """

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


class LoraSpec(BaseModel):
    """A LoRA the workflow is expected to load."""

    model_config = ConfigDict(frozen=True)

    filename: str
    strength: float = Field(ge=0.0, le=2.0)
    source_repo: str
    notes: str = ""


class ModelProfile(BaseModel):
    """Everything about a model that is true regardless of how it is deployed."""

    model_config = ConfigDict(frozen=True)

    key: str
    produces_video: bool

    canvases: tuple[Canvas, ...]
    dimension_multiple: int = Field(default=32, gt=0)
    """Both dimensions must be a multiple of this. 32 for H3, 16 for Flux —
    Flux packs its 8x-downsampled latent into 2x2 patches `[verified]`."""

    max_pixels: int | None = None

    fps: int | None = None
    frame_grid: FrameGrid | None = None

    default_steps: int = Field(gt=0)
    has_native_audio: bool = False
    supports_negative_prompt: bool = False
    max_prompt_chars: int = Field(gt=0)

    weights: dict[str, str] = Field(default_factory=dict)
    """quantisation -> the filename the workflow must load."""

    lora: LoraSpec | None = None

    # ------------------------------------------------------------- canvases

    def canvas(self, width: int, height: int) -> Canvas | None:
        return next(
            (c for c in self.canvases if c.width == width and c.height == height), None
        )

    def require_canvas(self, width: int, height: int) -> Canvas:
        """The canvas, or raise. Never guesses.

        Guessing is what the old `MEASURED_LATENCY_S.get((w, h), 300.0)` did:
        an off-table canvas quietly inherited 1344x768's timing, so its cost
        estimate and its job timeout were both someone else's numbers.
        """
        found = self.canvas(width, height)
        if found is None:
            raise UnknownKeyError(
                f"{self.key} canvas", f"{width}x{height}", [c.label for c in self.canvases]
            )
        return found

    @property
    def native_canvas(self) -> Canvas:
        """The canvas the model was trained at. Used when nobody chose one."""
        return next((c for c in self.canvases if c.native), self.canvases[0])

    # --------------------------------------------------------------- frames

    def frames_for(self, duration_s: float) -> int:
        """A legal frame count for roughly `duration_s`.

        The only supported way to turn a duration into frames. Snapping happens
        here, visibly, so that `submit` can refuse anything off-grid without
        also having to be the thing that fixes it.
        """
        if self.frame_grid is None or self.fps is None:
            raise ValueError(f"{self.key} has no frame grid; it does not produce video")
        return self.frame_grid.snap(round(duration_s * self.fps))

    def duration_for(self, frames: int) -> float:
        if self.fps is None:
            raise ValueError(f"{self.key} has no fps")
        return frames / self.fps

    @property
    def shortest_useful_duration_s(self) -> float:
        """The shortest clip worth generating.

        The adapter's recommended floor where there is one, otherwise the
        model's own minimum. Callers wanting "just make the short one" ask for
        this rather than spelling out a frame count — which is how `124` came
        to exist in four places at once.
        """
        if self.frame_grid is None:
            raise ValueError(f"{self.key} has no frame grid; it does not produce video")
        floor = self.frame_grid.recommended_minimum or self.frame_grid.minimum
        return self.duration_for(floor)


# --------------------------------------------------------------------- H3

MINIMAX_H3 = ModelProfile(
    key="minimax-h3",
    produces_video=True,
    canvases=(
        # 768px short edge is H3's native size [verified]. Local ComfyUI tops
        # out here — the 2K path is MiniMax's hosted regeneration module, which
        # is not open source.
        Canvas(width=1344, height=768, native=True, latency_s=300.0),
        Canvas(width=1280, height=736, latency_s=361.0),
        # Not native. Kept because it is ~2.3x faster [reported] and is what
        # this project has always defaulted to; whether that costs quality is
        # an open A/B, not a settled question.
        Canvas(width=864, height=480, latency_s=133.0),
        # Its own source calls this figure anomalous: slower than the larger
        # 864x480 while the same source's other column scales normally.
        Canvas(width=608, height=352, latency_s=182.0),
    ),
    dimension_multiple=32,
    max_pixels=1344 * 768,
    fps=24,
    frame_grid=FrameGrid(
        base=5,
        step=17,
        minimum=5,
        maximum=3600,
        recommended_minimum=124,
    ),
    default_steps=6,
    has_native_audio=True,
    supports_negative_prompt=False,
    max_prompt_chars=8000,
    weights={
        "int8": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "fp8": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    },
    lora=LoraSpec(
        filename="minimax_h3_turbo_v4_step600_ema.safetensors",
        strength=1.0,
        source_repo="larryvrh/MiniMax-H3-Turbo-Lora",
        notes=(
            "Must be driven through MiniMaxH3TurboLoRA, not the stock loader — "
            "this file's keys do not match ComfyUI's map. Enforced by "
            "comfy.validate."
        ),
    ),
)

# --------------------------------------------------------------------- Flux

FLUX_1_DEV = ModelProfile(
    key="flux-1-dev",
    produces_video=False,
    canvases=(Canvas(width=1024, height=1024, native=True, latency_s=None),),
    dimension_multiple=16,
    max_pixels=None,
    fps=None,
    frame_grid=None,
    default_steps=28,
    has_native_audio=False,
    # Structural, not a preference: BasicGuider takes one conditioning input and
    # has nowhere to put a negative. CFGGuider could, but ComfyUI skips the
    # uncond pass entirely at cfg 1.0, and above 1.0 costs a second forward pass
    # per step on a model that is guidance-distilled. [verified]
    supports_negative_prompt=False,
    max_prompt_chars=2000,
    weights={"fp8": "flux1-dev.safetensors"},
    lora=LoraSpec(
        filename="flux_nsfw_uncensored_v1.safetensors",
        strength=1.0,
        source_repo="Heartsync/Flux-NSFW-uncensored",
        notes=(
            "diffusers/PEFT key layout, which ComfyUI maps natively for Flux "
            "(comfy/lora.py registers 'transformer.{key}'). Renamed from "
            "lora.safetensors on the pod."
        ),
    ),
)


PROFILES: dict[str, ModelProfile] = {
    MINIMAX_H3.key: MINIMAX_H3,
    FLUX_1_DEV.key: FLUX_1_DEV,
}


def get_profile(key: str) -> ModelProfile:
    """Look up a profile, or raise. No default — an unknown model is a bug."""
    if key not in PROFILES:
        raise UnknownKeyError("model profile", key, sorted(PROFILES))
    return PROFILES[key]
