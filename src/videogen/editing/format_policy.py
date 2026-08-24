"""Mapping native model output onto a delivery canvas.

MiniMax H3 renders at 864x480 (aspect 1.800) or 1344x768 (aspect 1.750). The
delivery target for this project is 1920x1080 (aspect 1.778). Neither native
size matches, so something has to give — and *how much* it gives is the thing
this module refuses to leave to judgement.

Two mechanical rules, both of which reintroduce the jitter the grammar bans if
you get them wrong:

1. **Scale first, crop second. Never the reverse.** Cropping 864 down to 853.33
   to "fix" the aspect before scaling forces a non-integer, non-uniform scale
   factor. Scaling to 1944x1080 and then cropping 24 columns keeps the factor
   uniform and the geometry integral.

2. **The upscale limit applies to the foreground plane only.** In `HYBRID_PAD`
   the blurred backdrop is upscaled far past the limit, and that is fine
   *because it is blurred* — which is why a minimum blur sigma is enforced.
   Without it, "pad" becomes a loophole for unlimited upscaling.

`zoompan` is never used anywhere in this project (integer-coordinate jitter),
and `minterpolate` is blacklisted (it smears generative artifacts into
something worse than judder). Resampling is always lanczos.
"""

from __future__ import annotations

from videogen.core.enums import FormatStrategy
from videogen.core.errors import FormatPolicyViolation, UnknownKeyError
from videogen.core.models import FormatPlan

MAX_UPSCALE = 2.5
"""Beyond this the source is being asked to invent detail it does not have."""

MIN_AREA_RETAINED = 0.85
"""fill_crop must keep at least this fraction of the source frame."""

MIN_AREA_RETAINED_PADDED = 0.70
"""hybrid_pad may discard more, because padding absorbs the aspect difference."""

MIN_BACKDROP_BLUR_SIGMA = 20
"""Stops `pad` being used as unlimited free upscale."""

PAD_AREA_TARGET = 0.75
"""How much of the frame `hybrid_pad` keeps before padding.

Trading a quarter of the pixels for a taller on-screen image. 864x480 keeps
648x480, which scales to 1080x800 on a 1080x1920 canvas instead of the 1080x600
strip a zero-crop pad would give.
"""

RESAMPLER = "lanczos"

BANNED_FILTERS = frozenset({"zoompan", "minterpolate"})


class DeliveryTarget:
    """A named delivery canvas."""

    __slots__ = ("allow_pad", "height", "name", "width")

    def __init__(self, name: str, width: int, height: int, *, allow_pad: bool = False) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.allow_pad = allow_pad

    @property
    def aspect(self) -> float:
        return self.width / self.height


TARGETS: dict[str, DeliveryTarget] = {
    "native": DeliveryTarget("native", 0, 0),
    "yt_longform_1080p": DeliveryTarget("yt_longform_1080p", 1920, 1080),
    "yt_longform_720p": DeliveryTarget("yt_longform_720p", 1280, 720),
    "yt_shorts_1080x1920": DeliveryTarget("yt_shorts_1080x1920", 1080, 1920, allow_pad=True),
    "ig_feed_1080x1350": DeliveryTarget("ig_feed_1080x1350", 1080, 1350, allow_pad=True),
}


def get_target(name: str) -> DeliveryTarget:
    """Raises on an unknown target rather than guessing a default."""
    try:
        return TARGETS[name]
    except KeyError:
        raise UnknownKeyError("delivery target", name, TARGETS.keys()) from None


def _even(value: float) -> int:
    """Round to an even integer. h264 with yuv420p requires even dimensions."""
    return max(2, round(value / 2) * 2)


def plan_format(
    native_width: int,
    native_height: int,
    target_name: str,
    *,
    allow_lossy: bool = False,
) -> FormatPlan:
    """Derive the transform from native output to a delivery canvas.

    Raises `FormatPolicyViolation` when the result would exceed the upscale or
    area-retention limits, unless `allow_lossy` records a deliberate waiver.

    The error message names both real fixes: give the provider a native mode
    that matches the target, or waive on the record. Because this table is
    derived from the provider's declared capabilities, a model that gains a
    native portrait mode makes 9:16 legal with no code change at all.
    """
    target = get_target(target_name)

    if target.name == "native":
        return FormatPlan(
            strategy=FormatStrategy.NATIVE,
            target_name=target_name,
            native_width=native_width,
            native_height=native_height,
            target_width=native_width,
            target_height=native_height,
            crop_width=native_width,
            crop_height=native_height,
            scale_width=native_width,
            scale_height=native_height,
            crop_x=0,
            crop_y=0,
            upscale_factor=1.0,
            area_retained=1.0,
        )

    native_aspect = native_width / native_height

    # --- fill_crop: scale so the frame covers the canvas, then crop the excess
    scale = max(target.width / native_width, target.height / native_height)
    scale_w = _even(native_width * scale)
    scale_h = _even(native_height * scale)
    crop_x = max(0, (scale_w - target.width) // 2)
    crop_y = max(0, (scale_h - target.height) // 2)

    # In source pixels, how much of the frame survives the crop.
    kept_w = min(native_width, target.width / scale)
    kept_h = min(native_height, target.height / scale)
    area_retained = (kept_w * kept_h) / (native_width * native_height)

    plan = FormatPlan(
        strategy=FormatStrategy.FILL_CROP,
        target_name=target_name,
        native_width=native_width,
        native_height=native_height,
        target_width=target.width,
        target_height=target.height,
        crop_width=target.width,
        crop_height=target.height,
        scale_width=scale_w,
        scale_height=scale_h,
        crop_x=crop_x,
        crop_y=crop_y,
        upscale_factor=round(scale, 4),
        area_retained=round(area_retained, 4),
    )

    if scale <= MAX_UPSCALE and area_retained >= MIN_AREA_RETAINED:
        return plan

    # --- fill_crop is too destructive. Try hybrid_pad if the target allows it.
    if target.allow_pad:
        padded = _plan_hybrid_pad(native_width, native_height, target)
        if (
            padded.upscale_factor <= MAX_UPSCALE
            and padded.area_retained >= MIN_AREA_RETAINED_PADDED
        ):
            return padded

    message = (
        f"{native_width}x{native_height} (aspect {native_aspect:.3f}) -> "
        f"{target.width}x{target.height} (aspect {target.aspect:.3f}) needs "
        f"{scale:.2f}x upscale keeping {area_retained:.1%} of the frame; "
        f"limits are {MAX_UPSCALE}x and {MIN_AREA_RETAINED:.0%}. "
        "Fix by generating at a native size matching the target aspect, or "
        "pass allow_lossy=True to waive this on the record."
    )
    if not allow_lossy:
        raise FormatPolicyViolation(
            message, upscale=scale, area_retained=area_retained, waivable=True
        )
    return plan.model_copy(update={"waived": True, "waiver_reason": message})


def _plan_hybrid_pad(native_width: int, native_height: int, target: DeliveryTarget) -> FormatPlan:
    """Crop toward the target aspect, scale to fit the width, pad the rest.

    Used when the aspect change is too large to crop through — a 1.8 landscape
    source onto a 0.5625 vertical canvas discards 69% of the frame by cropping,
    but only 25% by cropping partway and letting a blurred backdrop carry the
    remaining height.
    """
    native_aspect = native_width / native_height

    # Crop a fixed fraction off the over-long axis, then let the backdrop carry
    # the rest. Cropping nothing would also work and would retain 100% of the
    # pixels — but it leaves the picture as a thin letterboxed strip. Giving up
    # a quarter of the frame buys a noticeably taller image on a phone, which is
    # the whole point of a vertical canvas. That trade is what PAD_AREA_TARGET
    # names; it is an aesthetic choice, and it is the only one in this module.
    if native_aspect > target.aspect:
        crop_w = _even(native_width * PAD_AREA_TARGET)
        crop_h = _even(native_height)
    else:
        crop_w = _even(native_width)
        crop_h = _even(native_height * PAD_AREA_TARGET)

    scale = min(target.width / crop_w, target.height / crop_h)
    scale_w = _even(crop_w * scale)
    scale_h = _even(crop_h * scale)

    area_retained = (crop_w * crop_h) / (native_width * native_height)

    return FormatPlan(
        strategy=FormatStrategy.HYBRID_PAD,
        target_name=target.name,
        native_width=native_width,
        native_height=native_height,
        target_width=target.width,
        target_height=target.height,
        crop_width=crop_w,
        crop_height=crop_h,
        scale_width=scale_w,
        scale_height=scale_h,
        crop_x=max(0, (native_width - crop_w) // 2),
        crop_y=max(0, (native_height - crop_h) // 2),
        upscale_factor=round(scale, 4),
        area_retained=round(area_retained, 4),
        backdrop_blur_sigma=MIN_BACKDROP_BLUR_SIGMA,
    )


def to_ffmpeg_filter(plan: FormatPlan) -> str:
    """Render a `FormatPlan` as an ffmpeg filter chain.

    Emitted into `render_manifest.json` so `grammar_gate` can assert over the
    literal filter string that shipped — which is what turns "we banned
    zoompan" from a code-review norm into an executable check.
    """
    if plan.strategy is FormatStrategy.NATIVE:
        return "null"

    if plan.strategy is FormatStrategy.FILL_CROP:
        return (
            f"scale={plan.scale_width}:{plan.scale_height}:flags={RESAMPLER},"
            f"crop={plan.crop_width}:{plan.crop_height}:{plan.crop_x}:{plan.crop_y}"
        )

    sigma = plan.backdrop_blur_sigma or MIN_BACKDROP_BLUR_SIGMA
    pad_x = (plan.target_width - plan.scale_width) // 2
    pad_y = (plan.target_height - plan.scale_height) // 2
    return (
        # backdrop: cover the canvas, blur hard
        f"split=2[bg][fg];"
        f"[bg]scale={plan.target_width}:{plan.target_height}:force_original_aspect_ratio=increase,"
        f"crop={plan.target_width}:{plan.target_height},gblur=sigma={sigma}[bgb];"
        # foreground: the real pixels, cropped toward the target and scaled once
        f"[fg]crop={plan.crop_width}:{plan.crop_height}:{plan.crop_x}:{plan.crop_y},"
        f"scale={plan.scale_width}:{plan.scale_height}:flags={RESAMPLER}[fgs];"
        f"[bgb][fgs]overlay={pad_x}:{pad_y}"
    )
