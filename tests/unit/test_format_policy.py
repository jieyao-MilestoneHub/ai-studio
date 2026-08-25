"""Format policy: the numbers quoted in the docs must actually hold."""

from __future__ import annotations

import pytest

from ai_studio.core.enums import FormatStrategy
from ai_studio.core.errors import FormatPolicyViolation, UnknownKeyError
from ai_studio.editing.format_policy import (
    BANNED_FILTERS,
    MAX_UPSCALE,
    plan_format,
    to_ffmpeg_filter,
)

H3_480 = (864, 480)
H3_768 = (1344, 768)


def test_native_passthrough_does_not_resample() -> None:
    plan = plan_format(*H3_480, "native")
    assert plan.strategy is FormatStrategy.NATIVE
    assert plan.upscale_factor == 1.0
    assert plan.area_retained == 1.0
    assert to_ffmpeg_filter(plan) == "null"


def test_864x480_to_1080p_matches_documented_geometry() -> None:
    """2.25x upscale, 24 columns cropped, 98.8% of the frame retained."""
    plan = plan_format(*H3_480, "yt_longform_1080p")
    assert plan.strategy is FormatStrategy.FILL_CROP
    assert (plan.scale_width, plan.scale_height) == (1944, 1080)
    assert (plan.crop_width, plan.crop_height) == (1920, 1080)
    assert plan.crop_x == 12 and plan.crop_y == 0
    assert plan.upscale_factor == pytest.approx(2.25, abs=1e-3)
    assert plan.area_retained == pytest.approx(0.988, abs=5e-3)


def test_1344x768_to_1080p_is_much_gentler() -> None:
    """The reason 1344x768 is worth 2.3x the GPU time: 1.43x instead of 2.25x."""
    plan = plan_format(*H3_768, "yt_longform_1080p")
    assert plan.upscale_factor == pytest.approx(1.4286, abs=1e-3)
    assert plan.area_retained == pytest.approx(0.984, abs=5e-3)
    assert plan.upscale_factor < plan_format(*H3_480, "yt_longform_1080p").upscale_factor


def test_720p_is_the_least_lossy_landscape_target() -> None:
    plan = plan_format(*H3_480, "yt_longform_720p")
    assert (plan.scale_width, plan.scale_height) == (1296, 720)
    assert plan.crop_x == 8
    assert plan.upscale_factor == pytest.approx(1.5, abs=1e-3)


def test_vertical_falls_back_to_hybrid_pad_rather_than_a_4x_crop() -> None:
    """A straight crop would keep 31% of the frame at 4x. Padding keeps 75%."""
    plan = plan_format(*H3_480, "yt_shorts_1080x1920")
    assert plan.strategy is FormatStrategy.HYBRID_PAD
    assert (plan.crop_width, plan.crop_height) == (648, 480)
    assert (plan.scale_width, plan.scale_height) == (1080, 800)
    assert plan.upscale_factor == pytest.approx(1.6667, abs=1e-3)
    assert plan.area_retained == pytest.approx(0.75, abs=1e-3)
    assert plan.backdrop_blur_sigma is not None


def test_over_limit_target_raises_and_names_the_fix() -> None:
    """A tiny source onto 1080p blows the upscale limit and must not be guessed at."""
    with pytest.raises(FormatPolicyViolation) as exc:
        plan_format(320, 180, "yt_longform_1080p")
    assert exc.value.upscale is not None and exc.value.upscale > MAX_UPSCALE
    assert exc.value.waivable
    assert "allow_lossy" in str(exc.value)


def test_waiver_is_recorded_not_silent() -> None:
    plan = plan_format(320, 180, "yt_longform_1080p", allow_lossy=True)
    assert plan.waived is True
    assert plan.waiver_reason  # the reason travels with the plan into result.json


def test_unknown_target_raises() -> None:
    with pytest.raises(UnknownKeyError):
        plan_format(*H3_480, "tiktok_but_made_up")


@pytest.mark.parametrize("target", ["yt_longform_1080p", "yt_longform_720p", "yt_shorts_1080x1920"])
def test_emitted_filters_are_lanczos_and_never_banned(target: str) -> None:
    chain = to_ffmpeg_filter(plan_format(*H3_480, target))
    assert "lanczos" in chain
    for banned in BANNED_FILTERS:
        assert banned not in chain


def test_all_emitted_dimensions_are_even() -> None:
    """h264 with yuv420p cannot encode odd dimensions."""
    for target in ("yt_longform_1080p", "yt_longform_720p", "yt_shorts_1080x1920"):
        for native in (H3_480, H3_768):
            plan = plan_format(*native, target)
            for value in (
                plan.scale_width,
                plan.scale_height,
                plan.crop_width,
                plan.crop_height,
            ):
                assert value % 2 == 0, f"{target}/{native}: {value} is odd"
