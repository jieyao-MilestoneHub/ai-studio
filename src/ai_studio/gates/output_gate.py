"""POST gate: did the model produce the thing it was asked for?

The first rule body in this package. Everything else here has been shell since
the layer was built.

It exists because of a specific and very cheap failure: **a broken generation is
not an error anywhere in the stack.** A NaN latent decodes to a uniformly black
or uniformly grey image; ComfyUI saves it, `/history` reports success, the
provider fetches it, `media.probe()` returns perfectly sensible dimensions, and
the bot pushes it into the group. The same is true of a LoRA that failed to
load — ComfyUI logs `lora key not loaded` at WARNING and returns a completely
valid picture of the wrong thing.

So the only place those can be caught is by looking at what came out and
comparing it against what was promised. That is this gate.

**Two artifacts, no ffmpeg.** `provider_manifest.json` says what the model was
configured to produce; `output.json` says what it actually produced, including
the luma measurement `media.luma_stats()` took at fetch time. Reading both from
disk is what keeps this a pure function — the layer contract forbids importing
`providers`, and shelling out to ffprobe here would make the gate untestable
against fixtures, which is most of the point.

Being a POST gate, this is a receipt rather than a saving: the GPU-seconds are
already spent by the time it runs. What it buys is that they are not *also*
spent on pushing the result to a person, and that a silent breakage becomes a
loud one.
"""

from __future__ import annotations

from typing import Any

from ai_studio.core.models import GateReport
from ai_studio.gates.core import GateContext, GateRun

GATE = "output"

MANIFEST = "provider_manifest.json"
OUTPUT = "output.json"

MIN_BYTES = 1024
"""Below this the file cannot be a real render.

A truncated fetch and a zero-byte write both land here. Deliberately far below
any plausible output — a measured 5-second H3 clip is 0.99MB 📏 — because this
is catching "nothing arrived", not "the file is small".
"""

FLAT_LUMA_SPREAD = 2.0
"""Luma range below which the file is a single flat colour.

Mirrors `media.FLAT_LUMA_SPREAD`, restated rather than imported: a gate reads
JSON, not modules, and a threshold that moved underneath a fixture would make
the fixture stop testing what it says it tests.
"""

DURATION_TOLERANCE_S = 0.15
"""Container timestamps do not land exactly on frames / fps."""


def output_gate(ctx: GateContext) -> GateReport:
    """Check a finished render against the capabilities it was produced under."""
    run = GateRun(GATE)
    manifest = ctx.artifact(MANIFEST)
    output = ctx.artifact(OUTPUT)

    caps: dict[str, Any] = manifest.get("capabilities") or {}
    luma: dict[str, Any] = output.get("luma") or {}
    is_video = bool(caps.get("native_fps"))

    # --- it arrived at all ------------------------------------------------
    size = int(output.get("size_bytes") or 0)
    run.count("size_bytes", size)
    run.assert_that(
        size >= MIN_BYTES,
        "OUT-SIZE",
        f"output is {size} bytes, which is too small to be a render",
        observed=size,
        expected=f">= {MIN_BYTES}",
    )

    # --- it is the shape that was ordered ---------------------------------
    width, height = output.get("width"), output.get("height")
    expected_w, expected_h = caps.get("native_width"), caps.get("native_height")
    run.assert_that(
        (width, height) == (expected_w, expected_h),
        "OUT-DIMS",
        f"output is {width}x{height}, not the {expected_w}x{expected_h} the "
        "provider was configured for",
        observed=f"{width}x{height}",
        expected=f"{expected_w}x{expected_h}",
    )

    if is_video:
        fps = float(output.get("fps") or 0.0)
        expected_fps = float(caps["native_fps"])
        run.assert_that(
            abs(fps - expected_fps) < 0.5,
            "OUT-FPS",
            f"output runs at {fps}fps, not {expected_fps}",
            observed=fps,
            expected=expected_fps,
        )

        frames = output.get("frames")
        duration = float(output.get("duration_s") or 0.0)
        run.count("duration_s", duration)
        if frames is not None and expected_fps:
            implied = float(frames) / expected_fps
            run.assert_that(
                abs(duration - implied) <= DURATION_TOLERANCE_S,
                "OUT-DURATION",
                f"{frames} frames at {expected_fps}fps implies {implied:.3f}s but "
                f"the file is {duration:.3f}s — frames were dropped or added",
                observed=duration,
                expected=f"{implied:.3f}s",
            )

        # H3 generates picture and audio jointly in one pass, so a silent clip
        # means the audio VAE never ran — not that the scene was quiet.
        expects_audio = bool(caps.get("has_native_audio"))
        run.assert_that(
            bool(output.get("has_audio")) == expects_audio,
            "OUT-AUDIO",
            "output has no audio track, but this model generates audio jointly "
            "with picture — a silent file means the audio VAE did not run"
            if expects_audio
            else "output carries an audio track this model does not generate",
            observed=bool(output.get("has_audio")),
            expected=expects_audio,
        )

    # --- it is a picture rather than a colour ------------------------------
    spread = luma.get("spread")
    if spread is None and {"y_min", "y_max"} <= luma.keys():
        spread = float(luma["y_max"]) - float(luma["y_min"])

    if spread is None:
        run.warn_unless(
            False,
            "OUT-FLAT-UNMEASURED",
            "output.json carries no luma measurement, so a black render cannot "
            "be told from a good one here",
        )
    else:
        run.count("luma_spread", float(spread))
        run.assert_that(
            float(spread) >= FLAT_LUMA_SPREAD,
            "OUT-FLAT",
            f"every pixel is within {float(spread):.1f} of every other — the "
            "output is one flat colour, which is what a NaN latent decodes to. "
            "Nothing upstream reports this as an error",
            observed=float(spread),
            expected=f">= {FLAT_LUMA_SPREAD}",
            source_url="https://github.com/comfyanonymous/ComfyUI/issues/4673",
        )

    return run.report()
