"""Poster generation for LINE preview images.

Marked `ffmpeg` because it shells out for real — there is no point testing a
thumbnail against a mock of the thing that makes thumbnails. Run the rest with
`-m "not ffmpeg"`.

The ceiling is the whole point: a `previewImageUrl` over 1MB does not degrade,
it makes LINE reject the entire message object, so the user loses the video and
not just its thumbnail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_studio import media

pytestmark = pytest.mark.ffmpeg


@pytest.fixture(scope="module")
def samples(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One clip and one Flux-shaped still, generated rather than committed."""
    if media.which(media.get_settings().ffmpeg_bin) is None:
        pytest.skip("ffmpeg is not on PATH")

    out = tmp_path_factory.mktemp("samples")
    clip = out / "clip.mp4"
    still = out / "flux.png"
    binary = media.get_settings().ffmpeg_bin
    subprocess.run(
        [binary, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=864x480:rate=24:duration=2", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    subprocess.run(
        [binary, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=1024x1024:rate=1:duration=1", "-frames:v", "1", str(still)],
        check=True,
    )
    return {"clip": clip, "still": still}


def test_a_video_poster_is_a_jpeg_under_the_ceiling(
    samples: dict[str, Path], tmp_path: Path
) -> None:
    out = media.poster(samples["clip"], tmp_path / "clip_poster.jpg")

    assert out.is_file()
    assert out.stat().st_size <= media.POSTER_MAX_BYTES
    assert out.read_bytes()[:2] == b"\xff\xd8", "not a JPEG"


def test_an_image_poster_is_a_new_jpeg_not_the_original(
    samples: dict[str, Path], tmp_path: Path
) -> None:
    """Flux writes PNGs; a 1024x1024 PNG routinely clears 1MB on its own."""
    out = media.poster(samples["still"], tmp_path / "flux_poster.jpg")

    assert out.stat().st_size <= media.POSTER_MAX_BYTES
    assert out.read_bytes()[:2] == b"\xff\xd8"
    assert out != samples["still"]


@pytest.mark.parametrize("key", ["clip", "still"])
def test_the_poster_keeps_the_aspect_ratio(
    key: str, samples: dict[str, Path], tmp_path: Path
) -> None:
    """LINE requires the preview's ratio to match the media's."""
    source = samples[key]
    out = media.poster(source, tmp_path / f"{key}_poster.jpg")

    original = media.probe(source) if key == "clip" else media.probe_image(source)
    preview = media.probe_image(out)

    assert preview.aspect == pytest.approx(original.aspect, abs=0.01)


def test_a_small_source_is_never_upscaled_to_meet_a_width(
    samples: dict[str, Path], tmp_path: Path
) -> None:
    """`scale='min(iw,W)'` only ever shrinks. Upscaling would grow the file to
    no benefit and toward the ceiling."""
    out = media.poster(samples["clip"], tmp_path / "small.jpg")

    assert media.probe_image(out).width <= media.probe(samples["clip"]).width


def test_an_impossible_ceiling_raises_rather_than_returning_an_oversized_file(
    samples: dict[str, Path], tmp_path: Path
) -> None:
    """Silently handing back 1.2MB costs the user their video, not its
    thumbnail — the whole message object is rejected."""
    with pytest.raises(media.FFmpegError, match="under 10 bytes"):
        media.poster(samples["still"], tmp_path / "impossible.jpg", max_bytes=10)


def test_a_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(media.FFmpegError, match="missing file"):
        media.poster(tmp_path / "nope.mp4", tmp_path / "out.jpg")
