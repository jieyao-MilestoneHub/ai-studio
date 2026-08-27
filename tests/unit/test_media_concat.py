"""`media.concat` and `media.normalize_loudness`: the two ffmpeg steps a drama
adds. Real ffmpeg (marked), because the thing under test is the argv."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ai_studio import media

pytestmark = pytest.mark.ffmpeg

ffmpeg_missing = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


def _clip(dest: Path, seconds: float, *, tone_hz: int) -> Path:
    """A tiny synthetic clip with audio, the shape H3 output takes."""
    media.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=96x54:rate=24:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency={tone_hz}:sample_rate=32000:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(dest),
        ],
        timeout_s=120.0,
    )
    return dest


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_concat_joins_in_order_and_keeps_audio(tmp_path: Path) -> None:
    a = _clip(tmp_path / "a.mp4", 1.0, tone_hz=440)
    b = _clip(tmp_path / "b.mp4", 1.5, tone_hz=880)
    out = tmp_path / "out" / "joined.mp4"

    argv = media.concat([a, b], out)

    info = media.probe(out)
    assert 2.3 <= info.duration_s <= 2.7
    assert info.has_audio
    assert info.width % 2 == 0 and info.height % 2 == 0
    assert argv[0] == "ffmpeg" and "-c:v" in argv and "libx264" in argv
    assert "+faststart" in argv
    listing = out.with_suffix(".concat.txt").read_text(encoding="utf-8").splitlines()
    assert listing[0].endswith("a.mp4'") and listing[1].endswith("b.mp4'")


def test_concat_of_nothing_or_of_a_missing_clip_raises(tmp_path: Path) -> None:
    with pytest.raises(media.FFmpegError, match="nothing"):
        media.concat([], tmp_path / "x.mp4")
    with pytest.raises(media.FFmpegError, match="missing clip"):
        media.concat([tmp_path / "ghost.mp4"], tmp_path / "x.mp4")


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_normalize_loudness_is_two_pass_and_linear(tmp_path: Path) -> None:
    src = _clip(tmp_path / "src.mp4", 1.0, tone_hz=440)
    dest = tmp_path / "lvl" / "src.mp4"

    measure, apply = media.normalize_loudness(src, dest)

    assert dest.is_file() and media.probe(dest).has_audio
    assert "-f" in measure and "null" in measure  # pass 1 writes nothing
    joined = " ".join(apply)
    assert "measured_I=" in joined and "linear=true" in joined
    assert "-c:v copy" in joined  # picture untouched


def test_normalize_loudness_of_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(media.FFmpegError, match="missing"):
        media.normalize_loudness(tmp_path / "ghost.mp4", tmp_path / "out.mp4")


def test_loudnorm_stats_parser_rejects_a_stderr_with_no_block() -> None:
    with pytest.raises(media.FFmpegError, match="no measurement"):
        media._loudnorm_stats("frame=1 fps=0\n")
    with pytest.raises(media.FFmpegError, match="lacks"):
        media._loudnorm_stats('{"input_i": "-20.0"}')
