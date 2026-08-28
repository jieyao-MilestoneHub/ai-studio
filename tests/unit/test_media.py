"""`ai_studio.media`: the ffmpeg/ffprobe helpers. These tests need a real
ffmpeg on PATH and are marked `ffmpeg` so `-m "not ffmpeg"` skips them."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.ffmpeg
def test_extract_audio_writes_an_m4a_with_the_clips_duration(tmp_path: Path) -> None:
    """Real ffmpeg: a 2-second test tone inside an mp4 comes out as an AAC
    m4a whose probed duration is what LINE's audio object is told."""
    from ai_studio import media

    src = tmp_path / "clip.mp4"
    media.run([
        media.get_settings().ffmpeg_bin, "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src),
    ])
    out, ms = media.extract_audio(src, tmp_path / "out" / "clip_audio.m4a")
    assert out.is_file() and out.suffix == ".m4a"
    assert 1800 <= ms <= 2300, ms
    assert media.probe_duration_s(out) > 1.5


@pytest.mark.ffmpeg
def test_extract_audio_refuses_a_clip_with_no_audio_track(tmp_path: Path) -> None:
    from ai_studio import media

    src = tmp_path / "silent.mp4"
    media.run([
        media.get_settings().ffmpeg_bin, "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(src),
    ])
    with pytest.raises(media.FFmpegError, match="no audio track"):
        media.extract_audio(src, tmp_path / "x.m4a")


@pytest.mark.ffmpeg
def test_extract_audio_enforces_the_callers_ceiling(tmp_path: Path) -> None:
    """The delivery channel's limit is the caller's number; over it, the
    failure must be ours and loud, not the channel's and silent."""
    from ai_studio import media

    src = tmp_path / "clip.mp4"
    media.run([
        media.get_settings().ffmpeg_bin, "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src),
    ])
    with pytest.raises(media.FFmpegError, match="ceiling"):
        media.extract_audio(src, tmp_path / "tiny.m4a", max_bytes=10)
