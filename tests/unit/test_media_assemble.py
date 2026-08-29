"""`media.assemble`: the one filter_complex a drama ends with. Real ffmpeg
(marked), because the argv and the output length are the things under test."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ai_studio import media

pytestmark = pytest.mark.ffmpeg

ffmpeg_missing = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


def _clip(dest: Path, seconds: float, *, tone_hz: int, size: str = "96x54") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    media.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency={tone_hz}:sample_rate=48000:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(dest),
        ],
        timeout_s=120.0,
    )
    return dest


HARD = media.AssembleBoundary("hard_cut", audio_fade_s=0.125)
DISSOLVE = media.AssembleBoundary("dissolve", overlap_s=0.5, audio_fade_s=0.5)


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_hard_cut_then_dissolve_lands_on_the_timeline_length(tmp_path: Path) -> None:
    clips = [_clip(tmp_path / "a.mp4", 1.0, tone_hz=440), _clip(tmp_path / "b.mp4", 1.5, tone_hz=660),
             _clip(tmp_path / "c.mp4", 1.0, tone_hz=880)]
    out = tmp_path / "out dir" / "joined.mp4"
    # a|b hard, b~c dissolve 0.5: c starts at 2.0, total 3.0
    argv = media.assemble(clips, out, boundaries=[HARD, DISSOLVE], clip_offsets=[0.0, 1.0, 2.0], total_s=3.0)

    info = media.probe(out)
    assert 2.85 <= info.duration_s <= 3.15 and info.has_audio
    graph = argv[argv.index("-filter_complex") + 1]
    assert "xfade=transition=fade:duration=0.5:offset=2" in graph
    assert "acrossfade=d=0.125:o=0" in graph and "acrossfade=d=0.5:o=1" in graph
    assert "fade=t=out:st=2.5" in graph and "ass=" not in graph
    assert "+faststart" in argv


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_subtitles_are_burned_in_from_an_escaped_path(tmp_path: Path) -> None:
    clips = [_clip(tmp_path / "a.mp4", 1.0, tone_hz=440), _clip(tmp_path / "b.mp4", 1.0, tone_hz=660)]
    ass = tmp_path / "with space" / "captions.ass"
    ass.parent.mkdir()
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 96\nPlayResY: 54\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Main,Sans,12,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,4,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.10,0:00:00.90,Main,,0,0,0,,hello\n",
        encoding="utf-8",
    )
    argv = media.assemble(clips, tmp_path / "o.mp4", boundaries=[HARD], clip_offsets=[0.0, 1.0], total_s=2.0, subtitles=ass)
    graph = argv[argv.index("-filter_complex") + 1]
    assert "ass=filename='" in graph and "with space" in graph


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_mismatched_clips_and_wrong_counts_raise_before_encoding(tmp_path: Path) -> None:
    a = _clip(tmp_path / "a.mp4", 1.0, tone_hz=440)
    b = _clip(tmp_path / "b.mp4", 1.0, tone_hz=660, size="64x36")
    with pytest.raises(media.FFmpegError, match="differ in size"):
        media.assemble([a, b], tmp_path / "o.mp4", boundaries=[HARD], clip_offsets=[0.0, 1.0], total_s=2.0)
    with pytest.raises(media.FFmpegError, match="need 1 boundaries"):
        media.assemble([a, a], tmp_path / "o.mp4", boundaries=[], clip_offsets=[0.0, 1.0], total_s=2.0)
    with pytest.raises(media.FFmpegError, match="nothing"):
        media.assemble([], tmp_path / "o.mp4", boundaries=[], clip_offsets=[], total_s=0.0)


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg/ffprobe not on PATH")
def test_a_timeline_that_disagrees_with_the_clips_is_loud(tmp_path: Path) -> None:
    a = _clip(tmp_path / "a.mp4", 1.0, tone_hz=440)
    b = _clip(tmp_path / "b.mp4", 1.0, tone_hz=660)
    with pytest.raises(media.FFmpegError, match="captions would drift"):
        media.assemble([a, b], tmp_path / "o.mp4", boundaries=[HARD], clip_offsets=[0.0, 1.0], total_s=5.0)


def test_boundary_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="unknown boundary"):
        media.AssembleBoundary("wipe")
    with pytest.raises(ValueError, match="no overlap"):
        media.AssembleBoundary("hard_cut", overlap_s=0.5)
    with pytest.raises(ValueError, match="needs an overlap"):
        media.AssembleBoundary("dissolve")


def test_filter_path_escapes_colons_and_quotes(tmp_path: Path) -> None:
    text = media._filter_path(tmp_path / "it's.ass")
    assert text.startswith("'") and text.endswith("'") and "\\'" in text
