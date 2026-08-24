"""ffmpeg / ffprobe invocation.

Two rules, both of which exist so that `grammar_gate` can later assert over
what a run actually did rather than what the code was supposed to do:

- **Commands are argv lists, never shell strings.** No quoting bugs, no shell
  injection through a prompt, and the exact argv is recordable into
  `render_manifest.json`.
- **Every invocation is returned as well as run**, so the caller can log it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from videogen.config.settings import get_settings
from videogen.core.errors import VideogenError

REQUIRED_FILTERS = (
    "xfade",
    "sidechaincompress",
    "loudnorm",
    "ass",
    "acrossfade",
    "colordetect",
)
"""Filters the editing layer will need. `videogen doctor` checks for them."""


class FFmpegError(VideogenError):
    """An ffmpeg/ffprobe invocation failed. Carries the tail of stderr."""


@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe says about a file."""

    width: int
    height: int
    fps: float
    duration_s: float
    has_audio: bool
    video_codec: str
    size_bytes: int

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def which(binary: str) -> str | None:
    return shutil.which(binary)


def run(argv: list[str], *, timeout_s: float = 900.0) -> subprocess.CompletedProcess[str]:
    """Run a command, raising `FFmpegError` with useful context on failure."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        raise FFmpegError(
            f"{argv[0]!r} not found on PATH. Install ffmpeg, or set VIDEOGEN_FFMPEG_BIN."
        ) from None
    except subprocess.TimeoutExpired:
        raise FFmpegError(f"{argv[0]} timed out after {timeout_s}s") from None

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise FFmpegError(f"{argv[0]} exited {proc.returncode}:\n{tail}")
    return proc


def probe(path: Path) -> MediaInfo:
    """Describe a media file. Used to verify what a provider actually produced."""
    path = Path(path)
    if not path.is_file():
        raise FFmpegError(f"cannot probe missing file: {path}")

    settings = get_settings()
    proc = run(
        [
            settings.ffprobe_bin,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout_s=60.0,
    )
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"{path} contains no video stream")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = _first_float(
        video.get("duration"), payload.get("format", {}).get("duration"), default=0.0
    )

    return MediaInfo(
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        duration_s=duration,
        has_audio=has_audio,
        video_codec=str(video.get("codec_name", "")),
        size_bytes=path.stat().st_size,
    )


def missing_filters(binary: str | None = None) -> list[str]:
    """Which of `REQUIRED_FILTERS` this ffmpeg build lacks."""
    binary = binary or get_settings().ffmpeg_bin
    try:
        proc = run([binary, "-hide_banner", "-filters"], timeout_s=60.0)
    except FFmpegError:
        return list(REQUIRED_FILTERS)
    available = proc.stdout
    return [f for f in REQUIRED_FILTERS if f" {f} " not in available]


def _parse_rate(rate: object) -> float:
    """ffprobe reports frame rates as the string ``"24/1"``."""
    if not rate:
        return 0.0
    text = str(rate)
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _first_float(*values: object, default: float = 0.0) -> float:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return default
