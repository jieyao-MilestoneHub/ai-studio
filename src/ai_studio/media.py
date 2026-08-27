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

from ai_studio.config.settings import get_settings
from ai_studio.core.errors import AIStudioError

REQUIRED_FILTERS = (
    "xfade",
    "sidechaincompress",
    "loudnorm",
    "ass",
    "acrossfade",
    "colordetect",
)
"""Filters the editing layer will need. `ai-studio doctor` checks for them."""


class FFmpegError(AIStudioError):
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
            f"{argv[0]!r} not found on PATH. Install ffmpeg, or set AI_STUDIO_FFMPEG_BIN."
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


@dataclass(frozen=True)
class ImageInfo:
    """What ffprobe says about a still image. No fps/duration — there is none."""

    width: int
    height: int
    size_bytes: int

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def probe_image(path: Path) -> ImageInfo:
    """Describe an image file. `probe()` raises on a file with no video stream,
    which every still image is, so this is a deliberate sibling rather than a
    branch inside it."""
    path = Path(path)
    if not path.is_file():
        raise FFmpegError(f"cannot probe missing file: {path}")

    settings = get_settings()
    proc = run(
        [
            settings.ffprobe_bin,
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            str(path),
        ],
        timeout_s=60.0,
    )
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])

    image = next((s for s in streams if s.get("codec_type") == "video"), None)
    if image is None:
        raise FFmpegError(f"{path} contains no image/video stream")

    return ImageInfo(
        width=int(image.get("width", 0)),
        height=int(image.get("height", 0)),
        size_bytes=path.stat().st_size,
    )


POSTER_MAX_BYTES = 1_000_000
"""LINE's ceiling for `previewImageUrl`, minus a little room.

The documented limit is 1 MB. A poster that goes over it does not degrade —
the whole message object is rejected, and the user gets nothing at all.
"""

_POSTER_LADDER: tuple[tuple[int, int], ...] = (
    (1024, 4),
    (720, 6),
    (512, 8),
    (320, 14),
)
"""(max width, JPEG quality) rungs, tried in order until one fits the ceiling.

Width first, then quality: a smaller poster still looks like the thing it is a
poster for, while a heavily quantised one at full size looks broken. `-q:v` is
mjpeg's scale where 2 is best and 31 is worst.
"""


def poster(
    src: Path, dest: Path, *, max_bytes: int = POSTER_MAX_BYTES
) -> Path:
    """Write a JPEG preview of `src` — first frame of a video, or a thumbnail
    of an image — small enough for LINE to accept it.

    One code path for both kinds because ffmpeg needs none: `-frames:v 1`
    takes the first frame of a clip and the only frame of a still.

    **Always a new JPEG, never the original reused.** Flux writes 1024x1024
    PNGs, which routinely clear 1 MB on their own; and a video has no image to
    reuse in the first place.

    Aspect ratio is preserved throughout — LINE requires the preview's ratio to
    match the media's, and `scale` here only ever shrinks (`min(iw,W)`) so a
    small source is never blown up to meet a width.

    Raises rather than returning an oversized file: a poster that is over the
    limit fails the *whole* message object, so a silent 1.2 MB poster costs the
    user their video, not just its thumbnail.
    """
    src, dest = Path(src), Path(dest)
    if not src.is_file():
        raise FFmpegError(f"cannot make a poster from a missing file: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    binary = get_settings().ffmpeg_bin
    sizes: list[int] = []
    for width, quality in _POSTER_LADDER:
        run(
            [
                binary,
                "-hide_banner", "-loglevel", "error",
                "-y",
                "-i", str(src),
                "-frames:v", "1",
                # min() so this only ever shrinks; -2 keeps the height even,
                # which the JPEG encoder's 4:2:0 chroma requires.
                "-vf", f"scale='min(iw,{width})':-2",
                "-pix_fmt", "yuvj420p",
                "-q:v", str(quality),
                "-f", "image2",
                str(dest),
            ],
            timeout_s=120.0,
        )
        size = dest.stat().st_size
        if size <= max_bytes:
            return dest
        sizes.append(size)

    raise FFmpegError(
        f"could not get a poster for {src.name} under {max_bytes} bytes; "
        f"tried {_POSTER_LADDER} and got {sizes}"
    )


def probe_duration_s(path: Path) -> float:
    """The container's duration in seconds, for a file that may have no video
    stream at all (an audio-only M4A). `probe()` insists on a video stream
    because it describes what a *clip* provider produced; this asks ffprobe
    only about the format."""
    path = Path(path)
    if not path.is_file():
        raise FFmpegError(f"cannot probe missing file: {path}")
    proc = run(
        [get_settings().ffprobe_bin, "-v", "error", "-print_format", "json",
         "-show_format", str(path)],
        timeout_s=60.0,
    )
    payload = json.loads(proc.stdout or "{}")
    return _first_float(payload.get("format", {}).get("duration"), default=0.0)


AUDIO_MAX_BYTES = 200 * 1024 * 1024
"""LINE's ceiling for an audio message object (`[reported]`, Messaging API
reference). An AAC track at 128 kbps is ~1 MB/min, so a phone clip never
comes close; the check is here so the failure is ours, not LINE's."""


def extract_audio(src: Path, dest: Path, *, bitrate: str = "128k") -> tuple[Path, int]:
    """Write the audio track of `src` as an M4A (AAC) at `dest`. Returns the
    path and the track's duration in **milliseconds** -- what LINE's audio
    message object wants.

    `/影音`'s one job. Pure ffmpeg, no GPU, no pod: the only trigger that
    costs nothing but the host's CPU. Re-encodes rather than `-c:a copy`
    because LINE plays M4A/AAC and a phone clip's track may be anything;
    `-vn` drops the picture. Raises `FFmpegError` when the clip has no audio
    stream at all -- an empty file "succeeding" is how a user waits on a
    message that never plays.
    """
    src, dest = Path(src), Path(dest)
    info = probe(src)
    if not info.has_audio:
        raise FFmpegError(f"{src.name} has no audio track to extract")
    dest.parent.mkdir(parents=True, exist_ok=True)
    binary = get_settings().ffmpeg_bin
    run([
        binary, "-v", "error", "-y", "-i", str(src),
        "-vn", "-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart",
        str(dest),
    ])
    if dest.stat().st_size > AUDIO_MAX_BYTES:
        raise FFmpegError(f"{dest.name} is {dest.stat().st_size} bytes; LINE's limit is {AUDIO_MAX_BYTES}")
    return dest, round(probe_duration_s(dest) * 1000)


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
