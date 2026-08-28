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
"""Default ceiling for a poster: small enough for any chat platform's
preview-image limit (they cluster around 1 MB). A delivery channel with a
different number passes `max_bytes`; a poster over the channel's limit does
not degrade there -- the whole message is rejected -- so the size is
enforced here, and loudly."""

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
    of an image — no larger than `max_bytes`.

    One code path for both kinds because ffmpeg needs none: `-frames:v 1`
    takes the first frame of a clip and the only frame of a still.

    **Always a new JPEG, never the original reused.** Flux writes 1024x1024
    PNGs, which routinely clear 1 MB on their own; and a video has no image to
    reuse in the first place.

    Aspect ratio is preserved throughout — a preview must look like the media
    it stands for, and `scale` here only ever shrinks (`min(iw,W)`) so a
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


def extract_audio(
    src: Path, dest: Path, *, bitrate: str = "128k", max_bytes: int | None = None
) -> tuple[Path, int]:
    """Write the audio track of `src` as an M4A (AAC) at `dest`. Returns the
    path and the track's duration in **milliseconds**.

    Pure ffmpeg, no GPU, no pod. Re-encodes rather than `-c:a copy` because
    M4A/AAC plays everywhere and a phone clip's track may be anything;
    `-vn` drops the picture. Raises `FFmpegError` when the clip has no audio
    stream at all -- an empty file "succeeding" is how a user waits on a
    message that never plays -- and when the result exceeds `max_bytes`
    (the delivery channel's ceiling, if it has one): the failure should be
    ours, not the channel's.
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
    if max_bytes is not None and dest.stat().st_size > max_bytes:
        raise FFmpegError(f"{dest.name} is {dest.stat().st_size} bytes, over the {max_bytes}-byte ceiling")
    return dest, round(probe_duration_s(dest) * 1000)

LOUDNORM_TARGET = "I=-14:TP=-1.5:LRA=11"
"""docs/editing-grammar.md section 5.3: the delivery loudness target."""


def normalize_loudness(src: Path, dest: Path, *, timeout_s: float = 600.0) -> list[list[str]]:
    """Two-pass `loudnorm` (grammar section 5.3). Returns both argv lists.

    Pass 1 measures; pass 2 applies the measured values **with `linear=true`**
    so one fixed gain is applied across the file. Single-pass loudnorm is a
    dynamic compressor -- it pumps -- and is forbidden by the grammar. Video is
    stream-copied: only the audio changes. `-ar 48000` because loudnorm
    internally resamples to 192 kHz and would otherwise emit that.

    This is also section 5.5 in practice: run it on every generated clip
    before concatenation, and N independently generated audio beds land at
    one level instead of jumping at every cut.
    """
    src, dest = Path(src), Path(dest)
    if not src.is_file():
        raise FFmpegError(f"cannot normalise a missing file: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    binary = get_settings().ffmpeg_bin

    measure = [
        binary, "-hide_banner", "-nostats", "-i", str(src),
        "-af", f"loudnorm={LOUDNORM_TARGET}:print_format=json",
        "-f", "null", "-",
    ]
    proc = run(measure, timeout_s=timeout_s)
    stats = _loudnorm_stats(proc.stderr)

    apply = [
        binary, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-af",
        (
            f"loudnorm={LOUDNORM_TARGET}"
            f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
            f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
            f":offset={stats['target_offset']}:linear=true:print_format=summary"
        ),
        "-ar", "48000",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
        str(dest),
    ]
    run(apply, timeout_s=timeout_s)
    return [measure, apply]


def _loudnorm_stats(stderr: str) -> dict[str, str]:
    """The JSON block loudnorm prints at the end of pass 1."""
    start, end = stderr.rfind("{"), stderr.rfind("}")
    if start == -1 or end <= start:
        raise FFmpegError("loudnorm pass 1 printed no measurement block")
    try:
        stats = json.loads(stderr[start : end + 1])
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"loudnorm measurement was not JSON: {exc}") from exc
    needed = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    missing = [k for k in needed if k not in stats]
    if missing:
        raise FFmpegError(f"loudnorm measurement lacks {missing}")
    return {k: str(stats[k]) for k in needed}


def concat(clips: list[Path], dest: Path, *, timeout_s: float = 900.0) -> list[str]:
    """Hard-cut `clips` together into `dest`. Returns the argv.

    The concat demuxer with a re-encode, not `-c copy`: the clips come from
    separate generations and a copy-concat only works when every stream
    parameter matches exactly -- when it does not, the failure is a file that
    plays with frozen frames or no audio after the first cut, discovered
    after delivery. A single `libx264 -crf 18` pass over a minute of 864x480
    is seconds of CPU and removes that class of bug. Even dimensions are kept
    by the source; `+faststart` because the file is streamed, not downloaded.
    """
    if not clips:
        raise FFmpegError("nothing to concatenate")
    paths = [Path(c) for c in clips]
    for path in paths:
        if not path.is_file():
            raise FFmpegError(f"cannot concatenate a missing clip: {path}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    listing = dest.with_suffix(".concat.txt")
    # The demuxer's list format; single quotes inside a path are escaped as
    # '\''. Written utf-8 explicitly -- the Windows default would not be.
    lines = ["file '" + str(p.resolve()).replace("'", "'\\''") + "'" for p in paths]
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    argv = [
        get_settings().ffmpeg_bin,
        "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(dest),
    ]
    run(argv, timeout_s=timeout_s)
    return argv


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
