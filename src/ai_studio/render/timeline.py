"""`resolve_timeline`: the one function that produces absolute time.

Authoring models carry `segment_id` and intended durations; captions carry a
`segment_id` and no time at all. Here, and only here, segments are laid end
to end, transitions take their overlap off the clock, and every start/end is
quantised to a frame. The result is written to exactly one file,
`offsets.json`, and everything that needs a number in seconds reads it
(the ffmpeg assembly, the ASS writer, a future delivery gate).

Why so strict: upstream shipped captions 2-3 s out of sync from hand-split
segments. Binding to an index and computing time once makes that class of
bug impossible rather than merely detectable.

Two kinds of boundary:

- **inside one generated clip** (a model-side sub-cut) -- the picture and
  the sound are already continuous, so it is a hard cut with no audio dip;
- **between two clips** -- the `Transition` decides: a hard cut with a short
  audio crossfade, or a dissolve whose overlap shortens the piece.

`clip_of` maps every segment to the clip it was rendered in; two adjacent
segments in different clips are a clip boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from ai_studio.core.enums import TransitionKind
from ai_studio.core.models import Segment
from ai_studio.core.timecode import DEFAULT_FPS, seconds_to_frames
from ai_studio.editing.transitions import Transition, hard_cut

OFFSETS_FILE = "offsets.json"

_SUPPORTED = frozenset({TransitionKind.HARD_CUT, TransitionKind.DISSOLVE})
"""What the assembly can render today. A whip/wipe/zoom needs shot-pair
evidence nothing produces yet; `editing.transitions` downgrades them, and a
caller that hands one over anyway gets a raise, not a silent hard cut."""


@dataclass(frozen=True)
class Boundary:
    """The splice after a segment."""

    after_segment_id: str
    kind: TransitionKind
    overlap_s: float
    audio_fade_s: float
    clip_boundary: bool


@dataclass(frozen=True)
class SegmentOffset:
    segment_id: str
    shot_id: str
    clip: str
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Timeline:
    fps: int
    segments: tuple[SegmentOffset, ...]
    boundaries: tuple[Boundary, ...]
    clip_offsets: tuple[float, ...]
    """Where each source clip starts on the output timeline, in clip order --
    what an `xfade` offset is computed from."""
    total_s: float

    def segment(self, segment_id: str) -> SegmentOffset:
        for s in self.segments:
            if s.segment_id == segment_id:
                return s
        raise KeyError(f"no segment {segment_id!r} on the timeline")

    @property
    def clips(self) -> tuple[str, ...]:
        seen: list[str] = []
        for s in self.segments:
            if s.clip not in seen:
                seen.append(s.clip)
        return tuple(seen)


def resolve_timeline(
    segments: Sequence[Segment],
    clip_of: Mapping[str, str],
    transitions: Sequence[Transition],
    *,
    fps: int = DEFAULT_FPS,
) -> Timeline:
    """Lay `segments` end to end. `transitions` has one entry per *clip*
    boundary, in order; sub-cuts inside a clip need none."""
    if not segments:
        raise ValueError("a timeline needs at least one segment")
    missing = [s.segment_id for s in segments if s.segment_id not in clip_of]
    if missing:
        raise ValueError(f"segments without a clip: {missing}")
    clip_boundaries = sum(1 for a, b in pairwise(segments) if clip_of[a.segment_id] != clip_of[b.segment_id])
    if len(transitions) != clip_boundaries:
        raise ValueError(f"{clip_boundaries} clip boundaries but {len(transitions)} transitions")
    for t in transitions:
        if t.kind not in _SUPPORTED:
            raise ValueError(f"cannot render a {t.kind.value}; no evidence path exists -- downgrade it first")

    offsets: list[SegmentOffset] = []
    boundaries: list[Boundary] = []
    clip_starts: dict[str, int] = {}
    frame = 0
    pending = iter(transitions)
    for i, seg in enumerate(segments):
        clip = clip_of[seg.segment_id]
        clip_starts.setdefault(clip, frame)
        length = seconds_to_frames(seg.intended_duration_s, fps)
        if length <= 0:
            raise ValueError(f"segment {seg.segment_id} is shorter than a frame")
        start, end = frame, frame + length
        offsets.append(SegmentOffset(
            segment_id=seg.segment_id, shot_id=seg.shot_id, clip=clip,
            start_s=start / fps, end_s=end / fps, start_frame=start, end_frame=end,
        ))
        frame = end
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            if clip_of[nxt.segment_id] != clip:
                t = next(pending)
                overlap = seconds_to_frames(t.overlap_s, fps)
                if overlap >= length:
                    raise ValueError(f"{t.kind.value} overlap {t.overlap_s}s swallows segment {seg.segment_id}")
                frame -= overlap
            else:
                t = hard_cut(audio_fade_s=0.0)
            boundaries.append(Boundary(
                after_segment_id=seg.segment_id, kind=t.kind, overlap_s=t.overlap_s,
                audio_fade_s=t.audio_fade_s, clip_boundary=clip_of[nxt.segment_id] != clip,
            ))

    ordered_clips = sorted(clip_starts, key=clip_starts.__getitem__)
    return Timeline(
        fps=fps, segments=tuple(offsets), boundaries=tuple(boundaries),
        clip_offsets=tuple(clip_starts[c] / fps for c in ordered_clips), total_s=frame / fps,
    )


def to_json(timeline: Timeline) -> dict[str, object]:
    return {
        "fps": timeline.fps,
        "total_s": timeline.total_s,
        "clips": list(timeline.clips),
        "clip_offsets": list(timeline.clip_offsets),
        "segments": [asdict(s) for s in timeline.segments],
        "boundaries": [{**asdict(b), "kind": b.kind.value} for b in timeline.boundaries],
    }


def write_offsets(run_dir: Path | str, timeline: Timeline) -> Path:
    """`runs/<id>/offsets.json` -- the only file absolute time is written to."""
    path = Path(run_dir) / OFFSETS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(timeline), indent=2), encoding="utf-8")
    return path
