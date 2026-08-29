"""ASS subtitle emission from segment-bound cues and a resolved timeline.

The cue carries a `segment_id`; the timeline carries the segment's start and
end; this module puts the two together and nothing else. It never adds
seconds of its own, so a caption cannot straddle a cut: its window *is* the
segment's window, shaved by a small margin so the text does not flash on the
exact frame of the splice.

The title card is the one event with a time typed in here, and it is typed
once: the first `until_s` of the piece, over whatever the first segment
shows, with a dark box and a fade -- the hook wears a name.

Pure text in, text out. `media.assemble` burns the file in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ai_studio.core.models import CaptionCue
from ai_studio.editing.captions import break_lines, resolve_color, strip_emoji
from ai_studio.render.timeline import Timeline

CAPTIONS_FILE = "captions.ass"
TITLE_UNTIL_S = 1.5
CUE_MARGIN_S = 0.1


@dataclass(frozen=True)
class AssStyle:
    name: str
    font: str
    size: int
    primary: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    bold: int = 0
    border_style: int = 1
    """1 = outline + shadow, 3 = opaque box (uses `back_color`)."""
    outline: int = 2
    shadow: int = 0
    alignment: int = 2
    """Numpad layout: 2 bottom centre, 5 dead centre, 8 top centre."""
    margin_v: int = 28

    def line(self) -> str:
        return (
            f"Style: {self.name},{self.font},{self.size},{self.primary},{self.primary},"
            f"{self.outline_color},{self.back_color},{self.bold},0,0,0,100,100,0,0,"
            f"{self.border_style},{self.outline},{self.shadow},{self.alignment},20,20,{self.margin_v},1"
        )


@dataclass(frozen=True)
class AssEvent:
    start_s: float
    end_s: float
    style: str
    text: str
    """Already escaped and line-broken with `\\N`."""
    tags: str = ""
    """Override block without braces, e.g. `\\fad(0,300)`."""

    def line(self) -> str:
        if self.end_s <= self.start_s:
            raise ValueError(f"event ends before it starts: {self.start_s}-{self.end_s}")
        body = f"{{{self.tags}}}{self.text}" if self.tags else self.text
        return f"Dialogue: 0,{_time(self.start_s)},{_time(self.end_s)},{self.style},,0,0,0,,{body}"


@dataclass(frozen=True)
class AssDocument:
    play_res: tuple[int, int]
    styles: tuple[AssStyle, ...]
    events: tuple[AssEvent, ...] = field(default_factory=tuple)


def _time(seconds: float) -> str:
    """`H:MM:SS.cc` -- ASS keeps centiseconds."""
    if seconds < 0:
        raise ValueError(f"negative time: {seconds}")
    cs = round(seconds * 100)
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def escape(text: str) -> str:
    """ASS treats `{}` as override blocks and `\\` as a tag lead-in."""
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")  # noqa: RUF001


def default_styles(font: str, play_res: tuple[int, int]) -> tuple[AssStyle, AssStyle]:
    """White-first body captions at the bottom, a boxed title in the centre.
    Sizes scale with the height so 480p and 1080p read the same."""
    _, h = play_res
    return (
        AssStyle(name="Main", font=font, size=round(h * 0.065), margin_v=round(h * 0.06)),
        AssStyle(name="Title", font=font, size=round(h * 0.10), bold=1, border_style=3, outline=6,
                 alignment=5, margin_v=0),
    )


def title_card(title: str, *, until_s: float = TITLE_UNTIL_S, fade_ms: int = 300) -> AssEvent:
    text = escape(strip_emoji(title))
    if not text:
        raise ValueError("a title card needs a title")
    return AssEvent(start_s=0.0, end_s=until_s, style="Title", text=text, tags=f"\\fad(0,{fade_ms})")


def cue_events(cues: Sequence[CaptionCue], timeline: Timeline, *, margin_s: float = CUE_MARGIN_S) -> list[AssEvent]:
    """One event per cue, windowed to its segment. Raises when a segment is
    too short to hold a caption after the margins -- that is a planning
    error, and the plan gate should have said so."""
    events: list[AssEvent] = []
    for cue in cues:
        resolve_color(cue.color_key)  # raises on an unknown key; only white renders today
        seg = timeline.segment(cue.segment_id)
        start, end = seg.start_s + margin_s, seg.end_s - margin_s
        if end <= start:
            raise ValueError(f"segment {cue.segment_id} ({seg.duration_s:.2f}s) cannot hold caption {cue.cue_id}")
        text = "\\N".join(escape(line) for line in break_lines(strip_emoji(cue.text)))
        events.append(AssEvent(start_s=start, end_s=end, style="Main", text=text))
    return events


def render(doc: AssDocument) -> str:
    w, h = doc.play_res
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {w}",
        f"PlayResY: {h}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        *(s.line() for s in doc.styles),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *(e.line() for e in sorted(doc.events, key=lambda e: (e.start_s, e.style))),
    ]
    return "\n".join(head) + "\n"


def write(path: Path | str, doc: AssDocument) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(doc), encoding="utf-8")
    return path
