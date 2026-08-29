"""Caption text rules (editing grammar section 4): line length, read speed,
no emoji, white first. Pure text; it does not know what a segment is or
how long it lasts -- the caller passes the seconds it has.

Chinese is counted in code points: a CJK character is a word, so 15 of them
is a full line and 5 a second is a comfortable read. ASCII runs count the
same way, which over-counts Latin text slightly; that errs toward slower
captions, the safe side.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from ai_studio.core.enums import Severity
from ai_studio.core.errors import UnknownKeyError
from ai_studio.core.models import GateFinding

SOURCE_URL = "https://github.com/Hao0321/video-autopilot-kit"

MAX_CHARS_PER_LINE = 15
MAX_LINES = 2
READ_SPEED_WARN = 5.0
READ_SPEED_FAIL = 7.0
"""[reported] section 4.3: chars per second a viewer can read Chinese at."""

MIN_DWELL_S = 0.6
"""[reported] upstream word_captions hold: under this a caption is a flash."""

_BREAK_AFTER = "。！？!?"  # noqa: RUF001 -- fullwidth marks on purpose
_SOFT_BREAK_AFTER = "，、；,;：: "  # noqa: RUF001

COLOR_KEYS = {"w": "&H00FFFFFF"}
"""ASS `&HAABBGGRR`. White is the only key until a palette exists; an
unknown key raises rather than rendering white by accident (section 4.2)."""


def resolve_color(key: str) -> str:
    try:
        return COLOR_KEYS[key]
    except KeyError:
        raise UnknownKeyError("caption colour key", key, COLOR_KEYS) from None


def strip_emoji(text: str) -> str:
    """Section 4.4: no emoji in a caption. Drops pictographs, variation
    selectors and joiners; keeps CJK, Latin, punctuation."""
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in (0xFE0F, 0xFE0E, 0x200D) or 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
            continue
        if unicodedata.category(ch) == "So":
            continue
        out.append(ch)
    return "".join(out).strip()


def break_lines(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> list[str]:
    """Split a caption into at most `MAX_LINES` lines of `max_chars`.

    Breaks after a sentence mark first, then after a clause mark, then hard
    at the limit. Raises when the text cannot fit: a caption that needs
    three lines is two captions, and only the author can split it.
    """
    text = " ".join(text.split())
    if not text:
        raise ValueError("empty caption")
    lines: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_chars:
            lines.append(rest)
            break
        window = rest[: max_chars + 1]
        cut = -1
        for marks in (_BREAK_AFTER, _SOFT_BREAK_AFTER):
            found = max((window.rfind(m) for m in marks), default=-1)
            if found >= 2:  # a break at index 0/1 leaves a one-char line
                cut = found + 1
                break
        if cut <= 0:
            cut = max_chars
        lines.append(rest[:cut].strip())
        rest = rest[cut:].strip()
        if len(lines) == MAX_LINES and rest:
            raise ValueError(f"caption needs more than {MAX_LINES} lines of {max_chars}: {text!r}")
    return lines


def read_speed(text: str, seconds: float) -> float:
    if seconds <= 0:
        raise ValueError(f"caption has no time on screen: {seconds}")
    return len(text.replace(" ", "")) / seconds


def check(cues: Sequence[tuple[str, float, str]]) -> list[GateFinding]:
    """Section 4.2-4.4 as findings over `(text, seconds_available, color_key)`."""
    findings: list[GateFinding] = []

    def add(rule: str, severity: Severity, message: str, *, where: str, observed: object = None,
            expected: object = None) -> None:
        findings.append(GateFinding(
            rule_id=rule, severity=severity, message=message, where=where,
            observed=None if observed is None else str(observed),
            expected=None if expected is None else str(expected), source_url=SOURCE_URL,
        ))

    for i, (text, seconds, color_key) in enumerate(cues):
        where = f"cue {i}"
        if strip_emoji(text) != text.strip():
            add("C-EMOJI", Severity.FAIL, f"{where} contains emoji; captions are text", where=where, observed=text)
        try:
            break_lines(text)
        except ValueError as exc:
            add("C-LINELEN", Severity.FAIL, f"{where}: {exc}", where=where, observed=len(text),
                expected=f"<= {MAX_LINES}x{MAX_CHARS_PER_LINE}")
        if color_key not in COLOR_KEYS:
            add("C-COLOR", Severity.FAIL, f"{where} uses unknown colour key {color_key!r}", where=where,
                observed=color_key, expected=sorted(COLOR_KEYS))
        if seconds <= 0:
            add("C-DWELL", Severity.FAIL, f"{where} has no time on screen", where=where, observed=seconds)
            continue
        if seconds < MIN_DWELL_S:
            add("C-DWELL", Severity.WARN, f"{where} shows for {seconds:.2f}s, a flash", where=where,
                observed=f"{seconds:.2f}", expected=f">= {MIN_DWELL_S}")
        speed = read_speed(text, seconds)
        if speed > READ_SPEED_FAIL:
            add("C-READ-FAIL", Severity.FAIL, f"{where} reads at {speed:.1f} chars/s", where=where,
                observed=f"{speed:.2f}", expected=f"<= {READ_SPEED_FAIL}")
        elif speed > READ_SPEED_WARN:
            add("C-READ-WARN", Severity.WARN, f"{where} reads at {speed:.1f} chars/s", where=where,
                observed=f"{speed:.2f}", expected=f"<= {READ_SPEED_WARN}")
    return findings
