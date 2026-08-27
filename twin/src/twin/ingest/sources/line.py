"""Parse LINE's plain-text chat export ("Export chat history" / トーク履歴を送信).

SPEC.md §6.4 decided LINE as v1's Surface (D27), which makes its own chat export
the natural first real, plain-text data source for Phase 1's minimal ingest.

Format (locale/app-version can shift this — validate against a real export
before trusting it beyond the fixture in tests/unit/test_ingest_line.py, and
prefer widening the regexes here over guessing at a byte-for-byte spec):

    [LINE] <chat name> のトーク履歴
    保存日時：2026/08/27 10:00

    2026/06/15(月)
    10:23\tAlice\t今天天氣真好
    10:24\tBob\t對啊

Each dated block starts with a `YYYY/MM/DD(<weekday>)` line; every following
line until the next date line or end-of-file is `HH:MM<TAB><sender><TAB><message>`.
Timestamps carry no timezone in the export, so they are parsed as naive local
datetimes — callers that need them alongside a cutoff (ingest.split.decide_split)
must pass equally-naive cutoffs, or Python's own datetime comparison will raise
rather than silently compare mismatched awareness.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime

_DATE_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})\([^)]*\)\s*$")
_MESSAGE_RE = re.compile(r"^(\d{2}):(\d{2})\t([^\t]+)\t(.*)$")


@dataclass(frozen=True)
class LineMessage:
    sender: str
    content: str
    sent_at: datetime


def parse_line_export(text: str) -> Iterator[LineMessage]:
    current_date: tuple[int, int, int] | None = None
    pending: LineMessage | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")

        date_match = _DATE_RE.match(line)
        if date_match:
            if pending is not None:
                yield pending
                pending = None
            y, m, d = date_match.groups()
            current_date = (int(y), int(m), int(d))
            continue

        if current_date is None:
            continue  # header lines before the first date block

        message_match = _MESSAGE_RE.match(line)
        if message_match:
            if pending is not None:
                yield pending
            hour, minute, sender, content = message_match.groups()
            year, month, day = current_date
            pending = LineMessage(
                sender=sender,
                content=content,
                sent_at=datetime(year, month, day, int(hour), int(minute)),
            )
            continue

        if line.strip() == "":
            continue

        if pending is not None:
            # A continuation line of a multi-line message — LINE keeps the newline.
            pending = replace(pending, content=pending.content + "\n" + line)
            continue

        raise ValueError(
            f"unrecognised line in LINE export (no active date block, no message "
            f"match, not blank): {line!r}. The export format may differ from what "
            f"this parser expects (see this module's docstring) — check against "
            f"the real export before assuming the data itself is bad."
        )

    if pending is not None:
        yield pending
