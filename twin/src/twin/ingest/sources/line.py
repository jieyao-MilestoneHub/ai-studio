"""Parse LINE's plain-text chat export ("Export chat history" / 傳送聊天記錄).

SPEC.md §6.4 decided LINE as v1's Surface (D27), which makes its own chat export
the natural first real, plain-text data source for Phase 1's minimal ingest.

Format, confirmed against a real Traditional-Chinese-app export (2026-08-29,
twin/PLAN.md Phase 1) — this replaces an earlier, unverified assumption of
slash-separated dates and tab-delimited fields (`[LINE] <chat> のトーク履歴`,
a Japanese-locale header); that guess did not survive contact with a real
file, exactly as this module's docstring always said it might not. Sample
below uses fictional names/content — never real export text, per SPEC.md §8
guardrail 2:

    2026.01.15 星期四
    09:00 Alice Chen 早安
    2026.01.16 星期五
    09:16 Bob 早 Alice，不好意思！今天睡過頭，十點前會抵達
    09:30 Bob已收回訊息
    09:31 已收回訊息

Each dated block starts with a `YYYY.MM.DD <weekday-word>` line (the
weekday's exact spelling is locale-dependent and never parsed — `event_time`
comes from HH:MM plus the date, not from this trailing text). Every
following line until the next date line or end-of-file is one of:

- `HH:MM <sender> <content>` — a normal message. The field separator is a
  single space, not a tab, and — unlike a tab — a bare space cannot itself
  mark where "sender" ends and "content" begins once a participant's own
  display name contains an internal space (a real, observed shape: a
  Chinese name plus an English nickname in one display name, e.g. "Alice
  Chen" above). There is no way to parse this correctly from the text alone, so the
  caller MUST supply the exact, closed set of every participant's display
  name as it appears in *this* export (`known_senders`) — the same "no safe
  default" reasoning `fragments_from_line_export`'s `principal_display_name`
  already uses, for the same underlying reason: guessing wrong here
  silently corrupts every message from that point on, not just one field.
- `HH:MM <sender>已收回訊息` — LINE's recall notice for someone else's
  message. Note there is NO space before "已收回訊息", unlike a real
  message — that absence is exactly what lets this be told apart from the
  general case above (this pattern is only tried after the general message
  pattern fails to match).
- `HH:MM 已收回訊息` — the recall notice for the export owner's own
  message; LINE omits the sender name in this one case, so
  `parse_line_export` resolves it to `principal_display_name` and therefore
  needs that parameter too.

Both recalled-message shapes become a `LineMessage` with content
`"[已收回訊息]"` — recorded, not silently dropped, even though the original
text is gone forever (LINE never exports it): losing the line entirely would
lose the fact that an event happened at all, not just its content.

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

_DATE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\s+\S.*$")
_RECALLED_MARKER = "已收回訊息"
_RECALLED_SELF_RE = re.compile(rf"^(\d{{2}}):(\d{{2}}) {_RECALLED_MARKER}$")


@dataclass(frozen=True)
class LineMessage:
    sender: str
    content: str
    sent_at: datetime


def _sender_alternation(known_senders: list[str]) -> str:
    # Longest-first: a shorter participant name that happens to be a prefix
    # of another's must not win the match.
    return "|".join(re.escape(sender) for sender in sorted(known_senders, key=len, reverse=True))


def parse_line_export(text: str, *, known_senders: list[str], principal_display_name: str) -> Iterator[LineMessage]:
    if not known_senders:
        raise ValueError(
            "known_senders is empty — a bare space cannot disambiguate sender from "
            "content once any participant's display name itself contains a space "
            "(a real, observed shape); there is no safe default here"
        )
    alternation = _sender_alternation(known_senders)
    message_re = re.compile(rf"^(\d{{2}}):(\d{{2}}) ({alternation}) (.*)$")
    recalled_by_re = re.compile(rf"^(\d{{2}}):(\d{{2}}) ({alternation}){_RECALLED_MARKER}$")

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

        year, month, day = current_date

        message_match = message_re.match(line)
        if message_match:
            if pending is not None:
                yield pending
            hour, minute, sender, content = message_match.groups()
            pending = LineMessage(
                sender=sender, content=content, sent_at=datetime(year, month, day, int(hour), int(minute))
            )
            continue

        recalled_by_match = recalled_by_re.match(line)
        if recalled_by_match:
            if pending is not None:
                yield pending
            hour, minute, sender = recalled_by_match.groups()
            pending = LineMessage(
                sender=sender,
                content=f"[{_RECALLED_MARKER}]",
                sent_at=datetime(year, month, day, int(hour), int(minute)),
            )
            continue

        recalled_self_match = _RECALLED_SELF_RE.match(line)
        if recalled_self_match:
            if pending is not None:
                yield pending
            hour, minute = recalled_self_match.groups()
            pending = LineMessage(
                sender=principal_display_name,
                content=f"[{_RECALLED_MARKER}]",
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
            f"this parser expects (see this module's docstring), or `known_senders` "
            f"may be missing a participant — check against the real export before "
            f"assuming the data itself is bad."
        )

    if pending is not None:
        yield pending
