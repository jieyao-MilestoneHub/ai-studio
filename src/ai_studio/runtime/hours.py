"""Time, in one file: the timezone, the day boundary, and the pod's lease.

There are no business hours any more. A pod is opened by any request at any
hour and closed by the idle reaper minutes after the last render -- with the
weights on a network volume a cold open is a ComfyUI restart, so the fixed
11:00-13:00 window that used to amortise a 15-minute download has nothing
left to amortise. What protects money now is the monthly budget guard, the
per-day open cap, and the three closers (`docs/schedule.md`).

What remains here is what several layers still have to agree on:

- `TZ` and `day_start`: the calendar day the per-day caps and the spend
  ledger count against (Taipei, so a UTC rollover at 08:00 local is not
  "tomorrow").
- `LEASE_HOURS` and `window_end_for`: how long a pod is allowed to live once
  opened. It is a backstop, not a schedule: the reaper closes long before
  this, and `--terminate-after` on the pod lands ten minutes after it.

It lives in `runtime` (L5) because `bots` (L6) may reach down to it while
`pipeline` (L4) cannot import it at all -- the worker takes what it needs by
protocol and the CLI injects it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ai_studio.core.errors import AIStudioError

TZ = ZoneInfo("Asia/Taipei")
"""Matches `runtime.budget.LEDGER_TZ` -- a session and the ledger entry it
produces must not be able to land on different days."""

LEASE_HOURS = 2.0
"""How long a freshly opened pod may live. Long enough for a queue of clips at
~100 s each plus a cold open; short enough that a worker that dies holding a
pod costs at most this much before the pod terminates itself."""


def _local(now: datetime | None = None) -> datetime:
    """`now` in Taipei. A naive datetime raises rather than being assumed UTC.

    Guessing at the timezone of a naive value is how a lease silently ends
    eight hours early or late, so this fails loudly instead.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise AIStudioError(f"hours needs an aware datetime, got naive {now!r}")
    return now.astimezone(TZ)


def window_end_for(now: datetime | None = None) -> datetime:
    """When a pod opened at `now` must be gone, UTC-aware: `now` + LEASE_HOURS.

    `open_session` adds `TERMINATE_BUFFER_MIN` to it for `--terminate-after`,
    so nobody invents a second deadline.
    """
    return (_local(now) + timedelta(hours=LEASE_HOURS)).astimezone(timezone.utc)


def day_start(now: datetime | None = None) -> datetime:
    """Midnight Taipei on the day `now` falls in, UTC-aware.

    The boundary the per-day caps count against. Taipei rather than UTC for the
    same reason the ledger uses it: a UTC rollover happens at 08:00 local,
    which is inside nobody's idea of "today".
    """
    return _local(now).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
