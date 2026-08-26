"""Business hours. The hardcode, in one file.

"Instant" means *instant inside business hours*: a pod is opened by the first
request that arrives while the shop is open, not by a clock that fires whether
or not anyone asked for anything. That makes 11:00-13:00 Asia/Taipei a fact
three different layers need to agree on -- `bots` writes the out-of-hours
reply, `pipeline` sleeps on it, `runtime` sets the pod's lease to it -- and a
constant that three layers copy is a constant that eventually disagrees with
itself.

It lives in `runtime` (L5) rather than in `session.py` because of the layer
contract, not because it is about pods: `bots` (L6) is a leaf that nobody may
import, so it can reach *down* to here, while nothing here can ever reach up
to it. `pipeline` (L4) sits below `runtime` and so cannot import this at all --
`pipeline.worker` takes these three functions by protocol and the CLI, which
is the composition root, injects them.

The window itself is unchanged and deliberately so: the fixed cost of a
session (boot, weight download, node install) is ~20 minutes against ~5 for a
clip, so the pod is still opened at most once a day. See `docs/schedule.md`.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from ai_studio.core.errors import AIStudioError

TZ = ZoneInfo("Asia/Taipei")
"""The window's timezone, matching `runtime.budget.LEDGER_TZ` -- a session and
the ledger entry it produces must not be able to land on different days."""

OPEN_LOCAL = time(11, 0)
CLOSE_LOCAL = time(13, 0)


class OutsideBusinessHours(AIStudioError):
    """A pod was asked for outside 11:00-13:00.

    Its own type rather than a generic error because the caller's response is
    specific and cheap: hold the request in the queue for the next window. A
    generic failure would be indistinguishable from "the ladder is empty",
    which is not something to retry in sixty seconds.
    """


def _local(now: datetime | None = None) -> datetime:
    """`now` in Taipei. A naive datetime raises rather than being assumed UTC.

    Guessing at the timezone of a naive value is how a window silently opens
    eight hours early, so this fails loudly instead.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise AIStudioError(f"hours needs an aware datetime, got naive {now!r}")
    return now.astimezone(TZ)


def is_open(now: datetime | None = None) -> bool:
    """True from 11:00:00 up to but not including 13:00:00, Taipei time.

    Half-open on purpose: 13:00 is when the shop closes, so 13:00 is closed.
    """
    return OPEN_LOCAL <= _local(now).time() < CLOSE_LOCAL


def window_end_for(now: datetime | None = None) -> datetime:
    """The 13:00 that ends the window `now` falls in, UTC-aware.

    This is the pod's lease end. `open_session` adds `TERMINATE_BUFFER_MIN` to
    it for `--terminate-after`, so a pod opened by a request at 12:40 still
    self-terminates at 13:10 without anyone inventing a second deadline.
    """
    local = _local(now)
    return local.replace(
        hour=CLOSE_LOCAL.hour, minute=CLOSE_LOCAL.minute, second=0, microsecond=0
    ).astimezone(timezone.utc)


def next_open(now: datetime | None = None) -> datetime:
    """The next 11:00, UTC-aware. What the out-of-hours reply quotes.

    Before today's opening it is today's; at or after it, tomorrow's. Someone
    who asks at 23:30 is told 11:00 *tomorrow*, which is the whole point of
    accepting the request instead of refusing it.
    """
    local = _local(now)
    opening = local.replace(
        hour=OPEN_LOCAL.hour, minute=OPEN_LOCAL.minute, second=0, microsecond=0
    )
    if local.time() >= OPEN_LOCAL:
        opening += timedelta(days=1)
    return opening.astimezone(timezone.utc)


def day_start(now: datetime | None = None) -> datetime:
    """Midnight Taipei on the day `now` falls in, UTC-aware.

    The boundary the per-day caps count against. Taipei rather than UTC for the
    same reason the ledger uses it: a UTC rollover happens at 08:00 local,
    which is inside nobody's idea of "today".
    """
    return _local(now).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
