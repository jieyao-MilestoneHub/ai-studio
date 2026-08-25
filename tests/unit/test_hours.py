"""Business hours.

These four boundaries are the whole contract, and every one of them is a
money or a silence bug if it moves: an early `is_open` opens a pod nobody
asked for, a late one leaves a request unanswered until tomorrow, and a
`window_end_for` that lands on the wrong day sets a pod's lease 24 hours long.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ai_studio.core.errors import AIStudioError
from ai_studio.runtime import hours


def _tpe(month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, second, tzinfo=hours.TZ)


# ------------------------------------------------------------------- is_open


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (_tpe(8, 25, 10, 59), False),
        (_tpe(8, 25, 10, 59, 59), False),
        (_tpe(8, 25, 11, 0), True),
        (_tpe(8, 25, 12, 59, 59), True),
        (_tpe(8, 25, 13, 0), False),
        (_tpe(8, 25, 13, 1), False),
        (_tpe(8, 25, 3, 0), False),
        (_tpe(8, 25, 23, 30), False),
    ],
)
def test_the_window_is_half_open_on_the_hour(when: datetime, expected: bool) -> None:
    """11:00 opens, 13:00 closes. 13:00 itself is closed."""
    assert hours.is_open(when) is expected


def test_is_open_judges_the_instant_not_the_wall_clock_it_was_handed() -> None:
    """A UTC caller must get the same answer as a Taipei one.

    03:30 UTC is 11:30 in Taipei, which is open — the conversion is the
    function's actual job, so a test that only ever passes Taipei-tagged
    datetimes would not notice if it stopped converting.
    """
    assert hours.is_open(datetime(2026, 8, 25, 3, 30, tzinfo=timezone.utc)) is True
    assert hours.is_open(datetime(2026, 8, 25, 5, 30, tzinfo=timezone.utc)) is False


def test_a_naive_datetime_raises_rather_than_being_assumed_utc() -> None:
    """Guessing is how a window opens eight hours early."""
    with pytest.raises(AIStudioError, match="naive"):
        hours.is_open(datetime(2026, 8, 25, 12, 0))


def test_is_open_with_no_argument_uses_now() -> None:
    """The default path is the one every caller actually uses."""
    assert hours.is_open() is hours.is_open(datetime.now(timezone.utc))


# ------------------------------------------------------------ window_end_for


def test_window_end_is_todays_close_in_utc() -> None:
    end = hours.window_end_for(_tpe(8, 25, 11, 30))
    assert end.tzinfo is timezone.utc
    assert end.astimezone(hours.TZ) == _tpe(8, 25, 13, 0)


def test_window_end_never_lands_on_another_day() -> None:
    """A lease is `--terminate-after`. A day out is a day billed."""
    for hour in (0, 11, 12, 23):
        end = hours.window_end_for(_tpe(8, 25, hour))
        assert end.astimezone(hours.TZ).date() == _tpe(8, 25, hour).date()


def test_window_end_is_at_most_two_hours_after_opening() -> None:
    opened = _tpe(8, 25, 11, 0)
    assert hours.window_end_for(opened) - opened == timedelta(hours=2)


# ------------------------------------------------------------------ next_open


def test_next_open_before_opening_is_today() -> None:
    assert hours.next_open(_tpe(8, 25, 9, 0)).astimezone(hours.TZ) == _tpe(8, 25, 11, 0)


def test_next_open_after_closing_rolls_over_to_tomorrow() -> None:
    """Someone who asks at 23:30 is told 11:00 tomorrow — which is why the
    request is accepted and held rather than refused."""
    assert hours.next_open(_tpe(8, 25, 23, 30)).astimezone(hours.TZ) == _tpe(8, 26, 11, 0)


def test_next_open_crosses_a_month_boundary() -> None:
    assert hours.next_open(_tpe(8, 31, 23, 59)).astimezone(hours.TZ) == _tpe(9, 1, 11, 0)


def test_next_open_inside_the_window_is_tomorrow() -> None:
    """At 12:00 the shop is already open, so the *next* opening is tomorrow's.
    Nothing quotes this value inside the window; pinning it stops the function
    from quietly returning a time in the past."""
    nxt = hours.next_open(_tpe(8, 25, 12, 0))
    assert nxt.astimezone(hours.TZ) == _tpe(8, 26, 11, 0)
    assert nxt > _tpe(8, 25, 12, 0)


# ----------------------------------------------------------------- day_start


def test_day_start_is_taipei_midnight_not_utc_midnight() -> None:
    """UTC midnight is 08:00 in Taipei — inside nobody's idea of "today", and
    the point at which a per-day cap would reset mid-morning."""
    start = hours.day_start(_tpe(8, 25, 12, 0))
    assert start.astimezone(hours.TZ) == _tpe(8, 25, 0, 0)
    assert start == datetime(2026, 8, 24, 16, 0, tzinfo=timezone.utc)


def test_day_start_matches_the_ledger_timezone() -> None:
    """A session and the ledger entry it produces must file on the same day."""
    from ai_studio.runtime.budget import LEDGER_TZ

    assert hours.TZ is LEDGER_TZ or str(hours.TZ) == str(LEDGER_TZ)
