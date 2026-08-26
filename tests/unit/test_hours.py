"""`runtime.hours` after the business hours went away: what is left is the
timezone the caps count in and the lease a pod gets. Both still have to be
exact -- a lease computed from a naive datetime or a day boundary at UTC
midnight is a pod closed at the wrong time in the money-losing direction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ai_studio.core.errors import AIStudioError
from ai_studio.runtime import hours


def test_the_lease_is_lease_hours_from_now_in_utc() -> None:
    now = datetime(2026, 8, 26, 15, 0, tzinfo=timezone.utc)
    end = hours.window_end_for(now)
    assert end == now + timedelta(hours=hours.LEASE_HOURS)
    assert end.tzinfo is not None and end.utcoffset() == timedelta(0)


def test_the_lease_does_not_care_what_hour_it_is() -> None:
    """There is no shop to be open. 03:00 gets the same lease as 12:00."""
    night = datetime(2026, 8, 26, 3, 0, tzinfo=hours.TZ)
    noon = datetime(2026, 8, 26, 12, 0, tzinfo=hours.TZ)
    assert hours.window_end_for(night) - night == hours.window_end_for(noon) - noon


def test_the_lease_is_short_enough_to_bound_a_dead_worker() -> None:
    """A worker that dies holding a pod costs at most this before the pod
    terminates itself."""
    assert 0 < hours.LEASE_HOURS <= 3


def test_window_end_with_no_argument_uses_now() -> None:
    before = datetime.now(timezone.utc)
    end = hours.window_end_for()
    assert end - before >= timedelta(hours=hours.LEASE_HOURS) - timedelta(seconds=2)


def test_a_naive_datetime_raises_rather_than_being_assumed_utc() -> None:
    with pytest.raises(AIStudioError):
        hours.window_end_for(datetime(2026, 8, 25, 12, 0))
    with pytest.raises(AIStudioError):
        hours.day_start(datetime(2026, 8, 25, 12, 0))


def test_day_start_is_taipei_midnight_not_utc_midnight() -> None:
    # 2026-08-25 01:00 UTC is 09:00 Taipei on the 25th; the day started at
    # 2026-08-24 16:00 UTC.
    when = datetime(2026, 8, 25, 1, 0, tzinfo=timezone.utc)
    assert hours.day_start(when) == datetime(2026, 8, 24, 16, 0, tzinfo=timezone.utc)


def test_day_start_matches_the_ledger_timezone() -> None:
    from ai_studio.runtime.budget import LEDGER_TZ

    assert hours.TZ == LEDGER_TZ
