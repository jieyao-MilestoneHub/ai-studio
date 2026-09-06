"""The monthly spend ledger and guard.

The ladder's own worst case already exceeds $50/month on GPU alone, so a "hard
$50/month cap" is only real if something actually refuses to spend past it —
these tests are that enforcement's own safety net.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_studio.core.errors import AIStudioError, CostCeilingExceeded
from ai_studio.runtime.budget import DailyBudgetGuard, MonthlyBudgetGuard, SpendLedger


@dataclass(frozen=True)
class _Tier:
    usd_per_hr: float


CANDIDATES = (_Tier(1.004), _Tier(0.804), _Tier(0.754), _Tier(0.354))  # price descending


@pytest.fixture
def ledger(tmp_path: Path) -> SpendLedger:
    return SpendLedger(tmp_path / "ledger.json")


# ------------------------------------------------------------------- ledger


def test_a_fresh_ledger_has_spent_nothing(ledger: SpendLedger) -> None:
    assert ledger.spent_this_month_usd() == 0.0


def test_recording_a_session_accumulates(ledger: SpendLedger) -> None:
    ledger.record_session(1.5, tier_label="4090/COMMUNITY", minutes=100)
    ledger.record_session(2.25, tier_label="4090/COMMUNITY", minutes=150)
    assert ledger.spent_this_month_usd() == 3.75


def test_the_ledger_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    SpendLedger(path).record_session(4.0)
    assert SpendLedger(path).spent_this_month_usd() == 4.0


def test_remaining_is_unclamped_so_going_over_is_visible(ledger: SpendLedger) -> None:
    """Flooring a negative remainder to 0 would hide that the cap was crossed."""
    ledger.record_session(100.0)
    remaining = ledger.remaining_this_month_usd(reserved_usd=5.0, cap_usd=50.0)
    assert remaining == -55.0


def test_a_stale_prior_month_is_not_carried_forward(tmp_path: Path) -> None:
    """A month rollover must reset spend, or December's spend would eat into
    January's budget forever."""
    path = tmp_path / "ledger.json"
    path.write_text(
        '{"month": "2020-01", "sessions": [{"date": "2020-01-01T00:00:00+08:00", '
        '"cost_usd": 999.0, "tier_label": "", "minutes": 0}]}',
        encoding="utf-8",
    )
    assert SpendLedger(path).spent_this_month_usd() == 0.0


def test_a_corrupted_ledger_file_raises_rather_than_silently_resetting_to_zero(
    tmp_path: Path,
) -> None:
    """A crash or disk-full write mid-session is exactly the failure mode this
    cap exists to guard against — silently resetting to '$0 spent' here would
    re-grant budget that may already be gone."""
    path = tmp_path / "ledger.json"
    path.write_text("not json at all", encoding="utf-8")
    with pytest.raises(AIStudioError):
        SpendLedger(path).spent_this_month_usd()


def test_a_ledger_missing_the_month_key_raises_too(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text('{"sessions": []}', encoding="utf-8")
    with pytest.raises(AIStudioError):
        SpendLedger(path).spent_this_month_usd()


# -------------------------------------------------------------------- guard


def test_refuse_if_broke_allows_a_healthy_budget(ledger: SpendLedger) -> None:
    guard = MonthlyBudgetGuard(ledger, cap_usd=50.0, vps_monthly_usd=5.0)
    guard.refuse_if_broke(CANDIDATES)  # must not raise


def test_refuse_if_broke_raises_once_the_month_is_essentially_spent(ledger: SpendLedger) -> None:
    ledger.record_session(49.5)  # $50 cap - $5 vps - $49.5 spent = -$4.50 left
    guard = MonthlyBudgetGuard(ledger, cap_usd=50.0, vps_monthly_usd=5.0)
    with pytest.raises(CostCeilingExceeded):
        guard.refuse_if_broke(CANDIDATES)


def test_refuse_if_broke_checks_against_the_priciest_rung(ledger: SpendLedger) -> None:
    """Which rung actually answers is unknown until after the pod is created,
    so the check must be pessimistic against the worst case, not the best.

    $0.15 remaining covers a 20min session at the cheapest rung
    (0.354/hr * 1/3h = $0.118) but not at the priciest (1.004/hr * 1/3h =
    $0.335) -- so this only raises if the guard checks the priciest rung.
    """
    ledger.record_session(44.85)  # $0.15 left
    guard = MonthlyBudgetGuard(ledger, cap_usd=50.0, vps_monthly_usd=5.0)
    with pytest.raises(CostCeilingExceeded):
        guard.refuse_if_broke(CANDIDATES)


def test_refuse_if_broke_finds_the_priciest_rung_even_if_not_listed_first(
    ledger: SpendLedger,
) -> None:
    """Must not trust positional ordering -- if CANDIDATES is ever reordered,
    this should stay pessimistic rather than silently checking the wrong rung."""
    ledger.record_session(44.85)  # $0.15 left, same as the ordered-priciest case above
    guard = MonthlyBudgetGuard(ledger, cap_usd=50.0, vps_monthly_usd=5.0)
    scrambled = (_Tier(0.354), _Tier(0.754), _Tier(1.004), _Tier(0.804))  # priciest is 3rd
    with pytest.raises(CostCeilingExceeded):
        guard.refuse_if_broke(scrambled)


def test_throttle_shrinks_the_window_when_budget_is_tight(ledger: SpendLedger) -> None:
    ledger.record_session(44.0)  # $50 cap - $5 vps - $44 spent = $1.00 left
    guard = MonthlyBudgetGuard(ledger, cap_usd=50.0, vps_monthly_usd=5.0)

    opened_at = datetime.now(timezone.utc)
    requested_end = opened_at + timedelta(hours=2)
    throttled = guard.throttle(requested_end, opened_at, worst_case_hourly_usd=1.004)

    assert throttled < requested_end
    affordable_hours = 1.00 / 1.004
    assert throttled <= opened_at + timedelta(hours=affordable_hours) + timedelta(seconds=1)


def test_throttle_never_extends_the_window_when_budget_is_ample(ledger: SpendLedger) -> None:
    guard = MonthlyBudgetGuard(ledger, cap_usd=50.0, vps_monthly_usd=5.0)

    opened_at = datetime.now(timezone.utc)
    requested_end = opened_at + timedelta(hours=2)
    throttled = guard.throttle(requested_end, opened_at, worst_case_hourly_usd=1.004)

    assert throttled == requested_end


def test_throttle_with_no_budget_left_collapses_to_the_open_instant(ledger: SpendLedger) -> None:
    ledger.record_session(100.0)  # already over cap
    guard = MonthlyBudgetGuard(ledger, cap_usd=50.0, vps_monthly_usd=5.0)

    opened_at = datetime.now(timezone.utc)
    requested_end = opened_at + timedelta(hours=2)
    throttled = guard.throttle(requested_end, opened_at, worst_case_hourly_usd=1.004)

    assert throttled == opened_at


def test_a_rollover_retires_the_old_month_instead_of_discarding_it(tmp_path: Path) -> None:
    """Before 2026-08-28 the 1st of the month destroyed the previous month's
    sessions. Now they go to spend-<YYYY-MM>.json beside the ledger, once,
    and `history` names every retired month."""
    path = tmp_path / "ledger.json"
    path.write_text(
        '{"month": "2020-01", "sessions": [{"date": "2020-01-01T00:00:00+08:00", '
        '"cost_usd": 9.5, "tier_label": "x", "minutes": 60}]}',
        encoding="utf-8",
    )
    ledger = SpendLedger(path)
    assert ledger.spent_this_month_usd() == 0.0

    retired = tmp_path / "spend-2020-01.json"
    assert retired.exists()
    assert json.loads(retired.read_text(encoding="utf-8"))["sessions"][0]["cost_usd"] == 9.5
    ledger.record_session(1.0, tier_label="y", minutes=5)
    assert json.loads(path.read_text(encoding="utf-8"))["history"] == ["2020-01"]
    # reading again must not overwrite the retired file
    before = retired.read_text(encoding="utf-8")
    SpendLedger(path).spent_this_month_usd()
    assert retired.read_text(encoding="utf-8") == before


# ------------------------------------------------------ today's allowance

TPE = timezone(timedelta(hours=8))


def _guards(ledger: SpendLedger, *, now: datetime, open_spend: float = 0.0,
            cap: float = 50.0, vps: float = 5.0, storage: float = 21.0):
    monthly = MonthlyBudgetGuard(
        ledger, cap_usd=cap, vps_monthly_usd=vps, storage_monthly_usd=storage,
        open_spend_usd=lambda _since: open_spend, now=now,
    )
    return monthly, DailyBudgetGuard(monthly)


def test_storage_is_reserved_off_the_top_like_the_vps_is(ledger: SpendLedger) -> None:
    """The guard believed $45 of a $50 cap was available for GPU while the
    network volume was quietly taking $21 of it every month."""
    monthly, _ = _guards(ledger, now=datetime(2026, 9, 1, 12, tzinfo=TPE))
    assert monthly.gpu_month_usd() == 24.0
    assert monthly.remaining_usd() == 24.0
    blind, _ = _guards(ledger, now=datetime(2026, 9, 1, 12, tzinfo=TPE), storage=0.0)
    assert blind.remaining_usd() == 45.0, "dropping the reserve must be loud, not silent"


def test_the_allowance_is_the_month_remainder_spread_over_the_days_left(
    ledger: SpendLedger,
) -> None:
    _, daily = _guards(ledger, now=datetime(2026, 9, 1, 12, tzinfo=TPE))
    assert daily.days_left() == 30
    assert daily.allowance_usd() == 0.8


def test_an_underspent_month_raises_todays_allowance(ledger: SpendLedger) -> None:
    """The point of deriving rather than fixing: an idle first half of the
    month is not forfeited."""
    _, daily = _guards(ledger, now=datetime(2026, 9, 16, 12, tzinfo=TPE))
    assert daily.days_left() == 15
    assert daily.allowance_usd() == 1.6


def test_an_overspent_month_lowers_it(ledger: SpendLedger) -> None:
    ledger.record_session(18.0, when=datetime(2026, 9, 5, 12, tzinfo=TPE))
    _, daily = _guards(ledger, now=datetime(2026, 9, 16, 12, tzinfo=TPE))
    assert daily.allowance_usd() == 0.4


def test_the_last_days_cannot_dump_the_whole_remainder(ledger: SpendLedger) -> None:
    """Without MIN_SPREAD_DAYS the daily guard degenerates into the monthly
    one on the 30th -- one surge could spend everything left."""
    ledger.record_session(12.0, when=datetime(2026, 9, 5, 12, tzinfo=TPE))
    _, daily = _guards(ledger, now=datetime(2026, 9, 30, 12, tzinfo=TPE))
    assert daily.days_left() == 1
    assert daily.allowance_usd() == 3.0  # 12 / max(1, 4), not 12.00


def test_a_month_already_over_cap_allows_nothing_today(ledger: SpendLedger) -> None:
    ledger.record_session(100.0, when=datetime(2026, 9, 5, 12, tzinfo=TPE))
    _, daily = _guards(ledger, now=datetime(2026, 9, 6, 12, tzinfo=TPE))
    assert daily.allowance_usd() == 0.0
    with pytest.raises(CostCeilingExceeded):
        daily.refuse_if_broke(CANDIDATES)


# ------------------------------------------------- the pod that is open now


def test_the_open_pod_counts_against_today_before_it_is_recorded(
    ledger: SpendLedger,
) -> None:
    """A session reaches the ledger only at close_session(), so without this
    the guards are blind to the one pod that is definitely costing money."""
    _, daily = _guards(ledger, now=datetime(2026, 9, 1, 12, tzinfo=TPE), open_spend=0.6)
    assert daily.spent_today_usd() == 0.6
    assert daily.remaining_usd() < 0.2
    with pytest.raises(CostCeilingExceeded):
        daily.refuse_if_broke(CANDIDATES)  # well under 1.004 * 20/60


def test_money_already_spent_today_is_charged_slightly_twice_on_purpose(
    ledger: SpendLedger,
) -> None:
    """Today's spend lowers the month's remainder, and the allowance is a share
    of that remainder -- so it also lowers today's own share by 1/days_left.

    ($24 - $0.60) / 30 = $0.78 rather than a clean $0.80. The overcharge is
    2.5% here and always errs toward refusing, which is the safe direction for
    a ceiling; making it exact would mean adding today's spend back before
    dividing, for an accuracy nobody can perceive. Asserted so the direction
    is a decision on the record rather than a rounding surprise.
    """
    _, daily = _guards(ledger, now=datetime(2026, 9, 1, 12, tzinfo=TPE), open_spend=0.6)
    assert daily.allowance_usd() == 0.78


def test_the_open_pod_also_tightens_the_month(ledger: SpendLedger) -> None:
    monthly, _ = _guards(ledger, now=datetime(2026, 9, 1, 12, tzinfo=TPE), open_spend=3.0)
    assert monthly.spent_this_month_usd() == 3.0
    assert monthly.remaining_usd() == 21.0


def test_no_open_pod_contributes_nothing(ledger: SpendLedger) -> None:
    monthly = MonthlyBudgetGuard(ledger, cap_usd=50.0, vps_monthly_usd=5.0,
                                 storage_monthly_usd=21.0)
    assert monthly.spent_this_month_usd() == 0.0


# ------------------------------------------------------- the day boundary


def test_yesterdays_sessions_do_not_count_against_today(ledger: SpendLedger) -> None:
    ledger.record_session(2.0, when=datetime(2026, 9, 6, 23, 30, tzinfo=TPE))
    assert ledger.spent_on_day_usd(datetime(2026, 9, 7, 0, 30, tzinfo=TPE)) == 0.0


def test_the_day_boundary_is_taipei_not_utc(ledger: SpendLedger) -> None:
    """07:30 Taipei is 23:30 the previous day in UTC. UTC-day arithmetic would
    drop this row from today's total and re-grant its budget."""
    ledger.record_session(2.0, when=datetime(2026, 9, 7, 7, 30, tzinfo=TPE))
    assert ledger.spent_on_day_usd(datetime(2026, 9, 7, 9, 0, tzinfo=TPE)) == 2.0


def test_a_ledger_row_with_an_unparseable_date_counts_against_today(
    ledger: SpendLedger, tmp_path: Path
) -> None:
    """Corruption must never be the reason budget is re-granted."""
    ledger.record_session(2.0, when=datetime(2026, 9, 7, 12, tzinfo=TPE))
    data = json.loads(ledger.path.read_text(encoding="utf-8"))
    data["sessions"][0]["date"] = "not-a-date"
    ledger.path.write_text(json.dumps(data), encoding="utf-8")
    assert ledger.spent_on_day_usd(datetime(2026, 9, 7, 12, tzinfo=TPE)) == 2.0


@pytest.mark.parametrize(
    ("day", "expected"),
    [(datetime(2026, 2, 1, 12, tzinfo=TPE), 28), (datetime(2026, 1, 31, 12, tzinfo=TPE), 1)],
)
def test_days_left_is_inclusive_of_today(ledger: SpendLedger, day: datetime, expected: int) -> None:
    _, daily = _guards(ledger, now=day)
    assert daily.days_left() == expected


# ------------------------------------------------------------- exhaustion


def test_the_day_shrinks_the_lease_instead_of_refusing(ledger: SpendLedger) -> None:
    """Half a window is worth more than none."""
    now = datetime(2026, 9, 1, 12, tzinfo=TPE)
    _, daily = _guards(ledger, now=now, open_spend=0.4)  # $0.40 of $0.80 left
    end = daily.throttle(now + timedelta(hours=2), now, 0.754)
    assert timedelta(minutes=30) < end - now < timedelta(minutes=35)
    daily.refuse_if_broke((_Tier(0.754),))  # 0.40 > 0.754 * 20/60 = 0.251, so it opens


def test_the_day_never_extends_a_lease(ledger: SpendLedger) -> None:
    now = datetime(2026, 9, 1, 12, tzinfo=TPE)
    _, daily = _guards(ledger, now=now, cap=5000.0)
    requested = now + timedelta(hours=2)
    assert daily.throttle(requested, now, 0.754) == requested


def test_the_daily_refusal_names_the_month_it_is_derived_from(ledger: SpendLedger) -> None:
    """A short lease or a refusal is otherwise unexplainable to an operator."""
    _, daily = _guards(ledger, now=datetime(2026, 9, 1, 12, tzinfo=TPE), open_spend=0.79)
    with pytest.raises(CostCeilingExceeded) as caught:
        daily.refuse_if_broke(CANDIDATES)
    message = str(caught.value)
    assert "today's GPU allowance" in message
    assert "30 day(s) left" in message
    assert "midnight Asia/Taipei" in message


def test_every_provider_meters_with_the_rate_it_was_given() -> None:
    """`fetch` must bill at the live session's rate, not a module constant.

    ComfyUIProvider accepted `hourly_usd` and never stored it, so a clip's
    recorded cost was always the $0.74 default. The ladder spans $0.354-$1.004,
    so on other rungs that is -52% to +36% -- and the number feeds
    `chat_spent_this_month_usd()` against a real cap, so it is a budget input,
    not a display string.
    """
    import inspect

    from ai_studio.providers import chat, comfyui, flux, understanding

    for module in (comfyui, flux, understanding, chat):
        body = inspect.getsource(module)
        assert "self._hourly_usd" in body, f"{module.__name__} never keeps its injected rate"
        assert "cost_usd=round(DEFAULT_HOURLY_USD" not in body, (
            f"{module.__name__} meters with the module constant instead of the injected rate"
        )
