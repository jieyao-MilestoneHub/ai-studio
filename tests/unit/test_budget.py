"""The monthly spend ledger and guard.

The ladder's own worst case already exceeds $50/month on GPU alone, so a "hard
$50/month cap" is only real if something actually refuses to spend past it —
these tests are that enforcement's own safety net.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_studio.core.errors import AIStudioError, CostCeilingExceeded
from ai_studio.runtime.budget import MonthlyBudgetGuard, SpendLedger


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
    remaining = ledger.remaining_this_month_usd(vps_monthly_usd=5.0, cap_usd=50.0)
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
