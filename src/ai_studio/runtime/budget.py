"""Monthly spend ledger and guard.

The ladder's own worst case already exceeds $50/month on GPU alone
(`docs/schedule.md`: rung 1, $1.004/hr x 2h x 30d = $60.24), before the VPS or
LLM serverless cost is added — so a "hard $50/month cap" needs real
enforcement, not favourable pricing. This module is that enforcement: a small
JSON ledger of what each session actually cost, and a guard that refuses to
open a new window (or shrinks one) once the month's budget is running out.

Kept deliberately dependency-light — only `config.settings` and `core.errors`,
both below `runtime` in the layer list — so it is trivially unit-testable with
no queue, no pod, no network.
"""

from __future__ import annotations

import calendar
import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ai_studio.core.errors import AIStudioError, CostCeilingExceeded
from ai_studio.runtime import hours

_log = logging.getLogger("ai_studio.budget")

LEDGER_TZ = ZoneInfo("Asia/Taipei")
"""The cap is a human/billing concept tied to the same timezone the service
window itself is scheduled in — a UTC-midnight rollover would occasionally
misfile a session into the wrong month."""

DEFAULT_LEDGER_FILE = Path("runs/.spend_ledger.json")

MIN_SESSION_MINUTES = 20.0
"""The fixed cost of any session (boot, weight download, node install) per
`runtime.session`'s own docstring — the smallest amount of GPU time a window
open can ever actually cost, even before anything is rendered."""

MIN_SPREAD_DAYS = 4
"""The daily allowance never divides the month's remainder by fewer than this
many days.

Without a floor, `remaining / days_left` degenerates into the monthly guard on
the last day of the month: one surge on the 31st could spend everything left,
which is exactly the unpredictable degradation the daily cap exists to remove.
Four days means no single day may take more than a quarter of what is left. It
strands a little money at month end — spending the full allowance every day
converges rather than exhausting — and that is the safe direction for a
ceiling."""


class SpendLedger:
    """A durable, month-scoped record of what each window actually cost."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or DEFAULT_LEDGER_FILE)  # resolved late, so tests can redirect it

    def _month_key(self, when: datetime | None = None) -> str:
        when = (when or datetime.now(timezone.utc)).astimezone(LEDGER_TZ)
        return when.strftime("%Y-%m")

    def _retire(self, data: dict[str, Any]) -> None:
        """Write a finished month to `spend-<YYYY-MM>.json` beside the ledger,
        once; never raises (a failure here must not block a session close)."""
        try:
            dest = self.path.with_name(f"spend-{data['month']}.json")
            if not dest.exists():
                dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                _log.info("ledger month retired", extra={"reason": str(data["month"])})
        except Exception as exc:
            _log.warning("could not retire ledger month %s: %s", data.get("month"), exc)

    def _read(self) -> dict[str, Any]:
        fresh: dict[str, Any] = {"month": self._month_key(), "sessions": []}
        if not self.path.is_file():
            return fresh
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # A crash or a disk-full write mid-session is exactly the failure
            # mode this cap exists to guard against — silently resetting to
            # "$0 spent" here would re-grant budget that may already be gone.
            # A legitimate new month (below) is not this: that's a valid,
            # recognisable state, not corruption.
            raise AIStudioError(
                f"{self.path} is corrupted (invalid JSON) — refusing to silently reset "
                f"the monthly spend ledger to $0, which could re-grant already-spent "
                f"budget. Inspect it and fix or delete it deliberately: {exc}"
            ) from exc
        if not isinstance(data, dict) or "month" not in data:
            raise AIStudioError(
                f"{self.path} is not a recognisable ledger (missing or malformed "
                f"'month' key) — refusing to silently reset the monthly spend budget."
            )
        if data["month"] != fresh["month"]:
            # A genuine rollover to a new month — spend legitimately resets,
            # but the old month is not thrown away any more (it was, before
            # 2026-08-28): it goes to a sibling file the archive picks up,
            # and `history` names every month that has one.
            self._retire(data)
            fresh["history"] = sorted({*data.get("history", []), str(data["month"])})
            return fresh
        data.setdefault("sessions", [])
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record_session(
        self,
        cost_usd: float,
        *,
        tier_label: str = "",
        minutes: float = 0.0,
        when: datetime | None = None,
    ) -> None:
        data = self._read()
        when = when or datetime.now(timezone.utc)
        data["sessions"].append(
            {
                "date": when.astimezone(LEDGER_TZ).isoformat(),
                "cost_usd": round(cost_usd, 4),
                "tier_label": tier_label,
                "minutes": round(minutes, 2),
            }
        )
        self._write(data)

    def sessions(self) -> list[dict[str, Any]]:
        """This month's recorded sessions (`date`, `cost_usd`, `tier_label`,
        `minutes`), oldest first. Read-only; what `ai-studio metrics` exports."""
        return [dict(s) for s in self._read()["sessions"]]

    def spent_this_month_usd(self) -> float:
        data = self._read()
        return round(sum(float(s["cost_usd"]) for s in data["sessions"]), 4)

    def _day_key(self, when: datetime | None = None) -> str:
        when = (when or datetime.now(timezone.utc)).astimezone(LEDGER_TZ)
        return when.strftime("%Y-%m-%d")

    def spent_on_day_usd(self, when: datetime | None = None) -> float:
        """What the ledger recorded for the Asia/Taipei day `when` falls in.

        A row whose `date` will not parse is counted against *today* — the same
        stance `_read()` takes about corruption: a ledger that cannot be read
        must never be the reason budget is re-granted.
        """
        want = self._day_key(when)
        total = 0.0
        for session in self._read()["sessions"]:
            try:
                stamp = datetime.fromisoformat(str(session["date"]))
            except (ValueError, KeyError, TypeError):
                total += float(session.get("cost_usd", 0.0))
                continue
            if self._day_key(stamp) == want:
                total += float(session["cost_usd"])
        return round(total, 4)

    def remaining_this_month_usd(self, *, reserved_usd: float, cap_usd: float) -> float:
        """Deliberately unclamped: a caller must be able to see this has gone
        negative, not have it silently floored to 0.

        `reserved_usd` is every fixed monthly cost the cap must cover before a
        GPU-second is bought — the always-on host *and* the network volume,
        which bills whether or not a pod exists.
        """
        return round(cap_usd - reserved_usd - self.spent_this_month_usd(), 4)


class _BudgetGuard:
    """What the month guard and the day guard share: how to refuse, and how to
    shrink a lease.

    Both answer the same two questions against a different `remaining_usd()`,
    so the pessimism (`max()` over the ladder) and the never-extend property of
    `throttle` are written once rather than twice and drifting.
    """

    ledger: SpendLedger
    _open_spend: Callable[[datetime], float]
    _now_at: datetime | None

    def _now(self) -> datetime:
        return self._now_at or datetime.now(timezone.utc)

    def remaining_usd(self) -> float:  # pragma: no cover - subclass
        raise NotImplementedError

    def _refusal(self, worst_hourly: float, minimal_cost: float) -> str:  # pragma: no cover
        raise NotImplementedError

    def _scope(self) -> str:  # pragma: no cover - subclass
        raise NotImplementedError

    def refuse_if_broke(self, candidates: tuple[Any, ...]) -> None:
        """Raise if the remaining budget cannot cover even a minimal session at
        the priciest rung that might answer.

        Checked against the priciest rung deliberately: which rung actually
        answers is not known until `open_session()` has already created the
        pod, by which point `--terminate-after` is already set. Refusing
        before that point, pessimistically, is the only way to guarantee the
        cap is never crossed. Computed with `max()` rather than trusting
        `candidates[0]` to be priciest — the ladder is documented as
        price-descending and a test enforces that, but this guard should not
        silently go optimistic if that ordering is ever disturbed.
        """
        remaining = self.remaining_usd()
        worst_hourly = max(c.usd_per_hr for c in candidates)
        minimal_cost = worst_hourly * MIN_SESSION_MINUTES / 60.0
        if remaining < minimal_cost:
            _log.warning(
                "budget refused",
                extra={"reason": f"{self._scope()} cannot cover a minimal session",
                       "spent": round(self.ledger.spent_this_month_usd(), 2)},
            )
            raise CostCeilingExceeded(self._refusal(worst_hourly, minimal_cost))

    def throttle(
        self, requested_end: datetime, opened_at: datetime, worst_case_hourly_usd: float
    ) -> datetime:
        """Shrink `requested_end` if the remaining budget cannot cover the full
        window at the worst-case rate. Never extends it — a cheaper rung
        answering just means the real spend comes in under budget."""
        if worst_case_hourly_usd <= 0:
            return requested_end
        remaining = max(0.0, self.remaining_usd())
        affordable_end = opened_at + timedelta(hours=remaining / worst_case_hourly_usd)
        return min(requested_end, affordable_end)


class MonthlyBudgetGuard(_BudgetGuard):
    """Refuses to open a window the month's budget cannot cover, and shrinks
    one it can only partly cover."""

    def __init__(
        self,
        ledger: SpendLedger,
        *,
        cap_usd: float,
        vps_monthly_usd: float,
        storage_monthly_usd: float = 0.0,
        open_spend_usd: Callable[[datetime], float] | None = None,
        now: datetime | None = None,
    ) -> None:
        self.ledger = ledger
        self.cap_usd = cap_usd
        self.vps_monthly_usd = vps_monthly_usd
        self.storage_monthly_usd = storage_monthly_usd
        # What the pod that is billing *right now* has spent. The ledger only
        # learns a session's cost at close_session(), so without this every
        # guard is blind to the one pod that is definitely costing money.
        # Injected rather than imported: `runtime.session` imports this module,
        # so reaching back for `load_state()` here would be a cycle.
        self._open_spend = open_spend_usd or (lambda _since: 0.0)
        self._now_at = now

    @property
    def reserved_usd(self) -> float:
        """Every fixed monthly cost the cap must cover before a GPU-second."""
        return self.vps_monthly_usd + self.storage_monthly_usd

    def gpu_month_usd(self) -> float:
        """What the month has for GPU at all, reserves removed. Display only."""
        return round(max(0.0, self.cap_usd - self.reserved_usd), 4)

    def spent_this_month_usd(self) -> float:
        return round(
            self.ledger.spent_this_month_usd() + self._open_spend(hours.month_start(self._now())),
            4,
        )

    def remaining_usd(self) -> float:
        return round(self.cap_usd - self.reserved_usd - self.spent_this_month_usd(), 4)

    def _scope(self) -> str:
        return "month"

    def _refusal(self, worst_hourly: float, minimal_cost: float) -> str:
        return (
            f"${self.remaining_usd():.2f} left this month (cap ${self.cap_usd:.2f}, reserves "
            f"VPS ${self.vps_monthly_usd:.2f} + storage ${self.storage_monthly_usd:.2f}) — "
            f"not enough to safely cover even a {MIN_SESSION_MINUTES:.0f}min session at the "
            f"priciest rung (${worst_hourly:.2f}/hr, ${minimal_cost:.2f}). Skipping this window."
        )


class DailyBudgetGuard(_BudgetGuard):
    """How much may still be spent today — derived from the month, never set.

    A hand-set daily number is a second budget that never reconciles with the
    first: every idle day silently forfeits its share, so the cap configured is
    not the cap you get, and a day that overspends is left for the monthly
    guard to punish three weeks later. Both failures are the "silent, then
    sudden" shape this guard exists to remove. Spreading what the month has
    left over the days it has left self-corrects in both directions from
    numbers the ledger already holds.

    📏 Backtested against the real ledger (10 days with usage, mean $1.49, max
    $4.57): a flat $24/30 = $0.80/day would have refused 6 of those 10 days.
    The derived form gives $4.00/day when the month is young and untouched,
    and tightens only as it is actually spent.
    """

    def __init__(
        self, monthly: MonthlyBudgetGuard, *, min_spread_days: int = MIN_SPREAD_DAYS
    ) -> None:
        self.monthly = monthly
        self.ledger = monthly.ledger
        self.min_spread_days = min_spread_days
        self._open_spend = monthly._open_spend
        self._now_at = monthly._now_at

    def days_left(self) -> int:
        """Days remaining in the Taipei month, today included."""
        local = self._now().astimezone(LEDGER_TZ)
        return calendar.monthrange(local.year, local.month)[1] - local.day + 1

    def allowance_usd(self) -> float:
        """Today's share of what the month has left."""
        month_left = max(0.0, self.monthly.remaining_usd())
        return round(month_left / max(self.days_left(), self.min_spread_days), 4)

    def spent_today_usd(self) -> float:
        return round(
            self.ledger.spent_on_day_usd(self._now())
            + self._open_spend(hours.day_start(self._now())),
            4,
        )

    def remaining_usd(self) -> float:
        """Unclamped, like the month's: an overspent day must be visible."""
        return round(self.allowance_usd() - self.spent_today_usd(), 4)

    def _scope(self) -> str:
        return "today"

    def _refusal(self, worst_hourly: float, minimal_cost: float) -> str:
        return (
            f"today's GPU allowance is ${self.allowance_usd():.2f} and "
            f"${self.spent_today_usd():.2f} of it is gone (${self.remaining_usd():.2f} left) — "
            f"not enough for a {MIN_SESSION_MINUTES:.0f}min session at the priciest rung "
            f"(${worst_hourly:.2f}/hr, ${minimal_cost:.2f}). The month still has "
            f"${self.monthly.remaining_usd():.2f}; the allowance is that spread over the "
            f"{self.days_left()} day(s) left. Try again after midnight Asia/Taipei."
        )
