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

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ai_studio.core.errors import AIStudioError, CostCeilingExceeded

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

    def remaining_this_month_usd(self, *, vps_monthly_usd: float, cap_usd: float) -> float:
        """Deliberately unclamped: a caller must be able to see this has gone
        negative, not have it silently floored to 0."""
        return round(cap_usd - vps_monthly_usd - self.spent_this_month_usd(), 4)


class MonthlyBudgetGuard:
    """Refuses to open a window the month's budget cannot cover, and shrinks
    one it can only partly cover."""

    def __init__(self, ledger: SpendLedger, *, cap_usd: float, vps_monthly_usd: float) -> None:
        self.ledger = ledger
        self.cap_usd = cap_usd
        self.vps_monthly_usd = vps_monthly_usd

    def remaining_usd(self) -> float:
        return self.ledger.remaining_this_month_usd(
            vps_monthly_usd=self.vps_monthly_usd, cap_usd=self.cap_usd
        )

    def refuse_if_broke(self, candidates: tuple[Any, ...]) -> None:
        """Raise if the month's remaining budget cannot cover even a minimal
        session at the priciest rung that might answer.

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
            _log.warning("budget refused", extra={"reason": "month cannot cover a minimal session",
                                                   "spent": round(self.ledger.spent_this_month_usd(), 2), "cap": self.cap_usd})
            raise CostCeilingExceeded(
                f"${remaining:.2f} left this month (cap ${self.cap_usd:.2f}, VPS reserves "
                f"${self.vps_monthly_usd:.2f}) — not enough to safely cover even a "
                f"{MIN_SESSION_MINUTES:.0f}min session at the priciest rung "
                f"(${worst_hourly:.2f}/hr, ${minimal_cost:.2f}). Skipping this window."
            )

    def throttle(
        self, requested_end: datetime, opened_at: datetime, worst_case_hourly_usd: float
    ) -> datetime:
        """Shrink `requested_end` if the month's remaining budget cannot cover
        the full window at the worst-case rate. Never extends it — a cheaper
        rung answering just means the real spend comes in under budget."""
        if worst_case_hourly_usd <= 0:
            return requested_end
        remaining = max(0.0, self.remaining_usd())
        affordable_end = opened_at + timedelta(hours=remaining / worst_case_hourly_usd)
        return min(requested_end, affordable_end)
