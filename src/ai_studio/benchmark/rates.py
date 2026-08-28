"""What a GPU rents for right now, and what it has measured so far.

The data behind any "live score" display: the tier and $/hr of the pod
currently open, plus the per-`(kind, gpu_tier)` means from this month's
report. Deliberately below `runtime` in the layer list -- `live_rate` takes
the session as a duck-typed argument rather than importing it, so the
report and the archive that feeds it stay infrastructure-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_studio.benchmark.report import _benchmark_month_path
from ai_studio.core.observability import LOCAL_TZ


@dataclass(frozen=True)
class GpuRate:
    tier: str
    usd_per_hr: float
    vram_gb: int
    datacenter: str
    since: str
    """ISO timestamp the pod opened at."""


def live_rate(session: Any) -> GpuRate | None:
    """The rate of an open session (`runtime.session.Session`-shaped: needs
    `tier_label`, `cost_per_hr`, `vram_gb`, `datacenter`, `opened_at`), or
    None when there is no session."""
    if session is None:
        return None
    return GpuRate(
        tier=str(session.tier_label),
        usd_per_hr=float(session.cost_per_hr),
        vram_gb=int(session.vram_gb),
        datacenter=str(session.datacenter),
        since=str(session.opened_at),
    )


def _current_month() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m")


def month_report(runs_dir: Path, month: str | None = None) -> dict[str, Any] | None:
    """The report for `month` (default: this month), or None if nothing has
    been folded into it yet."""
    path = _benchmark_month_path(runs_dir, month or _current_month())
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def tier_stats(
    kind: str, gpu_tier: str, runs_dir: Path, month: str | None = None
) -> dict[str, Any] | None:
    """One group's running means (`count`, `seconds_mean`, `cost_usd_mean`,
    `vram_gb_mean`, `frames_per_s_mean` -- whichever have data), or None if
    that `(kind, gpu_tier)` has not rendered anything this month."""
    report = month_report(runs_dir, month)
    if report is None:
        return None
    group = report.get("groups", {}).get(f"{kind}/{gpu_tier}")
    return dict(group) if isinstance(group, dict) else None
