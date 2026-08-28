"""Snapshots of what the GPU has actually done, for the repo's `assets/`.

Three kinds, each a JSON file named `<kind>-<UTC stamp>Z.json`:

- `sessions` -- every pod session this month from the spend ledger, joined
  with the session records under `logs/sessions/`: which tier, how long,
  what it cost, why it closed. Per tier and per session; no pod ids, no
  request content.
- `benchmark` -- the monthly per-`(kind, gpu_tier)` render report
  (`benchmark.report`), when at least one day has been folded. Absent
  otherwise: an empty table is not a measurement.
- `measured` -- `benchmark.measured.MEASURED`, the figures the docs mark 📏.

`render_markdown` turns the latest snapshot of each kind into the block
`ai-studio metrics readme` writes between `<!-- metrics:start -->` and
`<!-- metrics:end -->` in README.md. Nothing in here reads the pod runtime:
the CLI hands the ledger and the session records in.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from ai_studio.benchmark.measured import MEASURED
from ai_studio.benchmark.report import BENCHMARK_DIR

START = "<!-- metrics:start -->"
END = "<!-- metrics:end -->"
KINDS = ("sessions", "benchmark", "measured")


def stamp(when: datetime) -> str:
    """`20260829T031500Z` -- sortable, filename-safe, unambiguous."""
    return when.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------ snapshots


def sessions_snapshot(
    ledger_sessions: Iterable[Mapping[str, Any]],
    session_records: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Per-tier totals and a de-identified per-session list.

    `ledger_sessions` are `runtime.budget.SpendLedger` rows (`date`,
    `tier_label`, `minutes`, `cost_usd`); `session_records` are the closed
    session files under `logs/sessions/` (`tier_label`, `datacenter`,
    `cloud`, `gpu`, `vram_gb`, `quantisation`, `cost_per_hr`, `reason`).
    The two are joined by tier, not by pod: the ledger is the money record,
    the session files add what the tier physically was.
    """
    rows = [dict(s) for s in ledger_sessions]
    records = [dict(r) for r in session_records]
    tiers: dict[str, dict[str, Any]] = {}
    for tier in sorted({str(r.get("tier_label") or "?") for r in rows}):
        mine = [r for r in rows if str(r.get("tier_label") or "?") == tier]
        minutes = [float(r.get("minutes") or 0) for r in mine]
        usd = [float(r.get("cost_usd") or 0) for r in mine]
        recs = [r for r in records if r.get("tier_label") == tier]
        tiers[tier] = {
            "sessions": len(mine),
            "gpu_minutes": round(sum(minutes), 2),
            "gpu_hours": round(sum(minutes) / 60, 2),
            "usd": round(sum(usd), 4),
            "minutes_min": round(min(minutes), 2) if minutes else None,
            "minutes_median": round(median(minutes), 2) if minutes else None,
            "minutes_max": round(max(minutes), 2) if minutes else None,
            "usd_per_hr": sorted({float(r["cost_per_hr"]) for r in recs if r.get("cost_per_hr")}),
            "gpu": sorted({str(r["gpu"]) for r in recs if r.get("gpu")}),
            "datacenters": sorted({str(r["datacenter"]) for r in recs if r.get("datacenter")}),
            "vram_gb": sorted({int(r["vram_gb"]) for r in recs if r.get("vram_gb")}),
            "quantisation": sorted({str(r["quantisation"]) for r in recs if r.get("quantisation")}),
            "closed_because": dict(Counter(str(r.get("reason") or "?") for r in recs)),
        }
    per_session = [
        {
            "date": str(r.get("date") or "")[:10],
            "tier": str(r.get("tier_label") or "?"),
            "minutes": round(float(r.get("minutes") or 0), 2),
            "usd": round(float(r.get("cost_usd") or 0), 4),
        }
        for r in rows
    ]
    return {"tiers": tiers, "sessions": per_session}


def benchmark_snapshot(runs_dir: Path) -> dict[str, Any] | None:
    """Every month's folded report, or None when nothing has been folded --
    the file is simply not written then, rather than written empty."""
    months: dict[str, Any] = {}
    for path in sorted((runs_dir / BENCHMARK_DIR).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("days_included"):
            months[path.stem] = data
    return {"months": months} if months else None


def measured_snapshot() -> dict[str, Any]:
    return {"rows": [m.as_dict() for m in MEASURED]}


def write_snapshots(
    out_dir: Path,
    *,
    when: datetime,
    ledger_sessions: Iterable[Mapping[str, Any]],
    session_records: Iterable[Mapping[str, Any]],
    runs_dir: Path,
) -> list[Path]:
    """Write one file per kind that has data; return what was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = stamp(when)
    generated = when.astimezone(timezone.utc).isoformat(timespec="seconds")
    written: list[Path] = []

    def emit(kind: str, note: str, data: dict[str, Any]) -> None:
        path = out_dir / f"{kind}-{ts}.json"
        path.write_text(
            json.dumps({"generated_at": generated, "kind": kind, "note": note, "data": data},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)

    emit("sessions", "Pod sessions from the spend ledger (this calendar month) joined with "
         "logs/sessions/ by tier. De-identified: no pod ids, no request content.",
         sessions_snapshot(ledger_sessions, session_records))
    emit("measured", "Figures measured on this project's own hardware (the 📏 rows in docs/). "
         "Nothing reported by others, nothing inferred.", measured_snapshot())
    bench = benchmark_snapshot(runs_dir)
    if bench is not None:
        emit("benchmark", "Per-(kind, gpu_tier) means over every real render, folded daily by "
             "`ai-studio archive` from benchmark.records lines.", bench)
    return written


def latest(metrics_dir: Path) -> dict[str, dict[str, Any]]:
    """The newest snapshot of each kind, by the stamp in the file name."""
    found: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        paths = sorted(metrics_dir.glob(f"{kind}-*Z.json"))
        if paths:
            found[kind] = json.loads(paths[-1].read_text(encoding="utf-8"))
    return found


# ------------------------------------------------------------------ rendering


def _num(value: Any, places: int = 1) -> str:
    return "-" if not isinstance(value, int | float) else f"{value:,.{places}f}".rstrip("0").rstrip(".")


def render_markdown(snapshots: Mapping[str, Mapping[str, Any]], *, metrics_dir: str = "assets/metrics") -> str:
    """The README block. Honest about gaps: a kind with no snapshot says so."""
    out: list[str] = []
    stamps = sorted({str(s.get("generated_at", "")) for s in snapshots.values()})
    out.append(f"_Snapshot {stamps[-1] if stamps else 'none'} — RunPod Secure Cloud, our own runs only. "
               f"Sources and notes per row are in the JSON._")
    out.append("")

    # ---- measured
    out.append("**Measured on our own hardware**")
    out.append("")
    measured = snapshots.get("measured", {}).get("data", {}).get("rows", [])
    if measured:
        out.append("| model | GPU | metric | value | on |")
        out.append("|---|---|---|---|---|")
        for r in measured:
            out.append(f"| {r['model']} | {r['gpu']} | {r['metric']} | **{_num(r['value'], 3)} {r['unit']}** | {r['measured_on']} |")
    else:
        out.append("_No measured rows in the snapshot._")
    out.append("")

    # ---- sessions
    out.append("**Pod sessions this month**")
    out.append("")
    tiers = snapshots.get("sessions", {}).get("data", {}).get("tiers", {})
    if tiers:
        out.append("| tier | datacenter | $/hr | sessions | GPU-hours | USD | minutes min / median / max |")
        out.append("|---|---|---|---|---|---|---|")
        for tier, t in tiers.items():
            out.append(
                f"| {tier} | {', '.join(t.get('datacenters') or ['-'])} "
                f"| {', '.join(_num(x, 3) for x in t.get('usd_per_hr') or []) or '-'} | {t['sessions']} "
                f"| {_num(t['gpu_hours'], 2)} | ${_num(t['usd'], 2)} "
                f"| {_num(t['minutes_min'])} / {_num(t['minutes_median'])} / {_num(t['minutes_max'])} |"
            )
    else:
        out.append("_No pod session recorded this month._")
    out.append("")

    # ---- benchmark
    out.append("**Per-render benchmark** (folded daily from real renders)")
    out.append("")
    months = snapshots.get("benchmark", {}).get("data", {}).get("months", {})
    if months:
        out.append("| month | kind / GPU tier | renders | seconds mean | cost mean | VRAM GB mean | frames/s mean |")
        out.append("|---|---|---|---|---|---|---|")
        for month, report in sorted(months.items()):
            for key, g in sorted(report.get("groups", {}).items()):
                out.append(f"| {month} | {key} | {g.get('count', 0)} | {_num(g.get('seconds_mean'))} "
                           f"| ${_num(g.get('cost_usd_mean'), 3)} | {_num(g.get('vram_gb_mean'))} | {_num(g.get('frames_per_s_mean'), 2)} |")
    else:
        out.append("_Nothing folded yet: the per-render record line landed on 2026-08-28 and no pod has "
                   "rendered since. The daily `ai-studio archive` fills `runs/benchmark/<month>.json`; "
                   "the next `metrics export` picks it up._")
    return "\n".join(out)


def update_readme(readme: str, block: str) -> str:
    """Replace what sits between the markers; the rest of the file is untouched.
    Raises if the markers are missing -- silently appending would be worse."""
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(readme):
        raise ValueError(f"README has no {START} ... {END} block to fill")
    return pattern.sub(lambda _: f"{START}\n{block}\n{END}", readme, count=1)
