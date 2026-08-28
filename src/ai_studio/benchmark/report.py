"""The monthly benchmark report: real renders folded into one durable
aggregate per month, grouped by `(kind, gpu_tier)`.

Fed by the daily archive run (`storage.archive.run_archive`) from the same
JSONL it tars; read by `benchmark.rates` and `ai-studio bench`. Lives here,
not in `storage`, because it is the measurement half of this package --
the archive only schedules it.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ai_studio.benchmark.records import BENCHMARK_FIELDS, BENCHMARK_MSGS, RENDER_STAGE
from ai_studio.core.observability import HOT_SUBDIRS, LOCAL_TZ, utc_now_iso

BENCHMARK_DIR = "benchmark"


def _benchmark_month_path(runs_dir: Path, month: str) -> Path:
    return runs_dir / BENCHMARK_DIR / f"{month}.json"


def collect_benchmark_records(log_dir: Path, *, today: date) -> dict[str, list[dict[str, Any]]]:
    """Every real render's benchmark-shaped JSONL line, by local day, from
    day files not yet archived (`day < today` -- the same boundary
    `collect_members` uses; a day still being written must not be read
    here either). Real generation jobs only: a `stage="render"` "fetched
    clip"/"fetched image" line, which `pipeline.drain.render_clip`/
    `render_image` emit for every actual /影片 or /圖片 job that completes --
    never a synthetic run, per this project's own "not a benchmark dataset"
    requirement.
    """
    by_day: dict[str, list[dict[str, Any]]] = {}
    if not log_dir.is_dir():
        return by_day
    for service_dir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
        if service_dir.name in HOT_SUBDIRS:
            continue
        for jsonl in sorted(service_dir.glob("*.jsonl")):
            day = jsonl.stem
            if day >= today.isoformat():
                continue
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("stage") == RENDER_STAGE and record.get("msg") in BENCHMARK_MSGS:
                    by_day.setdefault(day, []).append(record)
    return by_day


def _fold_group(group: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Merge one day's records into a group's running sums.

    Sums, not means: associative, so a day is safe to fold in exactly once
    and this never needs to re-derive a month's total from raw JSONL again --
    which may since have been deleted by `prune_hot`. `days_included` (set by
    the caller) is what makes "exactly once" hold across repeated runs.
    """
    group["count"] = group.get("count", 0) + len(records)
    for metric in BENCHMARK_FIELDS:
        values = [v for r in records if isinstance(v := r.get(metric), int | float)]
        if values:
            group[f"{metric}_sum"] = group.get(f"{metric}_sum", 0.0) + sum(values)
            group[f"{metric}_n"] = group.get(f"{metric}_n", 0) + len(values)
    throughput = [
        r["frames"] / r["seconds"]
        for r in records
        if isinstance(r.get("frames"), int | float)
        and isinstance(r.get("seconds"), int | float)
        and r["seconds"] > 0
    ]
    if throughput:
        group["frames_per_s_sum"] = group.get("frames_per_s_sum", 0.0) + sum(throughput)
        group["frames_per_s_n"] = group.get("frames_per_s_n", 0) + len(throughput)


def _with_means(group: dict[str, Any]) -> dict[str, Any]:
    """A read-friendly copy with `<field>_mean` alongside each running sum --
    this file's whole purpose is a person skimming it to decide whether a
    number is worth promoting, and nobody should have to do that division by
    hand."""
    out = dict(group)
    for metric in (*BENCHMARK_FIELDS, "frames_per_s"):
        n = group.get(f"{metric}_n", 0)
        if n:
            out[f"{metric}_mean"] = round(group[f"{metric}_sum"] / n, 3)
    return out


def update_benchmark_report(
    *, log_dir: Path, runs_dir: Path, today: date | None = None, dry_run: bool = False
) -> dict[str, int]:
    """Fold every real render not yet included into
    `runs/benchmark/<YYYY-MM>.json`, one small durable aggregate per month,
    grouped by `(kind, gpu_tier)`. Each day is folded in exactly once
    (`days_included`, stored in the file itself) as running sums rather than
    recomputed from raw JSONL on every run -- so a month's totals survive
    `prune_hot` deleting the source logs weeks later.

    Same per-month rollover shape as `runtime.budget.SpendLedger`
    (`runs/spend-<YYYY-MM>.json`), which already solves "roll over at a
    month boundary" -- reused rather than re-solved.

    Deliberately a plain data file, not a doc: per this project's "Number
    honesty" rule (CLAUDE.md) a figure only earns a 📏 row in
    `docs/model-h3.md` or a promoted entry in
    `providers.comfyui.MEASURED_LATENCY_S` once a person has actually looked
    at it -- this file is what they look at to decide, not something that
    writes those places itself.

    Returns `{month: days newly folded in}`, for the run's summary line.
    """
    today = today or datetime.now(LOCAL_TZ).date()
    by_day = collect_benchmark_records(log_dir, today=today)
    by_month: dict[str, list[str]] = {}
    for day in by_day:
        by_month.setdefault(day[:7], []).append(day)

    folded: dict[str, int] = {}
    for month, days in by_month.items():
        path = _benchmark_month_path(runs_dir, month)
        payload: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"month": month, "days_included": [], "groups": {}}
        )
        included = set(payload["days_included"])
        new_days = sorted(d for d in days if d not in included)
        if not new_days:
            continue
        for day in new_days:
            by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for record in by_day[day]:
                key = (str(record.get("kind") or "unknown"), str(record.get("gpu_tier") or "unknown"))
                by_group.setdefault(key, []).append(record)
            for (kind, gpu_tier), records in by_group.items():
                group = payload["groups"].setdefault(f"{kind}/{gpu_tier}", {})
                _fold_group(group, records)
            included.add(day)
        folded[month] = len(new_days)
        if dry_run:
            continue
        payload["days_included"] = sorted(included)
        payload["updated_at"] = utc_now_iso()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {**payload, "groups": {k: _with_means(v) for k, v in payload["groups"].items()}},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    return folded
