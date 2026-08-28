"""Measured performance: what each real render cost and how long it took,
per GPU tier -- the numbers this project exists to collect.

`records` defines the log line a render emits; `report` folds those lines
into `runs/benchmark/<YYYY-MM>.json`; `rates` reads that file and the open
session back out for display. Nothing in here opens a pod.
"""

from ai_studio.benchmark.rates import GpuRate, live_rate, month_report, tier_stats
from ai_studio.benchmark.records import msg_for, render_record
from ai_studio.benchmark.report import update_benchmark_report

__all__ = [
    "GpuRate",
    "live_rate",
    "month_report",
    "msg_for",
    "render_record",
    "tier_stats",
    "update_benchmark_report",
]
