"""`benchmark.rates`: the live GPU rate and this month's per-tier means."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ai_studio.benchmark import live_rate, month_report, tier_stats
from ai_studio.benchmark.records import msg_for, render_record


def test_live_rate_reads_a_session_shaped_object() -> None:
    session = SimpleNamespace(
        tier_label="L40S/OC-AU-1/SECURE", cost_per_hr=1.004, vram_gb=48,
        datacenter="OC-AU-1", opened_at="2026-08-28T01:02:03+00:00",
    )
    rate = live_rate(session)
    assert rate is not None
    assert (rate.tier, rate.usd_per_hr, rate.vram_gb) == ("L40S/OC-AU-1/SECURE", 1.004, 48)
    assert live_rate(None) is None


def test_tier_stats_reads_the_month_report(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "benchmark").mkdir(parents=True)
    (runs / "benchmark" / "2026-08.json").write_text(json.dumps({
        "month": "2026-08", "days_included": ["2026-08-27"],
        "groups": {"video/rtx4090": {"count": 3, "seconds_mean": 181.2}},
    }), encoding="utf-8")

    assert tier_stats("video", "rtx4090", runs, "2026-08") == {"count": 3, "seconds_mean": 181.2}
    assert tier_stats("image", "rtx4090", runs, "2026-08") is None
    assert month_report(runs, "2026-07") is None


def test_render_record_is_the_shape_the_fold_reads() -> None:
    record = render_record("video", seconds=12.34, polls=4, cost_usd=0.5, vram_gb=20.1,
                           gpu_tier="t", frames=97)
    assert record["stage"] == "render" and record["seconds"] == 12.3 and record["frames"] == 97
    assert "frames" not in render_record("image", seconds=1, polls=0, cost_usd=None, vram_gb=None, gpu_tier=None)
    assert msg_for("video") == "fetched clip" and msg_for("image") == "fetched image"
