"""`benchmark.export`: timestamped, de-identified snapshots and README's block."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_studio.benchmark import export
from ai_studio.benchmark.measured import MEASURED

WHEN = datetime(2026, 8, 29, 3, 15, 0, tzinfo=timezone.utc)
LEDGER = [
    {"date": "2026-08-26T22:30:20+08:00", "cost_usd": 0.1769, "tier_label": "RTX 4090/SECURE", "minutes": 14.35},
    {"date": "2026-08-27T02:03:10+08:00", "cost_usd": 0.5589, "tier_label": "RTX 4090/SECURE", "minutes": 45.32},
]
RECORDS = [
    {"pod_id": "secret", "tier_label": "RTX 4090/SECURE", "gpu": "NVIDIA GeForce RTX 4090", "datacenter": "EUR-IS-1",
     "cloud": "SECURE", "cost_per_hr": 0.74, "vram_gb": 24, "quantisation": "int8", "reason": "idle"},
]


def test_snapshots_are_stamped_and_benchmark_is_only_written_with_data(tmp_path: Path) -> None:
    written = export.write_snapshots(
        tmp_path / "m", when=WHEN, ledger_sessions=LEDGER, session_records=RECORDS, runs_dir=tmp_path / "runs",
    )
    assert sorted(p.name for p in written) == ["measured-20260829T031500Z.json", "sessions-20260829T031500Z.json"]
    sessions = json.loads(written[0].read_text(encoding="utf-8")) if written[0].name.startswith("sessions") \
        else json.loads(written[1].read_text(encoding="utf-8"))
    assert sessions["kind"] == "sessions" and sessions["generated_at"].startswith("2026-08-29T03:15:00")
    tier = sessions["data"]["tiers"]["RTX 4090/SECURE"]
    assert tier["sessions"] == 2 and tier["usd"] == pytest.approx(0.7358) and tier["usd_per_hr"] == [0.74]
    assert tier["closed_because"] == {"idle": 1} and tier["minutes_median"] == pytest.approx(29.84, abs=0.01)
    assert "secret" not in json.dumps(sessions), "no pod id leaves the snapshot"

    # a folded month -> the benchmark file appears
    bench = tmp_path / "runs" / "benchmark"
    bench.mkdir(parents=True)
    (bench / "2026-08.json").write_text(json.dumps({"month": "2026-08", "days_included": ["2026-08-28"],
                                                     "groups": {"video/RTX 4090/SECURE": {"count": 3, "seconds_mean": 81.0}}}), encoding="utf-8")
    written = export.write_snapshots(
        tmp_path / "m", when=WHEN, ledger_sessions=LEDGER, session_records=RECORDS, runs_dir=tmp_path / "runs",
    )
    assert any(p.name == "benchmark-20260829T031500Z.json" for p in written)


def test_the_measured_table_is_our_own_numbers_only() -> None:
    rows = export.measured_snapshot()["rows"]
    assert len(rows) == len(MEASURED) and all(r["measured_on"].startswith("2026-") for r in rows)
    assert all(r["source"].startswith("docs/") for r in rows)


def test_readme_block_is_replaced_in_place_and_idempotent(tmp_path: Path) -> None:
    export.write_snapshots(tmp_path / "m", when=WHEN, ledger_sessions=LEDGER, session_records=RECORDS, runs_dir=tmp_path / "runs")
    block = export.render_markdown(export.latest(tmp_path / "m"))
    assert "Nothing folded yet" in block and "RTX 4090/SECURE" in block
    readme = "# x\n\nintro\n\n<!-- metrics:start -->\nSTALE-BLOCK\n<!-- metrics:end -->\n\nafter\n"
    once = export.update_readme(readme, block)
    assert once.startswith("# x\n\nintro\n\n<!-- metrics:start -->\n") and once.endswith("<!-- metrics:end -->\n\nafter\n")
    assert "STALE-BLOCK" not in once and export.update_readme(once, block) == once
    with pytest.raises(ValueError):
        export.update_readme("no markers", block)
