"""`storage.archive`: snapshot, compress, verify, then -- only then -- prune.

The one property every test here defends: nothing is deleted that is not
provably inside a verified archive. Runs on the stdlib lzma path when zstd
is absent; when it is present both paths are exercised."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tarfile
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from ai_studio.benchmark import report as bench
from ai_studio.pipeline.queue import JobQueue
from ai_studio.storage import archive as arc

TODAY = date(2026, 8, 28)


def _tree(root: Path, *, today: date = TODAY) -> dict[str, Path]:
    """A repo-shaped tree with old and fresh material of every kind."""
    log_dir, runs, files, out = root / "logs", root / "runs", root / "files", root / "out"
    old = (today - timedelta(days=45)).isoformat()
    recent = (today - timedelta(days=3)).isoformat()
    paths = {
        "old_jsonl": log_dir / "worker" / f"{old}.jsonl",
        "recent_jsonl": log_dir / "worker" / f"{recent}.jsonl",
        "today_jsonl": log_dir / "worker" / f"{today.isoformat()}.jsonl",
        "session": log_dir / "sessions" / "p1-20260827T162113.json",
        "pod_log": log_dir / "pods" / "p1" / "setup.log",
        "drama_state": runs / "drama" / "tok1" / "state.json",
        "drama_manifest": runs / "drama" / "tok1" / "render_manifest.json",
        "ledger": runs / ".spend_ledger.json",
        "retired": runs / "spend-2026-07.json",
        "index": files / "index.jsonl",
        "stale_dryrun": runs / "_dryrun" / "old.mp4",
        "stale_out": out / "old.mp4",
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"which": "{key}"}}\n', encoding="utf-8")
    stale = time.time() - 60 * 86400
    for key in ("session", "pod_log", "stale_dryrun", "stale_out"):
        os.utime(paths[key], (stale, stale))
    (runs / "drama" / "empty-tok").mkdir(parents=True)
    with JobQueue(runs / "queue.sqlite3") as q:
        q.enqueue("evt-1", "Cgroup", "a cat", user_id="U1")
        q.append_chat_turn("U1", "Cgroup", "user", "old")
        conn = sqlite3.connect(runs / "queue.sqlite3")
        conn.execute("UPDATE chat_turns SET created_at = ?", (stale,))
        conn.commit()
        conn.close()
        q.append_chat_turn("U1", "Cgroup", "user", "fresh")
    paths.update(log_dir=log_dir, runs=runs, files=files, out=out, archive=root / "archive")
    return paths


def _run(root: Path, p: dict[str, Path], *, today: date = TODAY, dry_run: bool = False) -> arc.ArchiveResult:
    return arc.run_archive(
        root=root, log_dir=p["log_dir"], runs_dir=p["runs"], files_dir=p["files"], out_dir=p["out"],
        archive_dir=p["archive"], hot_days=30, keep_days=365, today=today, dry_run=dry_run,
    )


def _members(tar_path: Path) -> set[str]:
    return arc.verify_archive(tar_path)


@pytest.fixture(params=["lzma", "zstd"])
def compressor(request, monkeypatch):
    if request.param == "lzma":
        monkeypatch.setattr(arc.shutil, "which", lambda name: None)
    elif shutil.which("zstd") is None:
        pytest.skip("zstd not on PATH")
    return request.param


def test_a_run_archives_everything_and_prunes_only_what_it_proved(tmp_path: Path, compressor) -> None:
    p = _tree(tmp_path)
    result = _run(tmp_path, p)

    assert result.tar_path is not None and result.tar_path.is_file()
    assert result.tar_path.suffix == (".zst" if compressor == "zstd" else ".xz")
    names = _members(result.tar_path)
    for key in ("old_jsonl", "recent_jsonl", "session", "pod_log", "drama_state", "drama_manifest", "ledger", "retired", "index"):
        assert p[key].relative_to(tmp_path).as_posix() in names, key
    assert p["today_jsonl"].relative_to(tmp_path).as_posix() not in names, "today's file is still being written"
    assert "runs/snapshots/queue-2026-08-28.sqlite3" in names

    manifest = json.loads(result.plan.manifest_path.read_text(encoding="utf-8"))
    assert manifest["tar"] == result.tar_path.name and manifest["host"]
    by_path = {m["path"]: m for m in manifest["members"]}
    assert by_path["files/index.jsonl"]["sha256"] == arc.sha256_file(p["index"])
    assert result.bytes_after > 0 and result.bytes_before > 0

    # hot pruning: only the archived-and-old ones
    assert not p["old_jsonl"].exists() and not p["session"].exists() and not p["pod_log"].exists()
    assert p["recent_jsonl"].exists() and p["today_jsonl"].exists()
    assert result.hot_deleted == 3
    # runs/out sweeps and the empty drama dir
    assert not p["stale_dryrun"].exists() and not p["stale_out"].exists()
    assert not (p["runs"] / "drama" / "empty-tok").exists() and p["drama_state"].exists()
    # chat_turns pruned, jobs intact
    conn = sqlite3.connect(p["runs"] / "queue.sqlite3")
    assert conn.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    conn.close()
    assert result.chat_turns_deleted == 1


def test_the_snapshot_is_a_real_database_not_a_copy_of_a_wal_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(arc.shutil, "which", lambda name: None)
    p = _tree(tmp_path)
    result = _run(tmp_path, p)
    with tarfile.open(result.tar_path, "r:xz") as tar:
        tar.extract("runs/snapshots/queue-2026-08-28.sqlite3", path=tmp_path / "restore", filter="data")
    conn = sqlite3.connect(tmp_path / "restore" / "runs" / "snapshots" / "queue-2026-08-28.sqlite3")
    assert conn.execute("SELECT text FROM jobs").fetchone()[0] == "a cat"
    conn.close()


def test_an_unarchived_file_is_never_deleted(tmp_path: Path, monkeypatch) -> None:
    """The guarantee: a stray old file that no manifest names survives."""
    monkeypatch.setattr(arc.shutil, "which", lambda name: None)
    p = _tree(tmp_path)
    stray = p["log_dir"] / "worker" / "2020-01-01.jsonl"
    # write it AFTER planning would have seen it? no: simulate a manifest that omits it
    _run(tmp_path, p)
    stray.write_text("{}\n", encoding="utf-8")  # appears after the archive; not in any manifest
    second = _run(tmp_path, p)
    assert second.skipped_reason == "already archived today"
    assert stray.exists(), "old but unproven -> kept"


def test_a_second_run_the_same_day_is_a_no_op_for_the_tar(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(arc.shutil, "which", lambda name: None)
    p = _tree(tmp_path)
    first = _run(tmp_path, p)
    second = _run(tmp_path, p)
    assert second.tar_path is None and second.skipped_reason == "already archived today"
    assert sorted(x.name for x in first.tar_path.parent.iterdir()) == sorted(["manifest.json", first.tar_path.name])


def test_old_archives_are_removed_by_keep_days(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(arc.shutil, "which", lambda name: None)
    p = _tree(tmp_path)
    ancient = p["archive"] / (TODAY - timedelta(days=400)).isoformat()
    ancient.mkdir(parents=True)
    (ancient / "manifest.json").write_text('{"members": []}', encoding="utf-8")
    result = _run(tmp_path, p)
    assert result.archives_deleted == 1 and not ancient.exists()
    assert result.plan.manifest_path.exists()


def test_dry_run_writes_and_deletes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(arc.shutil, "which", lambda name: None)
    p = _tree(tmp_path)
    before = sorted(str(x) for x in tmp_path.rglob("*"))
    result = _run(tmp_path, p, dry_run=True)
    assert result.tar_path is None and result.skipped_reason == "dry run"
    assert result.members > 0
    assert sorted(str(x) for x in tmp_path.rglob("*")) == before
    assert not (tmp_path / "archive").exists()


def test_a_verification_gap_discards_the_tar_and_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(arc.shutil, "which", lambda name: None)
    p = _tree(tmp_path)
    monkeypatch.setattr(arc, "verify_archive", lambda path: set())
    with pytest.raises(RuntimeError, match="verification failed"):
        _run(tmp_path, p)
    assert not list((tmp_path / "archive").rglob("*.tar.*"))
    assert p["old_jsonl"].exists(), "nothing pruned when the archive could not be proved"


# ------------------------------------------------------------- benchmarking


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


_CLIP_RECORD = {
    "stage": "render", "msg": "fetched clip", "kind": "video", "gpu_tier": "RTX 4090/COMMUNITY",
    "seconds": 300.0, "cost_usd": 0.06, "vram_gb": 20.5, "frames": 124,
}


def test_update_benchmark_report_folds_real_renders_by_kind_and_tier(tmp_path: Path) -> None:
    log_dir, runs_dir = tmp_path / "logs", tmp_path / "runs"
    _write_jsonl(
        log_dir / "worker" / "2026-08-20.jsonl",
        [
            _CLIP_RECORD,
            {**_CLIP_RECORD, "seconds": 320.0, "cost_usd": 0.065, "vram_gb": 21.0},
            # A non-render line in the same file must not be counted.
            {"stage": "claim", "msg": "claimed", "kind": "video", "gpu_tier": "RTX 4090/COMMUNITY"},
        ],
    )
    folded = bench.update_benchmark_report(log_dir=log_dir, runs_dir=runs_dir, today=TODAY)
    assert folded == {"2026-08": 1}

    payload = json.loads((runs_dir / "benchmark" / "2026-08.json").read_text(encoding="utf-8"))
    assert payload["days_included"] == ["2026-08-20"]
    group = payload["groups"]["video/RTX 4090/COMMUNITY"]
    assert group["count"] == 2
    assert group["seconds_mean"] == pytest.approx(310.0)
    # The stored mean is rounded to 3dp (see `_with_means`), so compare at
    # that same precision rather than the raw float.
    assert group["frames_per_s_mean"] == round((124 / 300 + 124 / 320) / 2, 3)


def test_update_benchmark_report_folds_each_day_at_most_once(tmp_path: Path) -> None:
    log_dir, runs_dir = tmp_path / "logs", tmp_path / "runs"
    _write_jsonl(log_dir / "worker" / "2026-08-20.jsonl", [_CLIP_RECORD])

    first = bench.update_benchmark_report(log_dir=log_dir, runs_dir=runs_dir, today=TODAY)
    second = bench.update_benchmark_report(log_dir=log_dir, runs_dir=runs_dir, today=TODAY)

    assert first == {"2026-08": 1}
    assert second == {}, "the day is already in days_included; nothing new to fold"
    group = json.loads((runs_dir / "benchmark" / "2026-08.json").read_text(encoding="utf-8"))["groups"]
    assert group["video/RTX 4090/COMMUNITY"]["count"] == 1


def test_a_months_total_survives_its_source_log_being_pruned(tmp_path: Path) -> None:
    """Sums, not a recompute from raw JSONL: `prune_hot` deletes the day
    file this same run tars away, and the month's total must not shrink the
    next time this runs."""
    log_dir, runs_dir = tmp_path / "logs", tmp_path / "runs"
    jsonl = log_dir / "worker" / "2026-08-20.jsonl"
    _write_jsonl(jsonl, [_CLIP_RECORD])
    bench.update_benchmark_report(log_dir=log_dir, runs_dir=runs_dir, today=TODAY)

    jsonl.unlink()  # what prune_hot would have done weeks later
    folded = bench.update_benchmark_report(log_dir=log_dir, runs_dir=runs_dir, today=TODAY)

    assert folded == {}
    group = json.loads((runs_dir / "benchmark" / "2026-08.json").read_text(encoding="utf-8"))["groups"]
    assert group["video/RTX 4090/COMMUNITY"]["count"] == 1


def test_update_benchmark_report_dry_run_writes_nothing(tmp_path: Path) -> None:
    log_dir, runs_dir = tmp_path / "logs", tmp_path / "runs"
    _write_jsonl(log_dir / "worker" / "2026-08-20.jsonl", [_CLIP_RECORD])

    folded = bench.update_benchmark_report(log_dir=log_dir, runs_dir=runs_dir, today=TODAY, dry_run=True)

    assert folded == {"2026-08": 1}
    assert not (runs_dir / "benchmark").exists()


def test_update_benchmark_report_ignores_todays_still_open_file(tmp_path: Path) -> None:
    log_dir, runs_dir = tmp_path / "logs", tmp_path / "runs"
    _write_jsonl(log_dir / "worker" / f"{TODAY.isoformat()}.jsonl", [_CLIP_RECORD])

    assert bench.update_benchmark_report(log_dir=log_dir, runs_dir=runs_dir, today=TODAY) == {}


def test_run_archive_folds_benchmarks_from_the_same_jsonl_it_tars(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: a real `archive` run, not just a direct call, produces
    the rollup from the exact worker log it is about to tar away."""
    monkeypatch.setattr(arc.shutil, "which", lambda name: None)
    p = _tree(tmp_path)
    p["old_jsonl"].write_text(json.dumps(_CLIP_RECORD) + "\n", encoding="utf-8")

    result = _run(tmp_path, p)

    assert result.benchmark_folded
    group = json.loads((p["runs"] / "benchmark" / "2026-07.json").read_text(encoding="utf-8"))["groups"]
    assert group["video/RTX 4090/COMMUNITY"]["count"] == 1
