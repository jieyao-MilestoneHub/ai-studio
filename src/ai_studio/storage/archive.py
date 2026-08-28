"""The daily archive: snapshot, compress, verify, then -- and only then -- prune.

What the always-on host accumulates and where it would otherwise go:

- `logs/<service>/<day>.jsonl`, `logs/sessions/`, `logs/pods/` -- the trace;
  nothing rotated them (no logrotate on the box).
- `runs/queue.sqlite3` -- the audit trail; WAL, never checkpointed, and
  `sqlite3` the CLI is not installed, so `cp` would be undefined behaviour.
- `runs/drama/<token>/state.json` + `render_manifest.json`, the spend ledger
  and its retired months, `files/index.jsonl`.

One run, every day at 03:00 Asia/Taipei (`deploy/jetson_setup.sh`):

1. snapshot the queue with `sqlite3.Connection.backup()` (then checkpoint
   the WAL so the live file stops growing);
2. stream a tar of every member into `zstd -19 -T0`
   (`archive/<day>/ai-studio-<local stamp>.tar.zst`; `lzma` `.tar.xz` when
   `zstd` is not on PATH -- the tests take that path), and list it back;
3. write `manifest.json` next to it: when, host, git sha, every member with
   size and sha256, bytes before/after;
4. fold any real /影片 or /圖片 render since the last run into
   `runs/benchmark/<YYYY-MM>.json` (`update_benchmark_report`) -- a small
   durable aggregate of what this project's own GPU actually does, by
   `(kind, gpu_tier)`, read off the same JSONL this run just collected for
   the tar. **Must run before step 5**: a backlog older than `log_hot_days`
   gets tarred, proven, and deleted in that one step, and a fold that ran
   after would find nothing left to read for those days;
5. prune -- hot logs older than `log_hot_days` **only if a manifest names
   them** (never anything unarchived); archives older than
   `archive_keep_days`; `runs/_dryrun`, `runs/_stub`, `out/` at 30 d; empty
   `runs/drama/<token>` dirs; `chat_turns` rows older than 30 d; VACUUM.

Idempotent: a second run the same day finds the manifest, skips the tar and
still prunes; the benchmark rollup tracks its own `days_included` per month
and simply has nothing new to fold. Never deletes a file it cannot prove is
inside a verified archive. Restore: `tar -I zstd -xf <tar> -C <dir>`.

Decisions (2026-08-28): same disk (one NVMe, no NAS; an off-box push only
needs a destination added); hot 30 days, archive 365.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import socket
import sqlite3
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_studio.core.observability import LOCAL_TZ, local_now_iso, utc_now_iso
from ai_studio.storage.base import sha256_file
from ai_studio.storage.retention import SweepResult, sweep_old_files

_log = logging.getLogger("ai_studio.archive")

RUNS_SWEEP_DAYS = 30.0
CHAT_TURNS_KEEP_DAYS = 30.0
HOT_SUBDIRS = ("sessions", "pods")
"""Under log_dir, besides the per-service JSONL directories."""


@dataclass(frozen=True)
class ArchivePlan:
    day: str
    tar_path: Path
    manifest_path: Path
    members: list[Path]
    sqlite_source: Path | None
    already_archived: bool

    def summary(self) -> str:
        state = "already archived today" if self.already_archived else "new"
        return f"archive {self.day}: {len(self.members)} member(s), {state} -> {self.tar_path.name}"


@dataclass
class ArchiveResult:
    plan: ArchivePlan
    tar_path: Path | None = None
    members: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    hot_deleted: int = 0
    archives_deleted: int = 0
    chat_turns_deleted: int = 0
    drama_dirs_removed: int = 0
    swept: dict[str, SweepResult] = field(default_factory=dict)
    benchmark_folded: dict[str, int] = field(default_factory=dict)
    """`{month: days newly folded in}` from `update_benchmark_report`."""
    skipped_reason: str | None = None
    dry_run: bool = False

    def summary(self) -> str:
        head = "archive (dry run)" if self.dry_run else "archive"
        if self.tar_path is not None:
            ratio = f"{self.bytes_before / 1_048_576:.1f} -> {self.bytes_after / 1_048_576:.1f} MB"
            made = f"wrote {self.tar_path.name} ({self.members} members, {ratio})"
        else:
            made = f"no tar ({self.skipped_reason or 'nothing to archive'})"
        swept = sum(r.removed for r in self.swept.values())
        benchmark_days = sum(self.benchmark_folded.values())
        return (
            f"{head}: {made}; pruned hot={self.hot_deleted} archives={self.archives_deleted} "
            f"runs={swept} drama-dirs={self.drama_dirs_removed} chat_turns={self.chat_turns_deleted}; "
            f"benchmark days folded={benchmark_days}"
        )


# ------------------------------------------------------------------ planning


def local_today() -> date:
    return datetime.now(LOCAL_TZ).date()


def collect_members(
    *, log_dir: Path, runs_dir: Path, files_dir: Path, today: date
) -> list[Path]:
    """Everything worth keeping that is not media. Today's JSONL files are
    still being written to and are left for tomorrow's run."""
    members: list[Path] = []
    if log_dir.is_dir():
        for service_dir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
            if service_dir.name in HOT_SUBDIRS:
                members += sorted(p for p in service_dir.rglob("*") if p.is_file())
                continue
            for jsonl in sorted(service_dir.glob("*.jsonl")):
                day = jsonl.stem
                if day < today.isoformat():
                    members.append(jsonl)
    drama = runs_dir / "drama"
    if drama.is_dir():
        for token_dir in sorted(p for p in drama.iterdir() if p.is_dir()):
            for name in ("state.json", "render_manifest.json"):
                if (token_dir / name).is_file():
                    members.append(token_dir / name)
    for name in (".spend_ledger.json", ".session.json", ".reap_last.json"):
        if (runs_dir / name).is_file():
            members.append(runs_dir / name)
    members += sorted(runs_dir.glob("spend-*.json"))
    index = files_dir / "index.jsonl"
    if index.is_file():
        members.append(index)
    return members


def plan_archive(
    *, log_dir: Path, runs_dir: Path, files_dir: Path, archive_dir: Path, today: date | None = None
) -> ArchivePlan:
    today = today or local_today()
    day_dir = archive_dir / today.isoformat()
    manifest = day_dir / "manifest.json"
    stamp = datetime.now(LOCAL_TZ).strftime("%Y-%m-%dT%H%M%S%z")
    suffix = ".tar.zst" if shutil.which("zstd") else ".tar.xz"
    tar_path = day_dir / f"ai-studio-{stamp}{suffix}"
    members = collect_members(log_dir=log_dir, runs_dir=runs_dir, files_dir=files_dir, today=today)
    db = runs_dir / "queue.sqlite3"
    return ArchivePlan(
        day=today.isoformat(),
        tar_path=tar_path,
        manifest_path=manifest,
        members=members,
        sqlite_source=db if db.is_file() else None,
        already_archived=manifest.is_file(),
    )


# ------------------------------------------------------------------ writing


def snapshot_queue(db: Path, dest: Path) -> Path:
    """A consistent copy of a live WAL database, then a checkpoint so the
    live `-wal` file is truncated. `sqlite3.Connection.backup()` is the only
    correct way to copy a database with readers/writers attached."""
    src = sqlite3.connect(db, timeout=30.0)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
        try:
            src.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError as exc:  # a writer holds it; harmless
            _log.warning("wal checkpoint skipped: %s", exc)
    finally:
        src.close()
    return dest


def write_tar(members: list[Path], *, root: Path, dest: Path, extra: dict[str, Path]) -> tuple[int, int]:
    """Stream `members` (stored relative to `root`) plus `extra` ({arcname:
    path}) into `dest`. zstd via subprocess when the suffix says so, else
    stdlib lzma. Returns (bytes before, bytes after)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    before = sum(p.stat().st_size for p in members) + sum(p.stat().st_size for p in extra.values())
    tmp = dest.with_name(dest.name + ".tmp")

    def _fill(tar: tarfile.TarFile) -> None:
        for path in members:
            tar.add(path, arcname=_arcname(path, root))
        for arcname, path in extra.items():
            tar.add(path, arcname=arcname)

    if dest.suffix == ".zst":
        with open(tmp, "wb") as out:
            proc = subprocess.Popen(
                ["zstd", "-19", "-T0", "-q", "-"], stdin=subprocess.PIPE, stdout=out, stderr=subprocess.PIPE,
            )
            assert proc.stdin is not None
            try:
                with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                    _fill(tar)
            finally:
                proc.stdin.close()
            _, err = proc.communicate(timeout=3600)
            if proc.returncode != 0:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"zstd exited {proc.returncode}: {err.decode(errors='replace')[-300:]}")
    else:
        with tarfile.open(tmp, mode="w:xz") as tar:
            _fill(tar)
    tmp.replace(dest)
    return before, dest.stat().st_size


def _arcname(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix().lstrip("/")


def verify_archive(tar_path: Path) -> set[str]:
    """List the members back from the written file -- the proof `prune_hot`
    requires before it deletes anything."""
    if tar_path.suffix == ".zst":
        proc = subprocess.run(
            ["zstd", "-dc", "-q", str(tar_path)], capture_output=True, check=True, timeout=3600,
        )
        import io

        with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r|") as tar:
            return {m.name for m in tar}
    with tarfile.open(tar_path, mode="r:xz") as tar:
        return {m.name for m in tar}


def write_manifest(
    *,
    manifest_path: Path,
    tar_path: Path,
    members: list[Path],
    root: Path,
    extra: dict[str, Path],
    bytes_before: int,
    bytes_after: int,
) -> Path:
    def _entry(path: Path, arcname: str) -> dict[str, Any]:
        return {"path": arcname, "bytes": path.stat().st_size, "sha256": sha256_file(path)}

    payload = {
        "ts": utc_now_iso(),
        "local": local_now_iso(),
        "host": socket.gethostname(),
        "git_sha": _git_sha(root),
        "tar": tar_path.name,
        "tar_sha256": sha256_file(tar_path),
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "members": [_entry(p, _arcname(p, root)) for p in members] + [_entry(p, n) for n, p in extra.items()],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path)
    return manifest_path


def _git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------ pruning


def archived_paths(archive_dir: Path) -> set[str]:
    """Every arcname any manifest under `archive_dir` has ever listed."""
    names: set[str] = set()
    for manifest in archive_dir.glob("*/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names |= {str(m.get("path")) for m in data.get("members", []) if m.get("path")}
    return names


def prune_hot(
    *, log_dir: Path, root: Path, hot_days: float, archived: set[str], today: date, dry_run: bool = False
) -> int:
    """Delete hot-tier files older than `hot_days` -- only those whose
    arcname appears in a manifest. An unarchived file is never touched."""
    if hot_days <= 0:
        return 0
    cutoff = today - timedelta(days=hot_days)
    removed = 0
    if not log_dir.is_dir():
        return 0
    for service_dir in (p for p in log_dir.iterdir() if p.is_dir()):
        candidates = (
            sorted(p for p in service_dir.rglob("*") if p.is_file())
            if service_dir.name in HOT_SUBDIRS
            else sorted(service_dir.glob("*.jsonl"))
        )
        for path in candidates:
            if _arcname(path, root) not in archived:
                continue
            if service_dir.name in HOT_SUBDIRS:
                old = datetime.fromtimestamp(path.stat().st_mtime, LOCAL_TZ).date() < cutoff
            else:
                old = path.stem < cutoff.isoformat()
            if not old:
                continue
            if not dry_run:
                path.unlink(missing_ok=True)
            removed += 1
    return removed


def prune_archives(archive_dir: Path, *, keep_days: float, today: date, dry_run: bool = False) -> int:
    if keep_days <= 0 or not archive_dir.is_dir():
        return 0
    cutoff = (today - timedelta(days=keep_days)).isoformat()
    removed = 0
    for day_dir in sorted(p for p in archive_dir.iterdir() if p.is_dir()):
        if day_dir.name < cutoff:
            if not dry_run:
                shutil.rmtree(day_dir, ignore_errors=True)
            removed += 1
    return removed


def prune_runs(*, runs_dir: Path, out_dir: Path, days: float = RUNS_SWEEP_DAYS, dry_run: bool = False) -> dict[str, SweepResult]:
    results: dict[str, SweepResult] = {}
    for label, directory in (("runs/_dryrun", runs_dir / "_dryrun"), ("runs/_stub", runs_dir / "_stub"), ("out", out_dir)):
        if directory.is_dir():
            for sub in [directory, *sorted(p for p in directory.rglob("*") if p.is_dir())]:
                key = f"{label}/{sub.relative_to(directory).as_posix()}".rstrip("/.")
                results[key] = sweep_old_files(sub, max_age_days=days, dry_run=dry_run)
    return results


def remove_empty_drama_dirs(runs_dir: Path, *, dry_run: bool = False) -> int:
    drama = runs_dir / "drama"
    if not drama.is_dir():
        return 0
    removed = 0
    for token_dir in sorted(p for p in drama.iterdir() if p.is_dir()):
        if not any(token_dir.rglob("*")) or all(p.is_dir() for p in token_dir.rglob("*")):
            if not dry_run:
                shutil.rmtree(token_dir, ignore_errors=True)
            removed += 1
    return removed


def prune_chat_turns(db: Path, *, days: float = CHAT_TURNS_KEEP_DAYS, dry_run: bool = False) -> int:
    """`chat_turns` is the rolling context window, not the audit trail (that
    is `jobs`, which is never deleted). Rows older than `days` go; then a
    VACUUM in this connection -- it fails harmlessly if a writer is busy."""
    if not db.is_file():
        return 0
    cutoff = datetime.now(LOCAL_TZ).timestamp() - days * 86400
    conn = sqlite3.connect(db, timeout=30.0)
    try:
        n = conn.execute("SELECT COUNT(*) FROM chat_turns WHERE created_at < ?", (cutoff,)).fetchone()[0]
        if not dry_run and n:
            conn.execute("DELETE FROM chat_turns WHERE created_at < ?", (cutoff,))
            conn.commit()
        if not dry_run:
            try:
                conn.execute("VACUUM")
            except sqlite3.OperationalError as exc:
                _log.warning("vacuum skipped: %s", exc)
        return int(n)
    finally:
        conn.close()


# -------------------------------------------------------------- benchmarking

BENCHMARK_DIR = "benchmark"
_BENCHMARK_MSGS = {"fetched clip", "fetched image"}
_BENCHMARK_FIELDS = ("seconds", "cost_usd", "vram_gb")
"""Numeric fields averaged per `(kind, gpu_tier)` group. `frames_per_s` is
derived separately -- it needs two fields (`frames` / `seconds`) together."""


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
                if record.get("stage") == "render" and record.get("msg") in _BENCHMARK_MSGS:
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
    for metric in _BENCHMARK_FIELDS:
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
    for metric in (*_BENCHMARK_FIELDS, "frames_per_s"):
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
    today = today or local_today()
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


# ------------------------------------------------------------------ the run


def run_archive(
    *,
    root: Path,
    log_dir: Path,
    runs_dir: Path,
    files_dir: Path,
    out_dir: Path,
    archive_dir: Path,
    hot_days: float,
    keep_days: float,
    today: date | None = None,
    dry_run: bool = False,
) -> ArchiveResult:
    today = today or local_today()
    plan = plan_archive(log_dir=log_dir, runs_dir=runs_dir, files_dir=files_dir, archive_dir=archive_dir, today=today)
    result = ArchiveResult(plan=plan, dry_run=dry_run)

    if plan.already_archived:
        result.skipped_reason = "already archived today"
    elif not plan.members and plan.sqlite_source is None:
        result.skipped_reason = "nothing to archive"
    elif dry_run:
        result.skipped_reason = "dry run"
        result.members = len(plan.members) + (1 if plan.sqlite_source else 0)
    else:
        with tempfile.TemporaryDirectory(prefix="ai-studio-archive-") as tmpdir:
            extra: dict[str, Path] = {}
            if plan.sqlite_source is not None:
                snap = Path(tmpdir) / f"queue-{plan.day}.sqlite3"
                snapshot_queue(plan.sqlite_source, snap)
                extra[f"runs/snapshots/queue-{plan.day}.sqlite3"] = snap
            before, after = write_tar(plan.members, root=root, dest=plan.tar_path, extra=extra)
            listed = verify_archive(plan.tar_path)
            expected = {_arcname(p, root) for p in plan.members} | set(extra)
            missing = expected - listed
            if missing:
                plan.tar_path.unlink(missing_ok=True)
                raise RuntimeError(f"archive verification failed; {len(missing)} member(s) missing: {sorted(missing)[:5]}")
            write_manifest(
                manifest_path=plan.manifest_path, tar_path=plan.tar_path, members=plan.members,
                root=root, extra=extra, bytes_before=before, bytes_after=after,
            )
        result.tar_path = plan.tar_path
        result.members = len(expected)
        result.bytes_before, result.bytes_after = before, after
        _log.info(
            "archive written", extra={"members": result.members, "bytes_before": before, "bytes_after": after,
                                      "reason": plan.tar_path.name},
        )

    # Read-only over the same JSONL this run just collected for the tar (or
    # would have, on a skipped/dry-run pass) -- and it must run before
    # `prune_hot` below, not after: a backlog older than `hot_days` gets
    # tarred, proven, and deleted in one pass, and a benchmark fold that ran
    # after that would find nothing left to read for those days. Riding the
    # existing daily schedule rather than a timer of its own -- see this
    # module's own docstring and docs/schedule.md.
    result.benchmark_folded = update_benchmark_report(
        log_dir=log_dir, runs_dir=runs_dir, today=today, dry_run=dry_run
    )

    archived = archived_paths(archive_dir)
    result.hot_deleted = prune_hot(log_dir=log_dir, root=root, hot_days=hot_days, archived=archived, today=today, dry_run=dry_run)
    result.archives_deleted = prune_archives(archive_dir, keep_days=keep_days, today=today, dry_run=dry_run)
    result.swept = prune_runs(runs_dir=runs_dir, out_dir=out_dir, dry_run=dry_run)
    result.drama_dirs_removed = remove_empty_drama_dirs(runs_dir, dry_run=dry_run)
    if plan.sqlite_source is not None:
        result.chat_turns_deleted = prune_chat_turns(plan.sqlite_source, dry_run=dry_run)
    _log.info(result.summary(), extra={"stage": "archive"})
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
