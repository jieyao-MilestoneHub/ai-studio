"""The request queue.

SQLite rather than an in-process structure, because requests arrive around the
clock while the GPU only exists for two hours a day. A restart in between must
not lose anyone's request.

Two invariants are enforced by the database rather than by application code,
because both cost real money when they fail:

1. **`event_id` is UNIQUE.** LINE redelivers a webhook when it does not get a
   2xx, and documents that the same event may arrive more than once. If dedupe
   lived in Python, a redelivery during a deploy would generate — and bill for —
   the same clip twice. The unique index makes that impossible instead of
   unlikely.

2. **Claiming a job is a single atomic statement.** Two drainers (a stray
   process, a scheduler misfire) must not both pick up the same job and pay for
   it twice.

State machine:

    queued ──parse──▶ parsed ──claim──▶ running ──▶ done
       │                 │                 │
       └────────── failed ◀────────────────┘

`queued` means the webhook accepted it; `parsed` means the request has been
converted into a validated prompt (an LLM-built H3 prompt for video, a
lightly-validated Flux prompt for images) and is ready for a GPU. Only `parsed`
jobs are claimable, so a request whose prompt could not be built never occupies
window time.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ai_studio.core.enums import MediaKind

DEFAULT_DB = Path("runs/queue.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    token       TEXT    NOT NULL UNIQUE,
    event_id    TEXT    NOT NULL UNIQUE,
    group_id    TEXT    NOT NULL,
    user_id     TEXT,
    text        TEXT    NOT NULL,
    state       TEXT    NOT NULL DEFAULT 'queued',
    media_kind  TEXT    NOT NULL DEFAULT 'video',
    prompt_json TEXT,
    output_path TEXT,
    error       TEXT,
    gpu_tier    TEXT,
    created_at  REAL    NOT NULL,
    parsed_at   REAL,
    started_at  REAL,
    finished_at REAL,
    attempts    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_state   ON jobs(state, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""


class JobState(str, Enum):
    QUEUED = "queued"
    PARSED = "parsed"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.DONE, JobState.FAILED)


@dataclass(frozen=True)
class Job:
    id: int
    token: str
    event_id: str
    group_id: str
    user_id: str | None
    text: str
    state: JobState
    media_kind: MediaKind
    prompt_json: str | None
    output_path: str | None
    error: str | None
    gpu_tier: str | None
    created_at: float
    parsed_at: float | None
    started_at: float | None
    finished_at: float | None
    attempts: int

    @property
    def prompt(self) -> dict[str, Any] | None:
        return json.loads(self.prompt_json) if self.prompt_json else None

    @property
    def waited_s(self) -> float:
        return max(0.0, (self.started_at or time.time()) - self.created_at)


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        token=row["token"],
        event_id=row["event_id"],
        group_id=row["group_id"],
        user_id=row["user_id"],
        text=row["text"],
        state=JobState(row["state"]),
        media_kind=MediaKind(row["media_kind"]),
        prompt_json=row["prompt_json"],
        output_path=row["output_path"],
        error=row["error"],
        gpu_tier=row["gpu_tier"],
        created_at=row["created_at"],
        parsed_at=row["parsed_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        attempts=row["attempts"],
    )


class JobQueue:
    """A durable, dedupe-on-insert request queue."""

    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        conn = self._connect()
        conn.executescript(_SCHEMA)
        self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns to a database created before this feature shipped.

        `CREATE TABLE IF NOT EXISTS` never alters an existing table, and SQLite
        has no `ADD COLUMN IF NOT EXISTS`, so the existence check is required —
        an unconditional `ALTER TABLE` on a column that already exists raises.
        """
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "media_kind" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN media_kind TEXT NOT NULL DEFAULT 'video'")

    def _connect(self) -> sqlite3.Connection:
        """One connection per thread, all pointing at the same file.

        A sqlite3 connection may only be used from the thread that created it,
        and Starlette runs sync endpoints and background tasks on a threadpool —
        so a single shared connection raises `ProgrammingError` under uvicorn,
        not just in tests. Per-thread connections avoid that without serialising
        every call behind a mutex: WAL lets readers and one writer proceed
        concurrently, and `claim_next` is atomic because it is a single
        UPDATE ... RETURNING statement, whichever connection issues it.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        conn = sqlite3.connect(self.path, isolation_level=None, timeout=15.0)
        conn.row_factory = sqlite3.Row
        # WAL so the status page reading never blocks the webhook writing —
        # which matters when the webhook has a two-second budget.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._local.conn = conn
        with self._lock:
            self._connections.append(conn)
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._connect()

    def close(self) -> None:
        with self._lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            with contextlib.suppress(sqlite3.Error):  # already closed
                conn.close()
        self._local = threading.local()

    def __enter__(self) -> JobQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ write

    def enqueue(
        self,
        event_id: str,
        group_id: str,
        text: str,
        user_id: str | None = None,
        *,
        media_kind: MediaKind = MediaKind.VIDEO,
    ) -> tuple[Job, bool]:
        """Insert a request. Returns `(job, created)`.

        `created=False` means this `event_id` was already accepted — a LINE
        redelivery. The caller should reply with the existing job's status rather
        than making a second one. This is the load-bearing dedupe: it is enforced
        by the unique index, so it holds even across processes.
        """
        token = secrets.token_urlsafe(8)
        now = time.time()
        try:
            cur = self._conn.execute(
                "INSERT INTO jobs"
                " (token, event_id, group_id, user_id, text, state, media_kind, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING *",
                (
                    token, event_id, group_id, user_id, text,
                    JobState.QUEUED.value, media_kind.value, now,
                ),
            )
            row = cur.fetchone()
            return _row_to_job(row), True
        except sqlite3.IntegrityError:
            existing = self.by_event_id(event_id)
            if existing is None:  # pragma: no cover - only a token collision
                raise
            return existing, False

    def set_parsed(self, job_id: int, prompt: dict[str, Any]) -> Job | None:
        """Attach a validated prompt (H3 or Flux) and make the job claimable."""
        cur = self._conn.execute(
            "UPDATE jobs SET state=?, prompt_json=?, parsed_at=?"
            " WHERE id=? AND state=? RETURNING *",
            (JobState.PARSED.value, json.dumps(prompt, ensure_ascii=False),
             time.time(), job_id, JobState.QUEUED.value),
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    def claim_next(self, gpu_tier: str | None = None) -> Job | None:
        """Atomically take the oldest `parsed` job and mark it `running`.

        One statement, so two drainers cannot both claim the same job and pay
        for the same clip twice.
        """
        cur = self._conn.execute(
            "UPDATE jobs SET state=?, started_at=?, attempts=attempts+1, gpu_tier=?"
            " WHERE id = (SELECT id FROM jobs WHERE state=? ORDER BY created_at LIMIT 1)"
            " RETURNING *",
            (JobState.RUNNING.value, time.time(), gpu_tier, JobState.PARSED.value),
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    def complete(self, job_id: int, output_path: str) -> Job | None:
        cur = self._conn.execute(
            "UPDATE jobs SET state=?, output_path=?, finished_at=? WHERE id=? RETURNING *",
            (JobState.DONE.value, output_path, time.time(), job_id),
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    def fail(self, job_id: int, error: str, *, requeue: bool = False) -> Job | None:
        """Mark a job failed, or put it back for another attempt.

        `requeue` returns it to `parsed` — used when the failure was the
        machine's rather than the request's (the window closed, the pod died),
        so the request survives to the next window instead of being lost.
        """
        state = JobState.PARSED if requeue else JobState.FAILED
        cur = self._conn.execute(
            "UPDATE jobs SET state=?, error=?, finished_at=? WHERE id=? RETURNING *",
            (state.value, error[:2000], None if requeue else time.time(), job_id),
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    # ------------------------------------------------------------------- read

    def by_token(self, token: str) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE token=?", (token,)).fetchone()
        return _row_to_job(row) if row else None

    def by_event_id(self, event_id: str) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE event_id=?", (event_id,)).fetchone()
        return _row_to_job(row) if row else None

    def position(self, token: str) -> int | None:
        """1-based place in the pending line, or None if it is not waiting.

        Counts everything not yet finished and older than this job, so the
        number a user is told matches what actually has to happen before theirs.
        """
        job = self.by_token(token)
        if job is None or job.state.is_terminal or job.state is JobState.RUNNING:
            return None
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE state IN (?, ?, ?) AND created_at < ?",
            (JobState.QUEUED.value, JobState.PARSED.value, JobState.RUNNING.value,
             job.created_at),
        ).fetchone()
        return int(row["n"]) + 1

    def pending(self) -> list[Job]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state IN (?, ?, ?) ORDER BY created_at",
            (JobState.QUEUED.value, JobState.PARSED.value, JobState.RUNNING.value),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
        return {r["state"]: int(r["n"]) for r in rows}

    def recent(self, limit: int = 20) -> list[Job]:
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def unparsed(self, limit: int = 50) -> list[Job]:
        """Jobs not yet converted to a prompt — the background worker's input."""
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state=? ORDER BY created_at LIMIT ?",
            (JobState.QUEUED.value, limit),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def release_running(self, reason: str) -> int:
        """Return every `running` job to `parsed`. Call at window open.

        A job left `running` is one whose pod vanished mid-render — the window
        ended, the machine was preempted, the drainer crashed. Without this they
        would sit `running` forever and never be retried.
        """
        cur = self._conn.execute(
            "UPDATE jobs SET state=?, error=? WHERE state=?",
            (JobState.PARSED.value, f"requeued: {reason}", JobState.RUNNING.value),
        )
        return cur.rowcount
