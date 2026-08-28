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

`delivered_at` is deliberately a timestamp beside the state rather than a sixth
state. **`done` is not `delivered`.** Rendering and telling the group about it
fail independently — the clip can exist while the push is refused for quota —
and collapsing the two would mean a worker restart re-pushes everything it
finds finished. Push is billed per recipient, so that second send costs exactly
as much as the first and tells the user nothing new. `failed` needs the same
flag for the same reason: somebody is still waiting on it.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fun_workflow.core.kinds import JobKind

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
    first_frame_path TEXT,
    quote_token TEXT,
    message_id  TEXT,
    reply_message_id TEXT,
    requested_seconds REAL,
    input_media_path TEXT,
    prompt_json TEXT,
    output_path TEXT,
    result_text TEXT,
    cost_usd    REAL,
    error       TEXT,
    gpu_tier    TEXT,
    gpu_usd_per_hr REAL,
    created_at  REAL    NOT NULL,
    parsed_at   REAL,
    started_at  REAL,
    finished_at REAL,
    delivered_at REAL,
    attempts    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_state   ON jobs(state, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS pod_opens (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    pod_id    TEXT NOT NULL,
    opened_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pod_opens_at ON pod_opens(opened_at);

CREATE TABLE IF NOT EXISTS chat_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    group_id   TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_turns_user ON chat_turns(user_id, created_at);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value REAL NOT NULL
);
"""

DAY_TZ = ZoneInfo("Asia/Taipei")
"""The day boundary the per-day caps count against.

Repeated here rather than imported from `runtime.hours`, which is where the
business calendar actually lives, because `pipeline` sits *below* `runtime` in
the layer contract and importing upwards would break it. The callers that own
the calendar pass their own boundary in; this constant only backs the
no-argument default.
"""


class JobState(str, Enum):
    QUEUED = "queued"
    PARSED = "parsed"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """DONE or FAILED: nothing will change this row's state again."""
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
    media_kind: JobKind
    first_frame_path: str | None
    quote_token: str | None
    message_id: str | None
    """LINE's id of the request message itself. A later message that *quotes*
    it arrives with this id as `quotedMessageId`, which is how「讓我看看」
    names one job rather than all of them."""
    reply_message_id: str | None
    """The id of the bot's own「收到」reply, from the Reply API's response --
    quoting that message is the other natural way to point at a request."""
    requested_seconds: float | None
    """The clip length the user asked for (`/影片15s`), or None for the
    default. Clamped to the model's range at conversion, not here."""
    """LINE's quote token for the request message, so delivery can be a reply
    to it -- the same quoted-message UI a person gets by replying -- instead
    of a broadcast into the group. Expires on LINE's side; None if absent."""
    input_media_path: str | None
    """The local photo/audio/video an understanding job describes.

    A separate column from `first_frame_path` on purpose: that field already
    means two different things for two generation kinds (H3's real first
    frame, Flux's I2I source photo) -- a third meaning for a *consumed*
    rather than *produced* path would make it actively misleading."""
    prompt_json: str | None
    output_path: str | None
    result_text: str | None
    """The produced description, for an understanding job. `output_path`
    stays None for these -- there is no file, only text."""
    cost_usd: float | None
    """What a completed job actually cost, in dollars. Only populated for a
    chat job today (`pipeline.drain.render_chat` -> `record_chat_cost()`),
    which is the only kind with a monthly sub-budget to check against -- see
    `chat_spent_this_month_usd()`. None for every other kind, whose cost is
    still tracked only at the session level (`runtime.budget.SpendLedger`)."""
    error: str | None
    gpu_tier: str | None
    gpu_usd_per_hr: float | None
    """The claimed GPU tier's hourly rental rate at claim time, in dollars.

    Persisted rather than looked up live: the `Session` that served this job
    may have long since closed by the time someone opens `/q/{token}`, so a
    live read would silently go blank. Not accumulated spend -- see
    `Job.cost_usd` for that, which stays populated only for chat jobs today.
    """
    created_at: float
    parsed_at: float | None
    started_at: float | None
    finished_at: float | None
    delivered_at: float | None
    attempts: int

    @property
    def prompt(self) -> dict[str, Any] | None:
        """The converted prompt (`prompt_json` parsed), or None while still `queued`.
        Its keys depend on the kind; `_built_by` is always there once parsed.
        """
        return json.loads(self.prompt_json) if self.prompt_json else None

    @property
    def waited_s(self) -> float:
        """Seconds from acceptance to the start of rendering (or to now)."""
        return max(0.0, (self.started_at or time.time()) - self.created_at)


def _day_start_ts(now: datetime | None = None) -> float:
    """Local midnight as a POSIX timestamp, to compare against `created_at`."""
    local = (now or datetime.now(timezone.utc)).astimezone(DAY_TZ)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _month_start_ts(now: datetime | None = None) -> float:
    """The 1st of the local month as a POSIX timestamp, for
    `chat_spent_this_month_usd()`. Same `DAY_TZ` (Asia/Taipei) as the daily
    caps, and the same boundary `runtime.budget.SpendLedger` uses for its own
    month key -- repeated here rather than imported, for the same layering
    reason `DAY_TZ` itself is: `pipeline` sits below `runtime`."""
    local = (now or datetime.now(timezone.utc)).astimezone(DAY_TZ)
    return local.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        token=row["token"],
        event_id=row["event_id"],
        group_id=row["group_id"],
        user_id=row["user_id"],
        text=row["text"],
        state=JobState(row["state"]),
        media_kind=JobKind(row["media_kind"]),
        first_frame_path=row["first_frame_path"],
        quote_token=row["quote_token"],
        message_id=row["message_id"],
        reply_message_id=row["reply_message_id"],
        requested_seconds=row["requested_seconds"],
        input_media_path=row["input_media_path"],
        prompt_json=row["prompt_json"],
        output_path=row["output_path"],
        result_text=row["result_text"],
        cost_usd=row["cost_usd"],
        error=row["error"],
        gpu_tier=row["gpu_tier"],
        gpu_usd_per_hr=row["gpu_usd_per_hr"],
        created_at=row["created_at"],
        parsed_at=row["parsed_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        delivered_at=row["delivered_at"],
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
        if "delivered_at" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN delivered_at REAL")
        if "first_frame_path" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN first_frame_path TEXT")
        if "quote_token" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN quote_token TEXT")
        if "requested_seconds" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN requested_seconds REAL")
        if "input_media_path" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN input_media_path TEXT")
        if "result_text" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN result_text TEXT")
        if "cost_usd" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN cost_usd REAL")
        if "message_id" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN message_id TEXT")
        if "reply_message_id" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN reply_message_id TEXT")
        if "gpu_usd_per_hr" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN gpu_usd_per_hr REAL")

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
        """Close this thread's connection. Safe to call twice."""
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
        media_kind: JobKind = JobKind.VIDEO,
        first_frame_path: str | None = None,
        quote_token: str | None = None,
        requested_seconds: float | None = None,
        input_media_path: str | None = None,
        message_id: str | None = None,
    ) -> tuple[Job, bool]:
        """Insert a request. Returns `(job, created)`.

        `created=False` means this `event_id` was already accepted — a LINE
        redelivery. The caller should reply with the existing job's status rather
        than making a second one. This is the load-bearing dedupe: it is enforced
        by the unique index, so it holds even across processes.

        `first_frame_path` carries a locally-saved image through to whenever
        this job actually renders -- which may be hours later, in the next
        business window -- so it has to be a path on disk, not the request's
        in-memory bytes. `input_media_path` is the same idea for an
        understanding job's photo/audio/video to describe -- a separate
        column because it means something different (consumed, not produced).
        """
        token = secrets.token_urlsafe(8)
        now = time.time()
        try:
            cur = self._conn.execute(
                "INSERT INTO jobs"
                " (token, event_id, group_id, user_id, text, state, media_kind,"
                "  first_frame_path, quote_token, requested_seconds, input_media_path,"
                "  message_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING *",
                (
                    token, event_id, group_id, user_id, text,
                    JobState.QUEUED.value, media_kind.value, first_frame_path,
                    quote_token, requested_seconds, input_media_path, message_id, now,
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

    def claim_next(
        self,
        gpu_tier: str | None = None,
        *,
        usd_per_hr: float | None = None,
        media_kind: JobKind | None = None,
    ) -> Job | None:
        """Atomically take the oldest `parsed` job and mark it `running`.

        One statement, so two drainers cannot both claim the same job and pay
        for the same clip twice. `media_kind` narrows it to one kind -- the
        worker's model-affinity claim, which keeps the checkpoint already on
        the card busy instead of swapping (see `worker.next_kind`). `usd_per_hr`
        is the claiming session's live hourly rate for `gpu_tier`, persisted
        alongside it so `/q/{token}` can show what the GPU actually rents for
        without a live lookup -- see `Job.gpu_usd_per_hr`.
        """
        if media_kind is None:
            cur = self._conn.execute(
                "UPDATE jobs SET state=?, started_at=?, attempts=attempts+1, gpu_tier=?,"
                " gpu_usd_per_hr=?"
                " WHERE id = (SELECT id FROM jobs WHERE state=? ORDER BY created_at LIMIT 1)"
                " RETURNING *",
                (JobState.RUNNING.value, time.time(), gpu_tier, usd_per_hr,
                 JobState.PARSED.value),
            )
        else:
            cur = self._conn.execute(
                "UPDATE jobs SET state=?, started_at=?, attempts=attempts+1, gpu_tier=?,"
                " gpu_usd_per_hr=?"
                " WHERE id = (SELECT id FROM jobs WHERE state=? AND media_kind=?"
                "            ORDER BY created_at LIMIT 1)"
                " RETURNING *",
                (JobState.RUNNING.value, time.time(), gpu_tier, usd_per_hr,
                 JobState.PARSED.value, media_kind.value),
            )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    def complete(self, job_id: int, output_path: str) -> Job | None:
        """Mark a file-producing job DONE with where the file landed. The
        counterpart for text results is `complete_text`.
        """
        cur = self._conn.execute(
            "UPDATE jobs SET state=?, output_path=?, finished_at=? WHERE id=? RETURNING *",
            (JobState.DONE.value, output_path, time.time(), job_id),
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    def complete_text(self, job_id: int, result_text: str) -> Job | None:
        """Finish an understanding job: text produced, no output file.

        A sibling to `complete()` rather than a second optional parameter on
        it: `complete()`'s single required `output_path` is called from
        exactly two places today, and a parameter that means "ignore the
        other one" is the kind of implicit-degrade pattern this codebase
        avoids (see `core/errors.py`).
        """
        cur = self._conn.execute(
            "UPDATE jobs SET state=?, result_text=?, finished_at=? WHERE id=? RETURNING *",
            (JobState.DONE.value, result_text, time.time(), job_id),
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    def fail(
        self, job_id: int, error: str, *, requeue: bool = False, uncounted: bool = False
    ) -> Job | None:
        """Mark a job failed, or put it back for another attempt.

        `requeue` returns it to `parsed` — used when the failure was the
        machine's rather than the request's (the window closed, the pod died),
        so the request survives to the next window instead of being lost.
        """
        state = JobState.PARSED if requeue else JobState.FAILED
        # `uncounted`: a designed stop (a drama at the lease boundary) gives
        # its attempt back, so MAX_ATTEMPTS only ever counts real failures.
        attempts_sql = ", attempts=MAX(attempts-1, 0)" if (requeue and uncounted) else ""
        cur = self._conn.execute(
            f"UPDATE jobs SET state=?, error=?, finished_at=?{attempts_sql} WHERE id=? RETURNING *",
            (state.value, error[:2000], None if requeue else time.time(), job_id),
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    # ------------------------------------------------------------------- read

    def by_id(self, job_id: int) -> Job | None:
        """Re-read a job by primary key.

        The delivery step needs this: `complete`/`fail` have just rewritten the
        row, and the caller's in-memory copy still says `running` with no
        output path.
        """
        row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def by_token(self, token: str) -> Job | None:
        """The job behind a status-page URL, or None."""
        row = self._conn.execute("SELECT * FROM jobs WHERE token=?", (token,)).fetchone()
        return _row_to_job(row) if row else None

    def by_event_id(self, event_id: str) -> Job | None:
        """The job a LINE webhook event created, or None -- the dedupe lookup."""
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
        """Every row that is not finished (`queued`, `parsed`, `running`), oldest
        first. What the reaper's `hold` and the worker's idle check look at.
        """
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state IN (?, ?, ?) ORDER BY created_at",
            (JobState.QUEUED.value, JobState.PARSED.value, JobState.RUNNING.value),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def counts(self) -> dict[str, int]:
        """Rows per state, for `/healthz` and the status page."""
        rows = self._conn.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
        return {r["state"]: int(r["n"]) for r in rows}

    def recent(self, limit: int = 20) -> list[Job]:
        """The newest rows, for an operator's glance."""
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

    def mark_delivered(self, job_id: int) -> Job | None:
        """Record that the finished media actually reached the group.

        `done` is not `delivered`. Without the distinction a worker restart
        sees a `done` job, pushes it again, and push is billed per recipient —
        so the second send costs as much as the first and tells the user
        nothing new.

        Called *after* the push, which fixes the ordering: a push that
        succeeds and a mark that then fails sends one extra message, while the
        reverse loses the delivery entirely. One duplicate beats one silence.
        """
        cur = self._conn.execute(
            "UPDATE jobs SET delivered_at=? WHERE id=? RETURNING *", (time.time(), job_id)
        )
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    def set_reply_message_id(self, job_id: int, message_id: str) -> None:
        """Remember the id of the acknowledgement LINE sent, so a later quote-reply
        can find this job (`by_quoted_message`).
        """
        self._conn.execute(
            "UPDATE jobs SET reply_message_id=? WHERE id=?", (message_id, job_id)
        )
        self._conn.commit()

    def by_quoted_message(self, group_id: str, quoted_message_id: str) -> Job | None:
        """The job a quoted message points at: either the request itself or
        the bot's「收到」reply to it. Scoped to the group so an id from
        elsewhere can never surface another group's result."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE group_id=? AND (message_id=? OR reply_message_id=?)"
            " ORDER BY created_at DESC LIMIT 1",
            (group_id, quoted_message_id, quoted_message_id),
        ).fetchone()
        return _row_to_job(row) if row else None

    # ------------------------------------------------------------ push quota

    def note_push_quota_exhausted(self, *, now: float | None = None) -> None:
        """Record that a push just failed on LINE's monthly quota.

        Written by the worker (the only process that pushes) and read by the
        webhook (the only process that talks to the user), through the one
        store both already share. The condition lasts until the calendar
        month rolls over, so a timestamp is the whole record."""
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('push_quota_exhausted_at', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (time.time() if now is None else now,),
        )
        self._conn.commit()

    def push_quota_exhausted(self, *, now: float | None = None) -> bool:
        """Has a push failed on quota *this calendar month*? LINE resets the
        free-tier quota monthly; a marker from a previous month is stale."""
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='push_quota_exhausted_at'"
        ).fetchone()
        if row is None:
            return False
        current = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc)
        marked = datetime.fromtimestamp(float(row["value"]), tz=timezone.utc)
        return (marked.year, marked.month) == (current.year, current.month)

    def undelivered(self, limit: int = 50) -> list[Job]:
        """Finished work the group has not been told about.

        Both terminal states, not just `done`: a request that failed is one
        somebody is still waiting on, and silence there reads as a broken bot.
        """
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE state IN (?, ?) AND delivered_at IS NULL"
            " ORDER BY finished_at LIMIT ?",
            (JobState.DONE.value, JobState.FAILED.value, limit),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    # -------------------------------------------------------------- chat memory

    def append_chat_turn(self, user_id: str, group_id: str, role: str, content: str) -> None:
        """Record one turn of a `/himonkey` conversation.

        Host-side, never on the ephemeral pod -- see `pipeline.drain.
        render_chat` and `core.chat_spec`'s module docstring for why. Every
        row is keyed on `user_id`, the same identity axis `accepted_today()`
        already uses for the per-user daily cap: isolation between users'
        conversations is structural (a query never crosses the key), not a
        policy someone has to remember to enforce.
        """
        self._conn.execute(
            "INSERT INTO chat_turns (user_id, group_id, role, content, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, group_id, role, content, time.time()),
        )

    def recent_chat_turns(self, user_id: str, *, limit: int = 10) -> list[tuple[str, str]]:
        """This user's last `limit` turns (role, content), oldest first.

        A rolling window, not the full history: bounds prefill cost/latency/
        billed GPU-seconds as a conversation grows, and gives natural
        "forgetting" with no explicit reset command needed for a first cut.
        """
        rows = self._conn.execute(
            "SELECT role, content FROM chat_turns WHERE user_id=?"
            " ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [(r["role"], r["content"]) for r in reversed(rows)]

    def record_chat_cost(self, job_id: int, cost_usd: float) -> None:
        """Attach what a completed /himonkey turn actually cost, for
        `chat_spent_this_month_usd()`. Called once, right after `fetch()` in
        `pipeline.drain.render_chat` -- no other job kind persists a
        per-job cost today (their spend is tracked only at the session level
        by `runtime.budget.SpendLedger`), so this column stays chat-only
        rather than something every kind must populate."""
        self._conn.execute("UPDATE jobs SET cost_usd=? WHERE id=?", (cost_usd, job_id))

    def chat_spent_this_month_usd(self, *, since: float | None = None) -> float:
        """This calendar month's total /himonkey spend, checked against
        `AI_STUDIO_MAX_CHAT_MONTH_USD` by `pipeline.drain.render_chat` before
        it ever submits a job -- the sub-budget that keeps chat's traffic
        cadence from silently consuming the money video/image also depend
        on. Same Asia/Taipei month boundary as `runtime.budget.SpendLedger`.
        """
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM jobs"
            " WHERE media_kind=? AND created_at >= ?",
            (JobKind.CHAT.value, _month_start_ts() if since is None else since),
        ).fetchone()
        return round(float(row["total"]), 6)

    def accepted_chat_today(self, user_id: str, *, since: float | None = None) -> int:
        """How many `/himonkey` messages this user has had accepted since
        local midnight -- a separate counter from `accepted_today()`'s
        all-kinds total, so a normal conversation cannot silently exhaust a
        user's entire daily video/image allowance too."""
        if not user_id:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE user_id=? AND media_kind=? AND created_at >= ?",
            (user_id, JobKind.CHAT.value, _day_start_ts() if since is None else since),
        ).fetchone()
        return int(row["n"])

    # ---------------------------------------------------------------- caps

    def accepted_kind_today(self, media_kind: JobKind, *, since: float | None = None) -> int:
        """How many requests of one kind the *whole group* has had accepted
        since local midnight. Backs `AI_STUDIO_MAX_DRAMAS_PER_DAY`: a drama is
        15-30 GPU-minutes, so the cap is on the group's day, not one user's.
        Counts every state, failures included, like `accepted_today`."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE media_kind=? AND created_at >= ?",
            (media_kind.value, _day_start_ts() if since is None else since),
        ).fetchone()
        return int(row["n"])

    def accepted_today(self, user_id: str, *, since: float | None = None) -> int:
        """How many non-chat requests this user has had accepted since local
        midnight.

        Counts every state, failures included: the cap is on asking, not on
        succeeding, or a user whose prompts keep failing validation would have
        an unlimited allowance. `None` is never counted — LINE omits
        `source.userId` for a user who has not accepted the Official Account
        terms, and lumping all of those together under one budget would let one
        anonymous request starve the next.

        Excludes `/himonkey` chat jobs — see `accepted_chat_today()`, which
        has its own separate counter. A chat conversation's cadence would
        otherwise exhaust this same allowance the video/image/understanding
        triggers share.
        """
        if not user_id:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE user_id=? AND media_kind != ? AND created_at >= ?",
            (user_id, JobKind.CHAT.value, _day_start_ts() if since is None else since),
        ).fetchone()
        return int(row["n"])

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
