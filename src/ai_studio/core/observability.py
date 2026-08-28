"""Logging that can follow one request through the whole pipeline.

Two sinks, one record. Every process (`ai-studio worker`, `line serve`, the
timers) calls `configure_logging()` once at its composition root and gets:

- a human line on stderr (journald): local time with offset and milliseconds,
  level, logger, a `[job=12 token=aB3d kind=video]` context block, message;
- a JSONL file per local day under `<log_dir>/<service>/YYYY-MM-DD.jsonl`,
  one object per line, with the same context as fields -- this is what the
  daily archive compresses and what `grep '"token":"..."'` walks.

Correlation is a `contextvars.ContextVar`, not a parameter: `bind(job_id=...,
token=..., kind=...)` around `worker._run_one` is enough for every log line
inside -- drama stages, the pod LLM, the model swap, the push -- to carry the
job, because they run in the same task. Nothing changes its signature.

Why this lives in `core`: it is the only package every layer may import
(`pyproject.toml` layers contract), which is the point -- `providers`,
`storage`, `media` need it as much as `pipeline` does. The price is that it
cannot read settings (`core` sits below `config`); the roots pass `log_dir`
and `level` in, the same seam `WindowHost` and `LlmClient` use.

Logging must never take the pipeline down: the file handler swallows its
own errors (a full disk disables it until the next day; stderr still works),
and unknown `extra` keys are dropped rather than breaking a JSON line.

Before 2026-08-28 the worker configured no logging at all, so every
`_log.info` -- job durations, `_built_by`, the drama's six clips -- went to
`logging.lastResort` and was dropped (📏 `journalctl -u ai-studio-worker |
grep -c "done in"` -> 0 after a day of renders).
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Asia/Taipei")

HOT_SUBDIRS = ("sessions", "pods")
"""Subdirectories of the log dir that hold per-session/per-pod records
rather than daily JSONL: never archived by day, never scanned for render
records."""
"""The zone operators read and the timers are written in (docs/schedule.md).
JSONL carries both `ts` (UTC) and `local`; file names use the local day."""

HUMAN_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(ctx)s] %(message)s"
"""The stderr/journald line. `deploy/inference_server.py` carries a
byte-identical copy (it cannot import this package); a test pins the two."""

EXTRA_FIELDS: tuple[str, ...] = (
    "job_id", "token", "kind", "pod_id", "stage", "seconds", "cost_usd", "built_by",
    "model", "user", "message_id", "gpu_tier", "reason", "outcome", "deferred",
    "resident", "evicted", "attempts", "polls", "sha256", "load_s", "infer_s", "vram_gb",
    "minutes", "tier", "usd_per_hr", "window_end", "datacenter", "idle_min", "grace",
    "action", "spent", "cap", "opened_today", "month_spent", "quota_exhausted", "to",
    "messages", "modality", "pod_job", "members", "bytes_before", "bytes_after", "frames",
)
"""The keys a record may carry into JSONL, from `bind()` or `extra=`. An
allow-list so a stray `extra` can never produce a line that is not JSON."""

_HANDLER_TAG = "_ai_studio_observability"

_EMPTY: dict[str, Any] = {}
_CTX: ContextVar[dict[str, Any]] = ContextVar("ai_studio_log_ctx")
"""Unset means "no context"; `_ctx()` reads it. A ContextVar default must
not be a mutable literal (ruff B039), and every writer builds a new dict."""


def _ctx() -> dict[str, Any]:
    try:
        return _CTX.get()
    except LookupError:
        return _EMPTY


# ------------------------------------------------------------------ context


@contextlib.contextmanager
def bind(**ctx: Any) -> Iterator[None]:
    """Attach `ctx` to every log record emitted inside the block (this task
    and any it awaits). Nests: inner keys win, outer ones stay; restored on
    exit even when the block raises."""
    merged = {**_ctx(), **{k: v for k, v in ctx.items() if v is not None}}
    token = _CTX.set(merged)
    try:
        yield
    finally:
        _CTX.reset(token)


def current_context() -> dict[str, Any]:
    return dict(_ctx())


def utc_now_iso() -> str:
    """`2026-08-28T03:14:15.123+00:00` -- the `ts` field, and the value the
    output records (drama state, manifests, sessions) stamp themselves with."""
    return _iso(datetime.now(timezone.utc))


def local_now_iso() -> str:
    return _iso(datetime.now(LOCAL_TZ))


def local_day() -> str:
    """`YYYY-MM-DD` in `LOCAL_TZ` -- the JSONL file name and the archive day."""
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")


def _iso(when: datetime) -> str:
    return when.isoformat(timespec="milliseconds")


# --------------------------------------------------------------- formatting


class ContextFilter(logging.Filter):
    """Copies the bound context onto the record: `record.ctx` for the human
    line, and each key as an attribute for the JSONL formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _ctx()
        for key, value in ctx.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        parts = [f"{k}={getattr(record, k)}" for k in ("job_id", "token", "kind", "pod_id")
                 if getattr(record, k, None) is not None]
        record.ctx = " ".join(parts) if parts else "-"
        return True


class HumanFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(HUMAN_FORMAT)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return _iso(datetime.fromtimestamp(record.created, LOCAL_TZ))


class JsonlFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        when = datetime.fromtimestamp(record.created, timezone.utc)
        payload: dict[str, Any] = {
            "ts": _iso(when),
            "local": _iso(when.astimezone(LOCAL_TZ)),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "msg": record.getMessage(),
        }
        for key in EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            safe = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in payload.items()}
            return json.dumps(safe, ensure_ascii=False)


# ------------------------------------------------------------------ handlers


class DailyJsonlHandler(logging.Handler):
    """Appends one JSON line per record to `<log_dir>/<service>/<local day>.jsonl`.

    Reopens on day change (checked per record -- cheap, and avoids a timer).
    Append + line-buffered, so a crash loses at most the line being written
    and two processes appending to the same day interleave whole lines.
    Never raises: an `OSError` (disk full, permissions) disables this handler
    until the day changes and is reported once on stderr via `handleError`,
    which stdlib routes to `sys.stderr`, not to the caller.
    """

    def __init__(self, log_dir: Path, service: str) -> None:
        super().__init__()
        self.log_dir = Path(log_dir) / service
        self._stream: IO[str] | None = None
        self._day: str | None = None
        self._disabled_for: str | None = None

    def path_for(self, day: str) -> Path:
        return self.log_dir / f"{day}.jsonl"

    def _open(self, day: str) -> IO[str] | None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()
        self._stream, self._day = None, None
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.path_for(day), "a", encoding="utf-8", buffering=1)  # noqa: SIM115
            self._day = day
        except OSError:
            self._disabled_for = day
            self.handleError(logging.LogRecord(__name__, logging.ERROR, __file__, 0,
                                               f"cannot open {self.path_for(day)}", None, None))
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        day = local_day()
        if self._disabled_for == day:
            return
        stream = self._stream if self._day == day else self._open(day)
        if stream is None:
            return
        try:
            stream.write(self.format(record) + "\n")
        except Exception:
            self._disabled_for = day
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()
        self._stream = None
        super().close()


class _ContextFilterSingleton:
    _instance: ContextFilter | None = None

    @classmethod
    def get(cls) -> ContextFilter:
        if cls._instance is None:
            cls._instance = ContextFilter()
        return cls._instance


class _ContextLogger(logging.Logger):
    """A Logger whose records always pass through the context filter, so a
    logger created after `configure_logging` is decorated too."""

    def makeRecord(self, *args: Any, **kwargs: Any) -> logging.LogRecord:
        record = super().makeRecord(*args, **kwargs)
        _ContextFilterSingleton.get().filter(record)
        return record


def _install_context_filter(ctx_filter: ContextFilter) -> None:
    logging.setLoggerClass(_ContextLogger)
    root = logging.getLogger()
    if ctx_filter not in root.filters:
        root.addFilter(ctx_filter)
    for existing in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(existing, logging.Logger) and ctx_filter not in existing.filters:
            existing.addFilter(ctx_filter)


# -------------------------------------------------------------- entry point


def configure_logging(
    *,
    service: str,
    log_dir: Path | str | None,
    level: int | str = "INFO",
    stream: IO[str] | None = None,
) -> None:
    """Install the two sinks on the root logger. Idempotent: a second call
    replaces only the handlers this function installed (they are tagged), so
    pytest's `caplog` handler and anything else stays.

    `log_dir=None` means stderr only (tests, one-off CLI use)."""
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_TAG, False):
            root.removeHandler(existing)
            with contextlib.suppress(Exception):
                existing.close()

    # The context filter goes on the *loggers'* path, not on our handlers:
    # a filter on a handler only decorates records that handler sees, so
    # pytest's caplog (its own root handler) -- and any other sink someone
    # attaches -- would get records with no job/token on them. Python only
    # runs a logger's filters for records created on that logger, so the
    # filter is installed on the root and every "ai_studio.*" logger created
    # so far; `_ctx_logger_class` covers loggers created later.
    ctx_filter = _ContextFilterSingleton.get()
    _install_context_filter(ctx_filter)

    human = logging.StreamHandler(stream or sys.stderr)
    human.setFormatter(HumanFormatter())
    human.addFilter(ctx_filter)  # belt and braces: records from foreign loggers
    setattr(human, _HANDLER_TAG, True)
    root.addHandler(human)

    if log_dir is not None:
        jsonl = DailyJsonlHandler(Path(log_dir), service)
        jsonl.setFormatter(JsonlFormatter(service))
        jsonl.addFilter(ctx_filter)
        setattr(jsonl, _HANDLER_TAG, True)
        root.addHandler(jsonl)

    root.setLevel(logging.getLevelName(level) if isinstance(level, str) else level)
    # Chatter that adds nothing to a trace: every callback is already one
    # line under ai_studio.webhook, and every poll is one under its provider.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
