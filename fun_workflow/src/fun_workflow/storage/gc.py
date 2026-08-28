"""What the request side prunes on top of ai-studio's archive: the chat
context window and finished dramas' empty run dirs -- and what it asks the
archive to keep (drama state, the delivery index)."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from ai_studio.core.observability import LOCAL_TZ

_log = logging.getLogger("fun_workflow.gc")

CHAT_TURNS_KEEP_DAYS = 30.0


def archive_members(runs_dir: Path, files_dir: Path) -> list[Path]:
    """Every drama's resumable state and render manifest, plus the delivery
    index -- the non-media records `ai_studio.storage.archive` does not know
    about."""
    members: list[Path] = []
    drama = runs_dir / "drama"
    if drama.is_dir():
        for token_dir in sorted(p for p in drama.iterdir() if p.is_dir()):
            for name in ("state.json", "render_manifest.json"):
                if (token_dir / name).is_file():
                    members.append(token_dir / name)
    index = files_dir / "index.jsonl"
    if index.is_file():
        members.append(index)
    return members


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
