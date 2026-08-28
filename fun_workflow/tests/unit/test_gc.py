"""What the request side prunes and archives on top of ai-studio's archive."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from fun_workflow.pipeline.queue import JobQueue
from fun_workflow.storage.gc import archive_members, prune_chat_turns, remove_empty_drama_dirs


def test_old_chat_turns_go_and_jobs_stay(tmp_path: Path) -> None:
    db = tmp_path / "queue.sqlite3"
    stale = time.time() - 60 * 86400
    with JobQueue(db) as q:
        q.enqueue("evt-1", "Cgroup", "a cat", user_id="U1")
        q.append_chat_turn("U1", "Cgroup", "user", "old")
        conn = sqlite3.connect(db)
        conn.execute("UPDATE chat_turns SET created_at = ?", (stale,))
        conn.commit()
        conn.close()
        q.append_chat_turn("U1", "Cgroup", "user", "fresh")

    assert prune_chat_turns(db, dry_run=True) == 1
    assert prune_chat_turns(db) == 1
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    conn.close()
    assert prune_chat_turns(tmp_path / "missing.sqlite3") == 0


def test_empty_drama_dirs_are_removed_and_live_ones_kept(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "drama" / "empty-tok" / "stills").mkdir(parents=True)
    (runs / "drama" / "live-tok").mkdir(parents=True)
    (runs / "drama" / "live-tok" / "state.json").write_text("{}", encoding="utf-8")

    assert remove_empty_drama_dirs(runs, dry_run=True) == 1
    assert (runs / "drama" / "empty-tok").exists()
    assert remove_empty_drama_dirs(runs) == 1
    assert not (runs / "drama" / "empty-tok").exists()
    assert (runs / "drama" / "live-tok" / "state.json").exists()


def test_archive_members_are_drama_state_and_the_delivery_index(tmp_path: Path) -> None:
    runs, files = tmp_path / "runs", tmp_path / "files"
    (runs / "drama" / "tok").mkdir(parents=True)
    (runs / "drama" / "tok" / "state.json").write_text("{}", encoding="utf-8")
    (runs / "drama" / "tok" / "clip.mp4").write_bytes(b"x")
    files.mkdir()
    (files / "index.jsonl").write_text("", encoding="utf-8")
    (files / "a.mp4").write_bytes(b"x")

    assert archive_members(runs, files) == [runs / "drama" / "tok" / "state.json", files / "index.jsonl"]
