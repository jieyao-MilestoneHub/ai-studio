"""The request queue.

Both properties under test here cost money when they break: a duplicate insert
generates a second clip for one request, and a double-claim pays twice for the
same one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from videogen.core.enums import MediaKind
from videogen.pipeline.queue import Job, JobQueue, JobState

PROMPT = {"integrated_multimodal_description": "[Shot 1] a cat"}


@pytest.fixture
def q(tmp_path: Path) -> JobQueue:
    with JobQueue(tmp_path / "q.sqlite3") as queue:
        yield queue


def _add(q: JobQueue, event_id: str = "e1", text: str = "生成 一隻貓") -> Job:
    job, created = q.enqueue(event_id, "Cgroup", text, user_id="Uuser")
    assert created
    return job


# ------------------------------------------------------------------- dedupe


def test_same_event_id_does_not_create_a_second_job(q: JobQueue) -> None:
    """LINE redelivers when it does not get a 2xx. This must not bill twice."""
    first = _add(q, "evt-123")
    again, created = q.enqueue("evt-123", "Cgroup", "生成 一隻貓", user_id="Uuser")

    assert created is False
    assert again.id == first.id
    assert again.token == first.token
    assert len(q.recent()) == 1


def test_dedupe_returns_the_existing_job_so_the_reply_can_show_its_status(q: JobQueue) -> None:
    first = _add(q, "evt-1")
    q.set_parsed(first.id, PROMPT)

    again, created = q.enqueue("evt-1", "Cgroup", "生成 一隻貓")
    assert created is False
    assert again.state is JobState.PARSED


def test_different_events_with_identical_text_are_both_accepted(q: JobQueue) -> None:
    """Two people asking for the same thing are two requests, not a duplicate."""
    _add(q, "evt-a")
    _add(q, "evt-b")
    assert len(q.recent()) == 2


def test_tokens_are_unguessable_and_unique(q: JobQueue) -> None:
    tokens = {_add(q, f"evt-{i}").token for i in range(20)}
    assert len(tokens) == 20
    assert all(len(t) >= 10 for t in tokens)


# -------------------------------------------------------------- state machine


def test_a_queued_job_is_not_claimable(q: JobQueue) -> None:
    """Unparsed requests must never occupy window time."""
    _add(q)
    assert q.claim_next() is None


def test_only_a_parsed_job_is_claimable(q: JobQueue) -> None:
    job = _add(q)
    q.set_parsed(job.id, PROMPT)

    claimed = q.claim_next(gpu_tier="L40S/COMMUNITY")
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.state is JobState.RUNNING
    assert claimed.gpu_tier == "L40S/COMMUNITY"
    assert claimed.prompt == PROMPT


def test_claim_is_atomic_so_the_same_job_is_never_served_twice(q: JobQueue) -> None:
    job = _add(q)
    q.set_parsed(job.id, PROMPT)

    assert q.claim_next() is not None
    assert q.claim_next() is None, "a second drainer must not get the same job"


def test_claims_come_out_oldest_first(q: JobQueue) -> None:
    ids = []
    for i in range(3):
        j = _add(q, f"evt-{i}")
        q.set_parsed(j.id, PROMPT)
        ids.append(j.id)

    assert [q.claim_next().id for _ in ids] == ids  # type: ignore[union-attr]


def test_set_parsed_is_a_no_op_on_an_already_running_job(q: JobQueue) -> None:
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()
    assert q.set_parsed(job.id, PROMPT) is None


def test_complete_records_the_output(q: JobQueue) -> None:
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()

    done = q.complete(job.id, "files/abc.mp4")
    assert done is not None
    assert done.state is JobState.DONE
    assert done.output_path == "files/abc.mp4"
    assert done.state.is_terminal


# ---------------------------------------------------------------- resilience


def test_requeue_puts_a_machine_failure_back_in_line(q: JobQueue) -> None:
    """The window closing is not the requester's fault; keep their request."""
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()

    back = q.fail(job.id, "window closed", requeue=True)
    assert back is not None and back.state is JobState.PARSED
    assert q.claim_next() is not None, "it should be claimable again next window"


def test_a_real_failure_is_terminal(q: JobQueue) -> None:
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()

    dead = q.fail(job.id, "prompt rejected by the model")
    assert dead is not None and dead.state is JobState.FAILED
    assert q.claim_next() is None


def test_release_running_rescues_jobs_orphaned_by_a_dead_pod(q: JobQueue) -> None:
    """A job left `running` has no pod behind it and would never retry."""
    for i in range(2):
        j = _add(q, f"evt-{i}")
        q.set_parsed(j.id, PROMPT)
        q.claim_next()

    assert q.release_running("pod preempted") == 2
    assert len([j for j in q.pending() if j.state is JobState.PARSED]) == 2


def test_attempts_are_counted_across_retries(q: JobQueue) -> None:
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()
    q.fail(job.id, "boom", requeue=True)
    second = q.claim_next()
    assert second is not None and second.attempts == 2


# -------------------------------------------------------------------- reading


def test_position_counts_everything_ahead_of_you(q: JobQueue) -> None:
    first = _add(q, "evt-1")
    second = _add(q, "evt-2")
    third = _add(q, "evt-3")

    assert q.position(first.token) == 1
    assert q.position(second.token) == 2
    assert q.position(third.token) == 3


def test_position_is_none_once_the_job_is_no_longer_waiting(q: JobQueue) -> None:
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()
    assert q.position(job.token) is None

    q.complete(job.id, "x.mp4")
    assert q.position(job.token) is None


def test_unparsed_is_what_the_conversion_worker_reads(q: JobQueue) -> None:
    a = _add(q, "evt-a")
    b = _add(q, "evt-b")
    q.set_parsed(a.id, PROMPT)

    assert [j.id for j in q.unparsed()] == [b.id]


def test_counts_and_lookup_by_token(q: JobQueue) -> None:
    job = _add(q)
    assert q.counts() == {"queued": 1}
    assert q.by_token(job.token) is not None
    assert q.by_token("nope") is None


# ------------------------------------------------------------------ media_kind


def test_enqueue_defaults_to_video(q: JobQueue) -> None:
    job = _add(q)
    assert job.media_kind is MediaKind.VIDEO


def test_enqueue_accepts_an_image_request(q: JobQueue) -> None:
    job, created = q.enqueue("evt-img", "Cgroup", "畫圖 一隻貓", media_kind=MediaKind.IMAGE)
    assert created
    assert job.media_kind is MediaKind.IMAGE
    assert q.by_token(job.token).media_kind is MediaKind.IMAGE


def test_a_database_created_before_media_kind_shipped_migrates_cleanly(tmp_path: Path) -> None:
    """`CREATE TABLE IF NOT EXISTS` never alters an existing table, so opening
    an old database must add the column and backfill every existing row to
    'video' — the only thing any job could have been before this feature."""
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            token       TEXT    NOT NULL UNIQUE,
            event_id    TEXT    NOT NULL UNIQUE,
            group_id    TEXT    NOT NULL,
            user_id     TEXT,
            text        TEXT    NOT NULL,
            state       TEXT    NOT NULL DEFAULT 'queued',
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
        """
    )
    conn.execute(
        "INSERT INTO jobs (token, event_id, group_id, text, state, created_at)"
        " VALUES ('tok1', 'evt-old', 'Cgroup', '一隻貓', 'queued', 1.0)"
    )
    conn.commit()
    conn.close()

    with JobQueue(path) as queue:
        columns = {row["name"] for row in queue._conn.execute("PRAGMA table_info(jobs)")}
        assert "media_kind" in columns

        job = queue.by_token("tok1")
        assert job is not None
        assert job.media_kind is MediaKind.VIDEO


def test_state_survives_reopening_the_database(tmp_path: Path) -> None:
    """A restart between the request and the window must not lose anything."""
    path = tmp_path / "q.sqlite3"
    with JobQueue(path) as first:
        job = _add(first, "evt-persist")
        first.set_parsed(job.id, PROMPT)

    with JobQueue(path) as second:
        reopened = second.by_token(job.token)
        assert reopened is not None
        assert reopened.state is JobState.PARSED
        assert reopened.prompt == PROMPT
