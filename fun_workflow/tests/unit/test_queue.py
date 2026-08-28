"""The request queue.

Both properties under test here cost money when they break: a duplicate insert
generates a second clip for one request, and a double-claim pays twice for the
same one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from ai_studio.core.enums import MediaKind

from fun_workflow.pipeline.queue import Job, JobQueue, JobState

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


def test_claim_persists_the_claiming_sessions_hourly_rate(q: JobQueue) -> None:
    """`/q/{token}` shows this later, possibly long after the session that
    served the job has closed -- so it has to be stored at claim time, not
    looked up live. See `Job.gpu_usd_per_hr`."""
    job = _add(q)
    q.set_parsed(job.id, PROMPT)

    claimed = q.claim_next(gpu_tier="L40S/COMMUNITY", usd_per_hr=1.004)
    assert claimed is not None
    assert claimed.gpu_usd_per_hr == 1.004
    assert q.by_token(job.token).gpu_usd_per_hr == 1.004


def test_claim_without_a_rate_leaves_it_unset(q: JobQueue) -> None:
    """The manual `session drain` path may not always have a rate to pass;
    the page must simply omit the row rather than show a stale one."""
    job = _add(q)
    q.set_parsed(job.id, PROMPT)

    claimed = q.claim_next(gpu_tier="L40S/COMMUNITY")
    assert claimed is not None
    assert claimed.gpu_usd_per_hr is None


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


# ------------------------------------------------------------- delivery


def test_done_is_not_delivered(q: JobQueue) -> None:
    """The distinction push is billed for. Without it a worker restart sees a
    `done` job and pushes it a second time, at full price, saying nothing new."""
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()
    q.complete(job.id, "files/abc.mp4")

    assert q.by_id(job.id).delivered_at is None
    assert [j.id for j in q.undelivered()] == [job.id]

    q.mark_delivered(job.id)

    assert q.by_id(job.id).delivered_at is not None
    assert q.undelivered() == []


def test_a_failed_job_is_undelivered_too(q: JobQueue) -> None:
    """Somebody is still waiting on it, and silence reads as a broken bot."""
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()
    q.fail(job.id, "the pod died")

    assert [j.id for j in q.undelivered()] == [job.id]


def test_unfinished_work_is_not_undelivered(q: JobQueue) -> None:
    """`undelivered` means "finished and unannounced", not "not finished"."""
    job = _add(q)
    q.set_parsed(job.id, PROMPT)

    assert q.undelivered() == []
    q.claim_next()
    assert q.undelivered() == []


def test_a_requeued_job_is_not_treated_as_finished(q: JobQueue) -> None:
    """A transient failure goes back to `parsed`. Announcing it would bill a
    push for something the user does not need to know about."""
    job = _add(q)
    q.set_parsed(job.id, PROMPT)
    q.claim_next()
    q.fail(job.id, "proxy hiccup", requeue=True)

    assert q.by_id(job.id).state is JobState.PARSED
    assert q.undelivered() == []


def test_a_database_from_before_delivery_shipped_still_opens(tmp_path: Path) -> None:
    """`CREATE TABLE IF NOT EXISTS` never alters an existing table, so an old
    file has to be migrated rather than recreated — the alternative is losing
    every queued request on deploy."""
    path = tmp_path / "old.sqlite3"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT NOT NULL UNIQUE,
            event_id TEXT NOT NULL UNIQUE,
            group_id TEXT NOT NULL,
            user_id TEXT,
            text TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued',
            prompt_json TEXT,
            output_path TEXT,
            error TEXT,
            gpu_tier TEXT,
            created_at REAL NOT NULL,
            parsed_at REAL,
            started_at REAL,
            finished_at REAL,
            attempts INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO jobs (token, event_id, group_id, text, created_at)
        VALUES ('tok-old', 'evt-old', 'Cgroup', '生成 一隻貓', 1700000000.0);
        """
    )
    old.commit()
    old.close()

    with JobQueue(path) as queue:
        survivor = queue.by_token("tok-old")
        assert survivor is not None, "the pre-existing request was lost"
        assert survivor.delivered_at is None
        assert survivor.media_kind is MediaKind.VIDEO, "the older migration still applies"
        queue.mark_delivered(survivor.id)
        assert queue.by_token("tok-old").delivered_at is not None


# ------------------------------------------------------------- daily counters


def test_accepted_today_counts_one_users_requests(q: JobQueue) -> None:
    q.enqueue("e-a", "Cgroup", "one", user_id="Ualice")
    q.enqueue("e-b", "Cgroup", "two", user_id="Ualice")
    q.enqueue("e-c", "Cgroup", "three", user_id="Ubob")

    assert q.accepted_today("Ualice") == 2
    assert q.accepted_today("Ubob") == 1
    assert q.accepted_today("Unobody") == 0


def test_accepted_today_ignores_an_unidentifiable_user(q: JobQueue) -> None:
    """LINE omits `source.userId` for a user who has not accepted the terms.
    Lumping all of those under one budget would let one anonymous request
    starve the next."""
    q.enqueue("e-x", "Cgroup", "one", user_id=None)

    assert q.accepted_today("") == 0
    assert q.accepted_today("Ualice") == 0


def test_accepted_today_stops_at_the_day_boundary(q: JobQueue) -> None:
    import time as _time

    q.enqueue("e-old", "Cgroup", "yesterday", user_id="Ualice")
    tomorrow = _time.time() + 86_400

    assert q.accepted_today("Ualice", since=tomorrow) == 0
    assert q.accepted_today("Ualice", since=0.0) == 1


def _far_future() -> float:
    import time as _time

    return _time.time() + 86_400


# ------------------------------------------------------------- understanding


def test_input_media_path_round_trips_through_the_queue(q: JobQueue) -> None:
    job, created = q.enqueue(
        "evt-u1", "Cgroup", "", media_kind=MediaKind.IMAGE_UNDERSTAND,
        input_media_path="/incoming/photo.jpg",
    )
    assert created
    assert job.input_media_path == "/incoming/photo.jpg"
    assert job.first_frame_path is None
    assert q.by_id(job.id).input_media_path == "/incoming/photo.jpg"


def test_input_media_path_defaults_to_none(q: JobQueue) -> None:
    job = _add(q)
    assert job.input_media_path is None


def test_complete_text_finishes_the_job_with_no_output_path(q: JobQueue) -> None:
    job, _ = q.enqueue(
        "evt-u2", "Cgroup", "", media_kind=MediaKind.AUDIO_UNDERSTAND,
        input_media_path="/incoming/clip.m4a",
    )
    q.set_parsed(job.id, {"_built_by": "understanding"})
    q.claim_next()

    done = q.complete_text(job.id, "有人在說話,背景有鳥叫聲。")
    assert done is not None
    assert done.state is JobState.DONE
    assert done.result_text == "有人在說話,背景有鳥叫聲。"
    assert done.output_path is None


def test_a_database_created_before_understanding_shipped_migrates_cleanly(
    tmp_path: Path,
) -> None:
    """Same discipline as the media_kind migration test: an old database must
    gain `input_media_path`/`result_text` rather than fail to open."""
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
            media_kind  TEXT    NOT NULL DEFAULT 'video',
            first_frame_path TEXT,
            quote_token TEXT,
            requested_seconds REAL,
            prompt_json TEXT,
            output_path TEXT,
            error       TEXT,
            gpu_tier    TEXT,
            created_at  REAL    NOT NULL,
            parsed_at   REAL,
            started_at  REAL,
            finished_at REAL,
            delivered_at REAL,
            attempts    INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        "INSERT INTO jobs (token, event_id, group_id, text, created_at) "
        "VALUES ('tok', 'evt', 'Cgroup', 'hi', 0)"
    )
    conn.commit()
    conn.close()

    with JobQueue(path) as q:
        columns = {row[1] for row in q._conn.execute("PRAGMA table_info(jobs)")}
        assert "input_media_path" in columns
        assert "result_text" in columns

        job = q.by_token("tok")
        assert job is not None
        assert job.input_media_path is None
        assert job.result_text is None

        job, created = q.enqueue(
            "evt-new", "Cgroup", "", media_kind=MediaKind.VIDEO_UNDERSTAND,
            input_media_path="/incoming/clip.mp4",
        )
        assert created
        assert job.input_media_path == "/incoming/clip.mp4"


# --------------------------------------------------------------- chat memory


def test_chat_history_is_isolated_between_users(q: JobQueue) -> None:
    """The one test that most directly proves F's "never leaks across users"
    requirement: one user's turns must never surface in another's history."""
    q.append_chat_turn("Ualice", "Cgroup", "user", "alice's message")
    q.append_chat_turn("Ualice", "Cgroup", "assistant", "reply to alice")
    q.append_chat_turn("Ubob", "Cgroup", "user", "bob's message")

    assert q.recent_chat_turns("Ualice") == [
        ("user", "alice's message"),
        ("assistant", "reply to alice"),
    ]
    assert q.recent_chat_turns("Ubob") == [("user", "bob's message")]
    assert q.recent_chat_turns("Unobody") == []


def test_recent_chat_turns_is_a_rolling_window_oldest_first(q: JobQueue) -> None:
    for i in range(15):
        q.append_chat_turn("Ualice", "Cgroup", "user", f"turn {i}")

    turns = q.recent_chat_turns("Ualice", limit=10)
    assert len(turns) == 10
    assert [content for _, content in turns] == [f"turn {i}" for i in range(5, 15)]


def test_accepted_chat_today_counts_only_chat_jobs(q: JobQueue) -> None:
    q.enqueue("e-chat-1", "Cgroup", "hi", user_id="Ualice", media_kind=MediaKind.CHAT)
    q.enqueue("e-chat-2", "Cgroup", "hi again", user_id="Ualice", media_kind=MediaKind.CHAT)
    q.enqueue("e-video", "Cgroup", "生成 一隻貓", user_id="Ualice", media_kind=MediaKind.VIDEO)

    assert q.accepted_chat_today("Ualice") == 2
    assert q.accepted_chat_today("Ubob") == 0


def test_accepted_today_excludes_chat_so_it_cannot_exhaust_the_shared_cap(q: JobQueue) -> None:
    """A chat conversation's cadence must not eat a user's video/image
    allowance -- see `accepted_chat_today()`'s separate counter."""
    for i in range(5):
        q.enqueue(f"e-chat-{i}", "Cgroup", f"hi {i}", user_id="Ualice", media_kind=MediaKind.CHAT)
    q.enqueue("e-video", "Cgroup", "生成 一隻貓", user_id="Ualice", media_kind=MediaKind.VIDEO)

    assert q.accepted_today("Ualice") == 1
    assert q.accepted_chat_today("Ualice") == 5


def test_chat_spend_accumulates_within_the_month(q: JobQueue) -> None:
    a, _ = q.enqueue("e-chat-a", "Cgroup", "hi", media_kind=MediaKind.CHAT)
    b, _ = q.enqueue("e-chat-b", "Cgroup", "hi again", media_kind=MediaKind.CHAT)
    q.enqueue("e-video", "Cgroup", "生成 一隻貓", media_kind=MediaKind.VIDEO)

    assert q.chat_spent_this_month_usd() == 0.0

    q.record_chat_cost(a.id, 0.02)
    q.record_chat_cost(b.id, 0.03)

    assert q.chat_spent_this_month_usd() == 0.05
    assert q.by_id(a.id).cost_usd == 0.02


def test_chat_spend_ignores_non_chat_jobs_even_with_a_cost_recorded(q: JobQueue) -> None:
    """Only chat populates `cost_usd` today, but the query itself must stay
    kind-scoped rather than trusting that invariant implicitly."""
    video, _ = q.enqueue("e-video", "Cgroup", "生成 一隻貓", media_kind=MediaKind.VIDEO)
    q.record_chat_cost(video.id, 99.0)

    assert q.chat_spent_this_month_usd() == 0.0


def test_a_database_created_before_chat_shipped_migrates_cleanly(tmp_path: Path) -> None:
    """Same discipline as the media_kind/understanding migration tests: an
    old database must gain `cost_usd` and the `chat_turns` table rather than
    fail to open."""
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
            media_kind  TEXT    NOT NULL DEFAULT 'video',
            first_frame_path TEXT,
            quote_token TEXT,
            requested_seconds REAL,
            input_media_path TEXT,
            prompt_json TEXT,
            output_path TEXT,
            result_text TEXT,
            error       TEXT,
            gpu_tier    TEXT,
            created_at  REAL    NOT NULL,
            parsed_at   REAL,
            started_at  REAL,
            finished_at REAL,
            delivered_at REAL,
            attempts    INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        "INSERT INTO jobs (token, event_id, group_id, text, created_at) "
        "VALUES ('tok', 'evt', 'Cgroup', 'hi', 0)"
    )
    conn.commit()
    conn.close()

    with JobQueue(path) as q:
        columns = {row[1] for row in q._conn.execute("PRAGMA table_info(jobs)")}
        assert "cost_usd" in columns

        job = q.by_token("tok")
        assert job is not None
        assert job.cost_usd is None

        q.append_chat_turn("Ualice", "Cgroup", "user", "hi")
        assert q.recent_chat_turns("Ualice") == [("user", "hi")]


def test_an_uncounted_requeue_hands_the_attempt_back_and_floors_at_zero(q: JobQueue) -> None:
    """`fail(requeue=True, uncounted=True)` is for a designed stop (a drama at
    the lease boundary): the row goes back to parsed and the claim's
    attempts+1 is undone, never below zero."""
    job, _ = q.enqueue("evt-uncounted", "Cgroup", "a story", user_id="U1")
    q.set_parsed(job.id, {"_rendered": "t"})
    claimed = q.claim_next()
    assert claimed is not None and claimed.attempts == 1

    back = q.fail(job.id, "resume: lease boundary", requeue=True, uncounted=True)
    assert back is not None and back.state is JobState.PARSED and back.attempts == 0

    again = q.fail(job.id, "resume again", requeue=True, uncounted=True)
    assert again is not None and again.attempts == 0, "floors at zero"

    counted = q.fail(job.id, "provider: real failure", requeue=True)
    assert counted is not None and counted.attempts == 0, "a counted requeue does not touch attempts"
