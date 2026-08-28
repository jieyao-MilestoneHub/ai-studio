"""LINE webhook: signature, filtering, dedupe, and the two-second path."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from ai_studio.core.enums import MediaKind
from ai_studio.runtime import hours

from fun_workflow.bots.line.reply import NullReplyClient
from fun_workflow.bots.line.verify import sign, verify
from fun_workflow.bots.line.webhook import InvalidSignature, WebhookHandler
from fun_workflow.pipeline.queue import JobQueue, JobState

SECRET = "test-channel-secret"
GROUP = "Cae56f94637c1234567890abcdef12345"
OTHER_GROUP = "Cffffffffffffffffffffffffffffffff"


@pytest.fixture
def wired(tmp_path: Path):
    queue = JobQueue(tmp_path / "q.sqlite3")
    replier = NullReplyClient()
    handler = WebhookHandler(
        queue,
        replier,
        channel_secret=SECRET,
        allowed_group_id=GROUP,
        base_url="https://vg.example.com/",
        # Fixed inside business hours: the accepted-message wording branches
        # on the clock, and this fixture's tests care about the in-hours
        # text, not about whichever branch the real wall clock happens to
        # land on whenever the suite runs.
        clock=lambda: OPEN_INSTANT,
    )
    yield handler, queue, replier
    queue.close()


def _body(events: list[dict] | None) -> bytes:
    return json.dumps(
        {"destination": "U" + "0" * 32, "events": events if events is not None else []}
    ).encode("utf-8")


def _text_event(text: str, *, group: str = GROUP, event_id: str = "evt-1", **kw) -> dict:
    event = {
        "type": "message",
        "mode": "active",
        "webhookEventId": event_id,
        "timestamp": 1700000000000,
        "replyToken": "rt-" + event_id,
        "source": {"type": "group", "groupId": group, "userId": "U" + "1" * 32},
        "message": {"type": "text", "id": "m1", "text": text},
    }
    event.update(kw)
    return event


# ------------------------------------------------------------------ signature


def test_signature_roundtrip_matches_the_documented_algorithm() -> None:
    """HMAC-SHA256 over the raw bytes, base64-encoded."""
    body = b'{"events":[]}'
    assert verify(body, sign(body, SECRET), SECRET)


def test_a_reserialised_body_fails_verification() -> None:
    """Why the handler must never parse before verifying.

    Re-encoding JSON reorders nothing here but changes whitespace, and that is
    enough to invalidate the signature.
    """
    original = b'{"destination":"U1", "events": []}'
    signature = sign(original, SECRET)
    reserialised = json.dumps(json.loads(original)).encode("utf-8")

    assert reserialised != original
    assert not verify(reserialised, signature, SECRET)


def test_wrong_secret_missing_header_and_tampered_body_all_fail() -> None:
    body = _body([])
    assert not verify(body, sign(body, "other-secret"), SECRET)
    assert not verify(body, None, SECRET)
    assert not verify(body + b" ", sign(body, SECRET), SECRET)


@pytest.mark.asyncio
async def test_bad_signature_raises_so_the_route_can_return_400(wired) -> None:
    handler, _, _ = wired
    with pytest.raises(InvalidSignature):
        await handler.handle(_body([_text_event("/影片 一隻貓")]), "not-a-signature")


# ------------------------------------------------------- LINE's own probes


@pytest.mark.asyncio
async def test_an_empty_events_array_is_accepted(wired) -> None:
    """LINE's Verify button and its connectivity checks send events: []."""
    handler, _, _ = wired
    body = _body([])
    assert await handler.handle(body, sign(body, SECRET)) == []


# --------------------------------------------------------------- filtering


@pytest.mark.asyncio
async def test_only_the_allowlisted_group_is_served(wired) -> None:
    handler, queue, _ = wired
    body = _body([_text_event("/影片 一隻貓", group=OTHER_GROUP)])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "wrong_group"
    assert queue.counts() == {}


@pytest.mark.asyncio
async def test_ordinary_chatter_is_ignored_with_no_reply_at_all(wired) -> None:
    """The bot must be invisible during normal group conversation."""
    handler, queue, replier = wired
    body = _body([_text_event("今天午餐吃什麼")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "ignored"
    assert replier.sent == []
    assert queue.counts() == {}


@pytest.mark.asyncio
async def test_standby_mode_produces_no_send(wired) -> None:
    """LINE documents that a standby-mode bot must not send anything."""
    handler, _, replier = wired
    body = _body([_text_event("/影片 一隻貓", mode="standby")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "standby"
    assert replier.sent == []


@pytest.mark.asyncio
async def test_non_text_messages_are_ignored(wired) -> None:
    handler, _, _ = wired
    event = _text_event("x")
    event["message"] = {"type": "sticker", "id": "s1"}
    body = _body([event])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "ignored"


# ---------------------------------------------------------------- accepting


@pytest.mark.asyncio
async def test_a_trigger_message_is_queued_and_acknowledged_with_a_link(wired) -> None:
    handler, _queue, replier = wired
    body = _body([_text_event("/影片 一隻橘貓走在雨中")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))

    assert outcome.action == "accepted"
    assert outcome.job is not None
    assert outcome.job.text == "一隻橘貓走在雨中", "the trigger word is stripped"
    assert outcome.job.state is JobState.QUEUED

    (_, texts) = replier.sent[0]
    assert "想查進度可以看" in texts[0]
    assert f"https://vg.example.com/q/{outcome.job.token}" in texts[0]


@pytest.mark.asyncio
async def test_the_old_aliases_no_longer_trigger(wired) -> None:
    """One spelling per trigger. 生成/畫圖 bare, their slash forms, and the
    English aliases were all retired together: a bare word that is also
    ordinary Chinese is a request nobody meant, paid for in GPU-minutes."""
    handler, queue, replier = wired
    for i, text in enumerate(
        ("生成 一隻貓", "/生成 一隻貓", "/gen a cat", "畫圖 一隻貓", "/畫圖 一隻貓", "/img a cat")
    ):
        body = _body([_text_event(text, event_id=f"evt-old-{i}")])
        (outcome,) = await handler.handle(body, sign(body, SECRET))
        assert outcome.action == "ignored", text
    assert queue.counts() == {}
    assert not replier.sent, "an unmatched message gets no reply at all"


@pytest.mark.asyncio
async def test_the_image_trigger_enqueues_an_image_job(wired) -> None:
    handler, _queue, replier = wired
    body = _body([_text_event("/圖片 一隻橘貓")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))

    assert outcome.action == "accepted"
    assert outcome.job is not None
    assert outcome.job.media_kind is MediaKind.IMAGE
    assert outcome.job.text == "一隻橘貓"
    assert "想查進度可以看" in replier.sent[0][1][0]


@pytest.mark.asyncio
async def test_the_video_trigger_still_defaults_to_video(wired) -> None:
    """Regression: adding the image trigger must not change the existing path."""
    handler, _queue, _replier = wired
    body = _body([_text_event("/影片 一隻貓")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.job is not None
    assert outcome.job.media_kind is MediaKind.VIDEO


@pytest.mark.asyncio
async def test_the_three_triggers_do_not_overlap(wired) -> None:
    """/圖片 and /圖影 share a first character; each must map to its own
    kind and neither may shadow the other."""
    handler, _, _ = wired
    body = _body([_text_event("/圖片 一隻貓", event_id="evt-kind-img")])
    (img,) = await handler.handle(body, sign(body, SECRET))
    assert img.action == "accepted" and img.job.media_kind is MediaKind.IMAGE
    body = _body([_text_event("/影片 一隻貓", event_id="evt-kind-vid")])
    (vid,) = await handler.handle(body, sign(body, SECRET))
    assert vid.action == "accepted" and vid.job.media_kind is MediaKind.VIDEO
    assert vid.job.first_frame_path is None


@pytest.mark.asyncio
async def test_the_chat_trigger_enqueues_a_chat_job_and_claims_no_media(wired) -> None:
    handler, _queue, replier = wired
    body = _body([_text_event("/himonkey 你好嗎")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))

    assert outcome.action == "accepted"
    assert outcome.job is not None
    assert outcome.job.media_kind is MediaKind.CHAT
    assert outcome.job.text == "你好嗎"
    assert outcome.job.first_frame_path is None
    assert "想查進度可以看" in replier.sent[0][1][0]


@pytest.mark.asyncio
async def test_a_bare_chat_trigger_gets_usage_help_and_is_not_queued(wired) -> None:
    """Unlike the three describe triggers, /himonkey requires trailing
    text -- there is no attached media to fall back on."""
    handler, queue, replier = wired
    body = _body([_text_event("/himonkey")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "ignored"
    assert queue.counts() == {}
    assert "用法" in replier.sent[0][1][0]
    assert "/himonkey" in replier.sent[0][1][0]


@pytest.mark.asyncio
async def test_the_chat_trigger_does_not_shadow_or_get_shadowed_by_the_others(wired) -> None:
    """/himonkey shares no prefix with any of the seven other triggers, but
    the dispatch tuple is order-sensitive -- pin that adding it did not
    change any existing trigger's outcome."""
    handler, _, _ = wired
    for text, expected_kind in (
        ("/影片 貓", MediaKind.VIDEO),
        ("/圖片 貓", MediaKind.IMAGE),
        ("/himonkey 嗨", MediaKind.CHAT),
    ):
        body = _body([_text_event(text, event_id=f"evt-no-shadow-{expected_kind.value}")])
        (outcome,) = await handler.handle(body, sign(body, SECRET))
        assert outcome.action == "accepted"
        assert outcome.job.media_kind is expected_kind


@pytest.mark.asyncio
async def test_a_bare_image_trigger_gets_usage_help_and_is_not_queued(wired) -> None:
    handler, queue, replier = wired
    body = _body([_text_event("/圖片")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "ignored"
    assert queue.counts() == {}
    assert "用法" in replier.sent[0][1][0]
    assert "/圖片" in replier.sent[0][1][0]


@pytest.mark.asyncio
async def test_a_bare_trigger_word_gets_usage_help_and_is_not_queued(wired) -> None:
    handler, queue, replier = wired
    body = _body([_text_event("/影片")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "ignored"
    assert queue.counts() == {}
    assert "用法" in replier.sent[0][1][0]


# ------------------------------------------------------------------- dedupe


@pytest.mark.asyncio
async def test_a_redelivered_event_does_not_queue_a_second_time(wired) -> None:
    """LINE redelivers when it does not get a 2xx. This must not bill twice."""
    handler, queue, _ = wired
    body = _body([_text_event("/影片 一隻貓", event_id="evt-dup")])
    signature = sign(body, SECRET)

    first = await handler.handle(body, signature)
    second = await handler.handle(body, signature)

    assert first[0].action == "accepted"
    assert second[0].action == "duplicate"
    assert second[0].job is not None and second[0].job.id == first[0].job.id
    assert queue.counts() == {"queued": 1}


@pytest.mark.asyncio
async def test_two_events_in_one_delivery_are_both_handled(wired) -> None:
    handler, queue, _ = wired
    body = _body(
        [
            _text_event("/影片 貓", event_id="evt-a"),
            _text_event("/影片 狗", event_id="evt-b"),
        ]
    )
    outcomes = await handler.handle(body, sign(body, SECRET))
    assert [o.action for o in outcomes] == ["accepted", "accepted"]
    assert queue.counts() == {"queued": 2}


# ------------------------------------------------------------------- status


@pytest.mark.asyncio
async def test_asking_for_status_is_answered_from_the_queue(wired) -> None:
    """Replies are free, so polling by asking costs nothing."""
    handler, _queue, replier = wired
    body = _body([_text_event("/影片 一隻貓", event_id="evt-1")])
    await handler.handle(body, sign(body, SECRET))

    ask = _body([_text_event("好了嗎", event_id="evt-2")])
    (outcome,) = await handler.handle(ask, sign(ask, SECRET))

    assert outcome.action == "status"
    assert "排隊中 1 件" in replier.sent[-1][1][0]


@pytest.mark.asyncio
async def test_status_with_an_empty_queue_says_so(wired) -> None:
    handler, _, replier = wired
    body = _body([_text_event("進度", event_id="evt-s")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "status"
    assert "沒有排隊中" in replier.sent[-1][1][0]


# --------------------------------------------------------------- resilience


@pytest.mark.asyncio
async def test_a_failing_reply_does_not_break_acceptance(wired) -> None:
    """The 200 matters more than the reply: a non-2xx makes LINE redeliver."""
    handler, queue, _ = wired

    class Broken:
        async def reply_text(self, reply_token: str, *texts: str) -> None:
            raise RuntimeError("LINE is down")

    handler.replier = Broken()  # type: ignore[assignment]
    body = _body([_text_event("/影片 一隻貓")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "accepted"
    assert queue.counts() == {"queued": 1}


# ------------------------------------------------------------- capture mode


@pytest.fixture
def capture_mode(tmp_path: Path):
    """No allowlist configured yet — the state this project ships in."""
    queue = JobQueue(tmp_path / "q.sqlite3")
    replier = NullReplyClient()
    handler = WebhookHandler(
        queue,
        replier,
        channel_secret=SECRET,
        allowed_group_id=None,
        base_url="https://vg.example.com",
    )
    yield handler, queue, replier
    queue.close()


@pytest.mark.asyncio
async def test_an_unset_allowlist_never_accepts_work(capture_mode) -> None:
    """The hole this closes: unset must not mean 'serve every group'."""
    handler, queue, _ = capture_mode
    body = _body([_text_event("/影片 一隻貓", group=OTHER_GROUP)])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "capture"
    assert queue.counts() == {}, "capture mode must enqueue nothing"


@pytest.mark.asyncio
async def test_capture_mode_reports_the_group_id_for_copying(capture_mode) -> None:
    handler, _, replier = capture_mode
    body = _body([_text_event("/影片 一隻貓", group=OTHER_GROUP)])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.detail == OTHER_GROUP
    assert OTHER_GROUP in replier.sent[-1][1][0]
    assert "LINE_ALLOWED_GROUP_ID" in replier.sent[-1][1][0]


@pytest.mark.asyncio
async def test_capture_mode_stays_quiet_during_ordinary_chatter(capture_mode) -> None:
    """Setting the bot up must not spam the group."""
    handler, _, replier = capture_mode
    body = _body([_text_event("今天天氣真好")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "ignored"
    assert replier.sent == []


@pytest.mark.asyncio
async def test_capture_mode_explains_itself_in_a_one_to_one_chat(capture_mode) -> None:
    handler, _, replier = capture_mode
    event = _text_event("/影片 一隻貓")
    event["source"] = {"type": "user", "userId": "U" + "2" * 32}
    body = _body([event])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "capture"
    assert "群組" in replier.sent[-1][1][0]


# ----------------------------------------------------- the user allowlist

MEMBER = "U" + "1" * 32
STRANGER = "U" + "7" * 32


def _gated(tmp_path: Path, *, users):
    queue = JobQueue(tmp_path / "gated.sqlite3")
    replier = NullReplyClient()
    handler = WebhookHandler(
        queue,
        replier,
        channel_secret=SECRET,
        allowed_group_id=GROUP,
        allowed_user_ids=users,
        base_url="https://vg.example.com/",
    )
    return handler, queue, replier


async def _send(handler, event):
    body = _body([event])
    return await handler.handle(body, sign(body, SECRET))


@pytest.mark.asyncio
async def test_an_empty_user_allowlist_lets_any_group_member_through(tmp_path: Path) -> None:
    """The documented default. Group membership is the only boundary."""
    handler, queue, _ = _gated(tmp_path, users=())
    event = _text_event("/影片 一隻貓")
    del event["source"]["userId"]  # and even an unidentifiable one, by design

    outcomes = await _send(handler, event)

    assert outcomes[0].action == "accepted"
    queue.close()


@pytest.mark.asyncio
async def test_a_stranger_in_the_right_group_is_refused(tmp_path: Path) -> None:
    """The whole point: being in the group is not the same as being allowed.

    Anyone an existing member invites lands in the group without us doing
    anything, and a render costs real GPU time.
    """
    handler, queue, replier = _gated(tmp_path, users=[MEMBER])
    event = _text_event("/影片 一隻貓")
    event["source"]["userId"] = STRANGER

    outcomes = await _send(handler, event)

    assert outcomes[0].action == "wrong_user"
    assert STRANGER in outcomes[0].detail  # so the log alone can authorise them
    assert queue.counts() == {}, "a refused request must not reach the queue"
    assert replier.sent, "a refused trigger gets one free reply, not silence"
    queue.close()


@pytest.mark.asyncio
async def test_an_unknown_user_id_fails_closed(tmp_path: Path) -> None:
    """LINE omits source.userId for a user who has not accepted the Official
    Account terms. "We could not tell who this was" must never resolve to
    "let them spend a GPU-hour"."""
    handler, queue, _ = _gated(tmp_path, users=[MEMBER])
    event = _text_event("/影片 一隻貓")
    del event["source"]["userId"]

    outcomes = await _send(handler, event)

    assert outcomes[0].action == "wrong_user"
    assert queue.counts() == {}
    queue.close()


@pytest.mark.asyncio
async def test_an_allowed_member_still_gets_through(tmp_path: Path) -> None:
    handler, queue, _ = _gated(tmp_path, users=[MEMBER, STRANGER])
    outcomes = await _send(handler, _text_event("/影片 一隻貓"))
    assert outcomes[0].action == "accepted"
    assert queue.counts() == {"queued": 1}
    queue.close()


@pytest.mark.asyncio
async def test_chit_chat_from_a_stranger_is_ignored_not_refused(tmp_path: Path) -> None:
    """The gate sits in front of the paid action only. Refusing every ordinary
    message would make the bot talk constantly in a group it barely serves."""
    handler, queue, replier = _gated(tmp_path, users=[MEMBER])
    event = _text_event("今天天氣真好")
    event["source"]["userId"] = STRANGER

    outcomes = await _send(handler, event)

    assert outcomes[0].action == "ignored"
    assert not replier.sent
    queue.close()


# ------------------------------------------------------- membership events


@pytest.mark.asyncio
async def test_a_join_is_reported_with_the_new_user_ids(tmp_path: Path) -> None:
    """A join changes who can spend money, and LINE gives an unverified account
    no API to list a group's members - so this event is the only notice."""
    handler, queue, replier = _gated(tmp_path, users=[MEMBER])
    event = {
        "type": "memberJoined",
        "mode": "active",
        "webhookEventId": "evt-join",
        "timestamp": 1700000000000,
        "replyToken": "rt-join",
        "source": {"type": "group", "groupId": GROUP},
        "joined": {"members": [{"type": "user", "userId": STRANGER}]},
    }

    outcomes = await _send(handler, event)

    assert outcomes[0].action == "memberJoined"
    assert STRANGER in outcomes[0].detail
    assert not replier.sent, "a join is reported to the operator, not to the group"
    queue.close()


@pytest.mark.asyncio
async def test_a_leave_is_reported_too(tmp_path: Path) -> None:
    handler, queue, _ = _gated(tmp_path, users=[MEMBER])
    event = {
        "type": "memberLeft",
        "mode": "active",
        "webhookEventId": "evt-left",
        "timestamp": 1700000000000,
        "source": {"type": "group", "groupId": GROUP},
        "left": {"members": [{"type": "user", "userId": STRANGER}]},
    }
    outcomes = await _send(handler, event)
    assert outcomes[0].action == "memberLeft"
    assert STRANGER in outcomes[0].detail
    queue.close()


@pytest.mark.asyncio
async def test_a_join_in_another_group_is_not_reported(tmp_path: Path) -> None:
    """Only the served group's roster is our business. Any account can add this
    bot to any group, and each of those would otherwise write to our log."""
    handler, queue, _ = _gated(tmp_path, users=[MEMBER])
    event = {
        "type": "memberJoined",
        "mode": "active",
        "source": {"type": "group", "groupId": OTHER_GROUP},
        "joined": {"members": [{"type": "user", "userId": STRANGER}]},
    }
    outcomes = await _send(handler, event)
    assert outcomes[0].action == "ignored"
    queue.close()


# ---------------------------------------------------------- business hours

# 11:00-13:00 Asia/Taipei is the whole of "instant". These pin the two things
# the hours must never change: a request is accepted at any hour, and the
# acknowledgement stops promising "shortly" when it is eight hours away.

OPEN_INSTANT = datetime(2026, 8, 25, 12, 0, tzinfo=hours.TZ)
CLOSED_INSTANT = datetime(2026, 8, 25, 3, 0, tzinfo=hours.TZ)


def _at(tmp_path: Path, when: datetime, *, name: str = "hours", cap: int = 0, chat_cap: int = 0):
    queue = JobQueue(tmp_path / f"{name}.sqlite3")
    replier = NullReplyClient()
    handler = WebhookHandler(
        queue,
        replier,
        channel_secret=SECRET,
        allowed_group_id=GROUP,
        base_url="https://vg.example.com/",
        max_jobs_per_user_per_day=cap,
        max_chat_messages_per_user_per_day=chat_cap,
        clock=lambda: when,
    )
    return handler, queue, replier


@pytest.mark.asyncio
async def test_a_request_out_of_hours_is_still_accepted(tmp_path: Path) -> None:
    """Refusing would mean whatever someone thought of at midnight is lost.
    It waits in the queue for eleven instead."""
    handler, queue, _ = _at(tmp_path, CLOSED_INSTANT)

    outcomes = await _send(handler, _text_event("/影片 一隻貓", event_id="evt-night"))

    assert outcomes[0].action == "accepted"
    assert outcomes[0].job is not None
    assert queue.by_token(outcomes[0].job.token).state is JobState.QUEUED
    queue.close()


@pytest.mark.asyncio
async def test_the_same_request_gets_the_same_answer_in_and_out_of_hours(
    tmp_path: Path,
) -> None:
    """Same message, same outcome, same reply -- a deliberate choice: the
    accepted-message text no longer varies with the clock, only the status
    page behind the link does. Out-of-hours users get no ETA in the reply
    text itself any more; the link is the only place that distinction lives."""
    open_handler, open_queue, open_replier = _at(tmp_path, OPEN_INSTANT, name="open")
    shut_handler, shut_queue, shut_replier = _at(tmp_path, CLOSED_INSTANT, name="shut")

    event = _text_event("/影片 一隻貓", event_id="evt-same")
    open_outcomes = await _send(open_handler, event)
    shut_outcomes = await _send(shut_handler, event)

    assert open_outcomes[0].action == "accepted"
    assert shut_outcomes[0].action == "accepted"

    open_text = open_replier.sent[0][1][0]
    shut_text = shut_replier.sent[0][1][0]
    assert "想查進度可以看" in open_text
    assert "想查進度可以看" in shut_text
    assert open_text.split("/q/")[0] == shut_text.split("/q/")[0]

    open_queue.close()
    shut_queue.close()


# ------------------------------------------------------------- per-user cap


@pytest.mark.asyncio
async def test_a_user_over_the_daily_cap_is_refused_before_being_enqueued(
    tmp_path: Path,
) -> None:
    """Refused *before* the insert: accepting and then dropping it would still
    have spent an LLM conversion on a request that never runs."""
    handler, queue, replier = _at(tmp_path, OPEN_INSTANT, cap=2)

    for n in range(2):
        assert (await _send(handler, _text_event("/影片 貓", event_id=f"ok-{n}")))[0].action == (
            "accepted"
        )

    outcomes = await _send(handler, _text_event("/影片 貓", event_id="over"))

    assert outcomes[0].action == "rate_limited"
    assert outcomes[0].job is None
    assert len(queue.recent(10)) == 2, "the refused request was never inserted"
    assert "每日上限" in replier.sent[-1][1][0]
    queue.close()


@pytest.mark.asyncio
async def test_the_cap_counts_per_user_not_per_group(tmp_path: Path) -> None:
    handler, queue, _ = _at(tmp_path, OPEN_INSTANT, cap=1)
    first = _text_event("/影片 貓", event_id="u1")
    second = _text_event("/影片 狗", event_id="u2")
    second["source"]["userId"] = STRANGER

    assert (await _send(handler, first))[0].action == "accepted"
    assert (await _send(handler, second))[0].action == "accepted"
    queue.close()


@pytest.mark.asyncio
async def test_the_cap_is_off_by_default(tmp_path: Path) -> None:
    """A handler built without the setting must not silently throttle. The
    composition root passes the real value; the default here is `off`."""
    handler, queue, _ = _at(tmp_path, OPEN_INSTANT)

    for n in range(12):
        outcomes = await _send(handler, _text_event("/影片 貓", event_id=f"many-{n}"))
        assert outcomes[0].action == "accepted"
    queue.close()


@pytest.mark.asyncio
async def test_chat_has_its_own_separate_daily_cap(tmp_path: Path) -> None:
    """A `/himonkey` conversation must not exhaust a user's video/image
    allowance, and vice versa -- the two caps are independent counters."""
    handler, queue, replier = _at(tmp_path, OPEN_INSTANT, cap=1, chat_cap=2)

    for n in range(2):
        outcomes = await _send(handler, _text_event(f"/himonkey 嗨 {n}", event_id=f"chat-ok-{n}"))
        assert outcomes[0].action == "accepted"

    over = await _send(handler, _text_event("/himonkey 嗨 3", event_id="chat-over"))
    assert over[0].action == "rate_limited"
    assert "每日上限" in replier.sent[-1][1][0]

    # The video cap (1) is untouched by the two chat messages above.
    video = await _send(handler, _text_event("/影片 貓", event_id="video-ok"))
    assert video[0].action == "accepted"
    queue.close()


@pytest.mark.asyncio
async def test_chat_messages_do_not_count_against_the_video_image_cap(tmp_path: Path) -> None:
    handler, queue, _ = _at(tmp_path, OPEN_INSTANT, cap=1, chat_cap=0)

    for n in range(5):
        outcomes = await _send(handler, _text_event(f"/himonkey 嗨 {n}", event_id=f"c-{n}"))
        assert outcomes[0].action == "accepted"

    video = await _send(handler, _text_event("/影片 貓", event_id="video-still-ok"))
    assert video[0].action == "accepted"
    queue.close()


@pytest.mark.asyncio
async def test_a_failed_request_still_counts_against_the_cap(tmp_path: Path) -> None:
    """The cap is on asking, not on succeeding — otherwise a user whose prompts
    keep failing validation has an unlimited allowance."""
    handler, queue, _ = _at(tmp_path, OPEN_INSTANT, cap=1)
    outcomes = await _send(handler, _text_event("/影片 貓", event_id="doomed"))
    assert outcomes[0].job is not None
    queue.fail(outcomes[0].job.id, "prompt rejected")

    again = await _send(handler, _text_event("/影片 貓", event_id="second"))

    assert again[0].action == "rate_limited"
    queue.close()


# ------------------------------------------------------------ quoted reply


@pytest.mark.asyncio
async def test_the_request_quote_token_is_kept_for_delivery(wired) -> None:
    """Delivery replies to the request message (LINE's quoted-message card)
    rather than @-mentioning, so the token LINE hands us with the message has
    to survive until the render is done -- hours later, in another process."""
    handler, queue, _ = wired
    event = _text_event("/影片 一隻貓", event_id="evt-q")
    event["message"]["quoteToken"] = "qt-abc"
    body = _body([event])

    (outcome,) = await handler.handle(body, sign(body, SECRET))

    assert outcome.job is not None
    assert outcome.job.quote_token == "qt-abc"
    assert queue.by_id(outcome.job.id).quote_token == "qt-abc"


@pytest.mark.asyncio
async def test_a_message_without_a_quote_token_is_still_accepted(wired) -> None:
    handler, _, _ = wired
    body = _body([_text_event("/影片 一隻貓", event_id="evt-nq")])
    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "accepted" and outcome.job.quote_token is None


# ------------------------------------------------------------- clip length


@pytest.mark.asyncio
async def test_a_length_suffix_on_a_video_trigger_is_captured(wired) -> None:
    """`/影片15s a cat` asks for a 15-second clip; the number rides on the
    trigger and the rest is the prompt."""
    handler, queue, _ = wired
    body = _body([_text_event("/影片15s 一隻貓", event_id="evt-15")])
    (outcome,) = await handler.handle(body, sign(body, SECRET))

    assert outcome.action == "accepted"
    assert outcome.job.text == "一隻貓", "the length is stripped, not left in the prompt"
    assert outcome.job.requested_seconds == 15.0
    assert queue.by_id(outcome.job.id).requested_seconds == 15.0


@pytest.mark.asyncio
async def test_the_cjk_seconds_suffix_also_works(wired) -> None:
    handler, _, _ = wired
    body = _body([_text_event("/圖影10秒 讓照片動起來", event_id="evt-10")])
    # no photo cached, so it is refused -- but the parse still had to succeed
    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.detail == "no pending image"


@pytest.mark.asyncio
async def test_no_suffix_leaves_requested_seconds_none(wired) -> None:
    handler, _, _ = wired
    body = _body([_text_event("/影片 一隻貓", event_id="evt-none")])
    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.job.requested_seconds is None


@pytest.mark.asyncio
async def test_an_image_trigger_ignores_a_number_in_the_prompt(wired) -> None:
    """`/圖片` has no length. A prompt that starts with a number is prompt,
    not a length: `3 cats` stays `3 cats`."""
    handler, _, _ = wired
    body = _body([_text_event("/圖片3 隻貓", event_id="evt-img3")])
    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.job.requested_seconds is None
    assert outcome.job.text == "3 隻貓"


@pytest.mark.asyncio
async def test_a_fullwidth_slash_from_a_mobile_ime_still_triggers(wired) -> None:
    """A Chinese IME renders a leading "/" as the fullwidth solidus (U+FF0F)
    more often than not. Without normalising it the message matched nothing
    and the bot looked dead to the user."""
    handler, _, _ = wired
    body = _body([_text_event("\uff0f圖片 一隻貓", event_id="evt-fw")])
    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "accepted"
    assert outcome.job.media_kind is MediaKind.IMAGE
    assert outcome.job.text == "一隻貓"


# ------------------------------------------------------- 讓我看看 (pull)


def _finish(queue, job_id: int, output: str | None = None, *, text: str | None = None,
            error: str | None = None):
    job = queue.by_id(job_id)
    queue.set_parsed(job.id, {})
    queue.claim_next()
    if error is not None:
        queue.fail(job.id, error, requeue=False)
    elif text is not None:
        queue.complete_text(job.id, text)
    else:
        queue.complete(job.id, output or "runs/x/clip.mp4")
    return queue.by_id(job.id)


@pytest.mark.asyncio
async def test_quoting_a_request_hands_back_only_that_job(wired, tmp_path: Path) -> None:
    handler, queue, replier = wired
    handler.files_dir = tmp_path
    body = _body([_text_event("/影片 一隻貓", event_id="e1"),
                  _text_event("/影片 一隻狗", event_id="e2")])
    body_json = json.loads(body)
    body_json["events"][0]["message"]["id"] = "m-cat"
    body_json["events"][1]["message"]["id"] = "m-dog"
    body = json.dumps(body_json).encode()
    cat, dog = await handler.handle(body, sign(body, SECRET))
    _finish(queue, cat.job.id, "runs/x/cat.mp4")
    (tmp_path / "cat_poster.jpg").write_bytes(b"jpg")
    _finish(queue, dog.job.id, "runs/x/dog.mp4")

    ask = _body([_text_event("讓我看看", event_id="e3", message={
        "type": "text", "id": "m3", "text": "讓我看看", "quotedMessageId": "m-cat"})])
    (outcome,) = await handler.handle(ask, sign(ask, SECRET))

    assert outcome.action == "show" and outcome.job.id == cat.job.id
    _, messages = replier.sent_messages[-1]
    assert messages[0]["type"] == "video"
    assert messages[0]["originalContentUrl"].endswith("/files/cat.mp4")
    assert messages[0]["previewImageUrl"].endswith("/files/cat_poster.jpg")
    assert queue.by_id(cat.job.id).delivered_at is not None
    assert queue.by_id(dog.job.id).delivered_at is None  # the other one stays for later


@pytest.mark.asyncio
async def test_quoting_the_bots_own_receipt_names_the_request_too(wired, tmp_path: Path) -> None:
    handler, queue, replier = wired
    body = _body([_text_event("/影片 一隻貓", event_id="e1")])
    (accepted,) = await handler.handle(body, sign(body, SECRET))
    receipt_id = replier.sent[-1][0]  # NullReplyClient ids are sent-<token>-<n>
    assert queue.by_id(accepted.job.id).reply_message_id == f"sent-{receipt_id}-0"
    _finish(queue, accepted.job.id, text="一隻橘貓坐在窗台上")

    ask = _body([_text_event("讓我看看", event_id="e2", message={
        "type": "text", "id": "m2", "text": "讓我看看",
        "quotedMessageId": f"sent-{receipt_id}-0"})])
    (outcome,) = await handler.handle(ask, sign(ask, SECRET))
    assert outcome.detail == "handed over"
    assert "一隻橘貓" in replier.sent_messages[-1][1][-1]["text"]


@pytest.mark.asyncio
async def test_quoting_an_unfinished_or_failed_job_says_so_explicitly(wired) -> None:
    handler, queue, replier = wired
    body = _body([_text_event("/影片 一隻貓", event_id="e1", message={
        "type": "text", "id": "m-cat", "text": "/影片 一隻貓"})])
    (accepted,) = await handler.handle(body, sign(body, SECRET))

    def ask(eid):
        return _body([_text_event("讓我看看", event_id=eid, message={
            "type": "text", "id": eid, "text": "讓我看看", "quotedMessageId": "m-cat"})])

    (o,) = await handler.handle(ask("e2"), sign(ask("e2"), SECRET))
    assert o.detail == "not done" and "還沒好" in replier.sent[-1][1][0]
    assert queue.by_id(accepted.job.id).delivered_at is None

    _finish(queue, accepted.job.id, error="OOM on the pod")
    (o,) = await handler.handle(ask("e3"), sign(ask("e3"), SECRET))
    assert o.detail == "failed" and "失敗" in replier.sent[-1][1][0]
    assert "OOM" in replier.sent[-1][1][0]

    unknown = _body([_text_event("讓我看看", event_id="e4", message={
        "type": "text", "id": "e4", "text": "讓我看看", "quotedMessageId": "nope"})])
    (o,) = await handler.handle(unknown, sign(unknown, SECRET))
    assert o.detail == "quoted message unknown"
    assert "沒有對應" in replier.sent[-1][1][0]


@pytest.mark.asyncio
async def test_bare_show_hands_over_everything_undelivered_and_marks_it(wired, tmp_path: Path) -> None:
    handler, queue, replier = wired
    handler.files_dir = tmp_path
    body = _body([_text_event("/圖片 一隻貓", event_id="e1"), _text_event("/影片 一隻狗", event_id="e2")])
    a, b = await handler.handle(body, sign(body, SECRET))
    _finish(queue, a.job.id, "runs/x/cat.png")
    _finish(queue, b.job.id, "runs/x/dog.mp4")  # no poster -> link only

    ask = _body([_text_event("/讓我看看", event_id="e3")])
    (outcome,) = await handler.handle(ask, sign(ask, SECRET))

    assert outcome.detail == "2 handed over"
    _, messages = replier.sent_messages[-1]
    assert [m["type"] for m in messages] == ["image", "text"]
    assert "dog.mp4" not in messages[-1]["text"] and "/q/" in messages[-1]["text"]
    assert all(queue.by_id(j.id).delivered_at for j in (a.job, b.job))

    again = _body([_text_event("讓我看看", event_id="e4")])
    await handler.handle(again, sign(again, SECRET))
    assert "沒有等著給你看的成品" in replier.sent[-1][1][0]


@pytest.mark.asyncio
async def test_accept_warns_about_the_pull_once_push_quota_is_gone(wired) -> None:
    handler, queue, replier = wired
    queue.note_push_quota_exhausted()
    body = _body([_text_event("/影片 一隻貓", event_id="e1")])
    await handler.handle(body, sign(body, SECRET))
    assert "讓我看看" in replier.sent[-1][1][0] and "推播額度" in replier.sent[-1][1][0]


@pytest.mark.asyncio
async def test_every_link_sits_on_its_own_line(wired, tmp_path: Path) -> None:
    """LINE renders a URL glued to text as one unbroken run; a newline before
    it makes the link tappable and the sentence readable (asked 2026-08-27)."""
    import re

    handler, queue, replier = wired
    body = _body([_text_event("/影片 一隻貓", event_id="e1")])
    (accepted,) = await handler.handle(body, sign(body, SECRET))
    _finish(queue, accepted.job.id, "runs/x/cat.mp4")
    ask = _body([_text_event("讓我看看", event_id="e2")])
    await handler.handle(ask, sign(ask, SECRET))
    status = _body([_text_event("進度", event_id="e3")])
    await handler.handle(status, sign(status, SECRET))

    texts = [t for _, ts in replier.sent for t in ts]
    texts += [m["text"] for _, ms in replier.sent_messages for m in ms if m["type"] == "text"]
    for text in texts:
        for m in re.finditer(r"https://", text):
            assert m.start() == 0 or text[m.start() - 1] == "\n", text


# ----------------------------------------------------------------- /短劇


@pytest.mark.asyncio
async def test_the_drama_trigger_enqueues_a_drama_job(wired) -> None:
    handler, _queue, replier = wired
    body = _body([_text_event("/短劇 一個夜市老闆娘發現攤位下藏著一封信")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))

    assert outcome.action == "accepted"
    assert outcome.job is not None
    assert outcome.job.media_kind is MediaKind.DRAMA
    assert outcome.job.text == "一個夜市老闆娘發現攤位下藏著一封信"
    assert outcome.job.first_frame_path is None and outcome.job.requested_seconds is None
    (_, texts) = replier.sent[0]
    assert "短劇" in texts[0] and "分鐘" in texts[0]


@pytest.mark.asyncio
async def test_a_fullwidth_slash_drama_trigger_still_fires(wired) -> None:
    handler, _queue, _replier = wired
    body = _body([_text_event("\uff0f短劇 一個故事")])  # the IME's fullwidth solidus
    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "accepted" and outcome.job.media_kind is MediaKind.DRAMA


@pytest.mark.asyncio
async def test_a_drama_never_parses_a_length_suffix(wired) -> None:
    """The length is the six shots. `15s` stays in the premise, untouched."""
    handler, _queue, _replier = wired
    body = _body([_text_event("/短劇15s 一個故事")])
    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "accepted"
    assert outcome.job.requested_seconds is None
    assert outcome.job.text == "15s 一個故事"


@pytest.mark.asyncio
async def test_an_empty_drama_premise_gets_usage(wired) -> None:
    handler, queue, replier = wired
    body = _body([_text_event("/短劇")])
    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "ignored"
    assert queue.counts() == {}
    assert "一句故事前提" in replier.sent[0][1][0]


@pytest.mark.asyncio
async def test_the_group_wide_drama_cap_refuses_the_next_one(tmp_path: Path) -> None:
    """Group-wide, not per user: two different members share the day's dramas,
    and the third is refused with a reply that names the other triggers."""
    queue = JobQueue(tmp_path / "q.sqlite3")
    replier = NullReplyClient()
    handler = WebhookHandler(
        queue, replier, channel_secret=SECRET, allowed_group_id=GROUP,
        base_url="https://vg.example.com/", clock=lambda: OPEN_INSTANT, max_dramas_per_day=2,
    )
    try:
        for i, user in enumerate(("U" + "1" * 32, "U" + "2" * 32)):
            body = _body([_text_event(f"/短劇 故事 {i}", event_id=f"evt-drama-{i}",
                                      source={"type": "group", "groupId": GROUP, "userId": user})])
            (outcome,) = await handler.handle(body, sign(body, SECRET))
            assert outcome.action == "accepted", i

        body = _body([_text_event("/短劇 故事 3", event_id="evt-drama-3")])
        (outcome,) = await handler.handle(body, sign(body, SECRET))
        assert outcome.action == "rate_limited" and outcome.detail == "drama cap"
        assert "短劇額度" in replier.sent[-1][1][0]
        assert queue.accepted_kind_today(MediaKind.DRAMA) == 2

        # Other kinds are untouched by the drama cap.
        body = _body([_text_event("/影片 一隻貓", event_id="evt-video-after-cap")])
        (outcome,) = await handler.handle(body, sign(body, SECRET))
        assert outcome.action == "accepted"
    finally:
        queue.close()
