"""LINE webhook: signature, filtering, dedupe, and the two-second path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videogen.bots.line.reply import NullReplyClient
from videogen.bots.line.verify import sign, verify
from videogen.bots.line.webhook import InvalidSignature, WebhookHandler
from videogen.pipeline.queue import JobQueue, JobState

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
        await handler.handle(_body([_text_event("生成 一隻貓")]), "not-a-signature")


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
    body = _body([_text_event("生成 一隻貓", group=OTHER_GROUP)])

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
    body = _body([_text_event("生成 一隻貓", mode="standby")])

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
    body = _body([_text_event("生成 一隻橘貓走在雨中")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))

    assert outcome.action == "accepted"
    assert outcome.job is not None
    assert outcome.job.text == "一隻橘貓走在雨中", "the trigger word is stripped"
    assert outcome.job.state is JobState.QUEUED

    (_, texts) = replier.sent[0]
    assert "排隊第 1 位" in texts[0]
    assert f"https://vg.example.com/q/{outcome.job.token}" in texts[0]


@pytest.mark.asyncio
async def test_slash_forms_also_trigger(wired) -> None:
    handler, _, _ = wired
    for i, text in enumerate(("/生成 一隻貓", "/gen a cat")):
        body = _body([_text_event(text, event_id=f"evt-slash-{i}")])
        (outcome,) = await handler.handle(body, sign(body, SECRET))
        assert outcome.action == "accepted"


@pytest.mark.asyncio
async def test_a_bare_trigger_word_gets_usage_help_and_is_not_queued(wired) -> None:
    handler, queue, replier = wired
    body = _body([_text_event("生成")])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "ignored"
    assert queue.counts() == {}
    assert "用法" in replier.sent[0][1][0]


# ------------------------------------------------------------------- dedupe


@pytest.mark.asyncio
async def test_a_redelivered_event_does_not_queue_a_second_time(wired) -> None:
    """LINE redelivers when it does not get a 2xx. This must not bill twice."""
    handler, queue, _ = wired
    body = _body([_text_event("生成 一隻貓", event_id="evt-dup")])
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
            _text_event("生成 貓", event_id="evt-a"),
            _text_event("生成 狗", event_id="evt-b"),
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
    body = _body([_text_event("生成 一隻貓", event_id="evt-1")])
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
    body = _body([_text_event("生成 一隻貓")])

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
    body = _body([_text_event("生成 一隻貓", group=OTHER_GROUP)])

    (outcome,) = await handler.handle(body, sign(body, SECRET))
    assert outcome.action == "capture"
    assert queue.counts() == {}, "capture mode must enqueue nothing"


@pytest.mark.asyncio
async def test_capture_mode_reports_the_group_id_for_copying(capture_mode) -> None:
    handler, _, replier = capture_mode
    body = _body([_text_event("生成 一隻貓", group=OTHER_GROUP)])

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
    event = _text_event("生成 一隻貓")
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
    event = _text_event("生成 一隻貓")
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
    event = _text_event("生成 一隻貓")
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
    event = _text_event("生成 一隻貓")
    del event["source"]["userId"]

    outcomes = await _send(handler, event)

    assert outcomes[0].action == "wrong_user"
    assert queue.counts() == {}
    queue.close()


@pytest.mark.asyncio
async def test_an_allowed_member_still_gets_through(tmp_path: Path) -> None:
    handler, queue, _ = _gated(tmp_path, users=[MEMBER, STRANGER])
    outcomes = await _send(handler, _text_event("生成 一隻貓"))
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
