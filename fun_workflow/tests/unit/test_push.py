"""Push delivery: the message objects, the quoted reply, and the two degradations.

Push is billed **per recipient**, so every assertion here is either about
sending the right thing once, or about not going silent when sending fails.
Silence is the expensive failure: the user is waiting, and a bot that says
nothing is indistinguishable from a bot that is broken — so they ask again,
which costs another conversion and another GPU slot.
"""

from __future__ import annotations

from typing import Any

import pytest

from fun_workflow.bots.line.push import (
    LineMediaRejected,
    LinePushError,
    LineQuotaExhausted,
    NullPushClient,
    deliver,
    delivered_messages,
    failed_messages,
    image_message,
    text_message,
    understood_messages,
    video_message,
)

GROUP = "Cae56f94637c1234567890abcdef12345"
USER = "U" + "1" * 32
BASE = "https://vg.example.com"
QUOTE = "quote-token-from-the-request-message"


class FakePushClient:
    """Records payloads, and can fail the first send on demand."""

    def __init__(self, *, fail_first: Exception | None = None) -> None:
        self.fail_first = fail_first
        self.sent: list[tuple[str, list[dict[str, Any]], str | None]] = []

    async def push(
        self, to: str, messages: list[dict[str, Any]], *, retry_key: str | None = None
    ) -> None:
        if self.fail_first is not None and not self.sent:
            self.sent.append((to, messages, retry_key))
            raise self.fail_first
        self.sent.append((to, messages, retry_key))


# ---------------------------------------------------------- message objects


def test_a_video_delivery_is_a_video_object_then_a_text_object() -> None:
    """Media first so the group sees the thing before the caption — and the
    quote has to live on the text, because a video object cannot carry one."""
    messages = delivered_messages(
        media_url=f"{BASE}/files/abc.mp4",
        preview_url=f"{BASE}/files/abc_poster.jpg",
        status_url=f"{BASE}/q/abc",
        is_video=True,
        prompt="一隻橘貓走在雨中",
        quote_token=QUOTE,
    )

    assert [m["type"] for m in messages] == ["video", "text"]
    assert messages[1]["quoteToken"] == QUOTE
    assert messages[0]["originalContentUrl"] == f"{BASE}/files/abc.mp4"
    assert messages[0]["previewImageUrl"] == f"{BASE}/files/abc_poster.jpg"
    assert "quoteToken" not in messages[0]


def test_an_image_delivery_is_an_image_object_then_a_text_object() -> None:
    messages = delivered_messages(
        media_url=f"{BASE}/files/abc.png",
        preview_url=f"{BASE}/files/abc_poster.jpg",
        status_url=f"{BASE}/q/abc",
        is_video=False,
        prompt="a fox",
        quote_token=QUOTE,
    )

    assert [m["type"] for m in messages] == ["image", "text"]
    assert messages[0]["originalContentUrl"].endswith(".png")


def test_a_preview_is_never_the_original() -> None:
    """Flux writes 1024x1024 PNGs, which routinely clear the 1MB preview
    ceiling on their own. Reusing the original is how a whole message object
    gets rejected."""
    messages = delivered_messages(
        media_url=f"{BASE}/files/abc.png",
        preview_url=f"{BASE}/files/abc_poster.jpg",
        status_url=f"{BASE}/q/abc",
        is_video=False,
        prompt="a fox",
        quote_token=QUOTE,
    )
    assert messages[0]["previewImageUrl"] != messages[0]["originalContentUrl"]


# -------------------------------------------------------------- quoted reply


def test_the_caption_is_a_reply_to_the_request_message() -> None:
    """Not an @-mention: the legacy mention object shipped as the literal text
    "@你", and the group asked for the reply UI a person gets when someone
    answers them. That is `quoteToken` on the text object."""
    message = text_message("你的影片好了", quote_token=QUOTE)

    assert message == {"type": "text", "text": "你的影片好了", "quoteToken": QUOTE}
    assert "mention" not in message and "substitution" not in message


def test_no_quote_token_means_a_plain_message_not_an_error() -> None:
    """LINE omits the quote token on some message kinds. Losing the whole
    delivery over a missing quote is a far worse trade than delivering it
    unquoted."""
    message = text_message("你的影片好了", quote_token=None)

    assert message == {"type": "text", "text": "你的影片好了"}


def test_an_understanding_delivery_is_a_single_text_message() -> None:
    """/說圖 /說音 /說影 produce text, not a file -- there is no media object
    and nothing to attach a caption to, unlike `delivered_messages`."""
    (message,) = understood_messages(
        result_text="一隻橘貓坐在窗邊,望向外面下雨的街道。",
        status_url=f"{BASE}/q/abc",
        quote_token=QUOTE,
    )

    assert message["type"] == "text"
    assert message["quoteToken"] == QUOTE
    assert "一隻橘貓坐在窗邊" in message["text"]
    assert f"{BASE}/q/abc" in message["text"]


def test_a_failed_request_still_replies_to_the_person_waiting_on_it() -> None:
    (message,) = failed_messages(
        reason="provider failed 3x: the pod died",
        status_url=f"{BASE}/q/abc",
        prompt="一隻橘貓",
        quote_token=QUOTE,
    )

    assert message["type"] == "text"
    assert message["quoteToken"] == QUOTE
    assert "the pod died" in message["text"]
    assert f"{BASE}/q/abc" in message["text"]


# ------------------------------------------------------------------ deliver


@pytest.mark.asyncio
async def test_a_successful_delivery_sends_once_and_carries_the_retry_key() -> None:
    """`X-Line-Retry-Key` is what makes a retry after a timeout free rather
    than a second charge."""
    client = FakePushClient()

    outcome = await deliver(
        client, to=GROUP, messages=[text_message("hi")],
        fallback_text="hi", retry_key="tok-123",
    )

    assert outcome == "pushed"
    assert len(client.sent) == 1
    to, _, retry_key = client.sent[0]
    assert to == GROUP and retry_key == "tok-123"


@pytest.mark.asyncio
async def test_quota_exhaustion_degrades_to_text_rather_than_to_silence() -> None:
    """429 means no media can be sent for the rest of the month. The user still
    has to be told where their video is."""
    client = FakePushClient(fail_first=LineQuotaExhausted("monthly limit reached"))

    outcome = await deliver(
        client, to=GROUP,
        messages=delivered_messages(
            media_url=f"{BASE}/files/a.mp4", preview_url=f"{BASE}/files/a.jpg",
            status_url=f"{BASE}/q/a", is_video=True, prompt="cat", quote_token=QUOTE,
        ),
        fallback_text=f"cat 完成了\n{BASE}/q/a",
        retry_key="tok-1",
        quote_token=QUOTE,
    )

    assert outcome == "quota-exhausted-text-only"
    assert len(client.sent) == 2
    fallback = client.sent[1][1]
    assert [m["type"] for m in fallback] == ["text"]
    assert f"{BASE}/q/a" in fallback[0]["text"]
    assert fallback[0]["quoteToken"] == QUOTE


@pytest.mark.asyncio
async def test_a_rejected_media_object_is_a_different_outcome_from_a_quota_wall() -> None:
    """400 is a bug — an oversized poster, a mismatched aspect, a host that
    will not answer a range request — and it will fail identically until
    someone fixes it. 429 is a money condition. Same fallback, different
    diagnosis, so they must not report as one thing."""
    client = FakePushClient(fail_first=LineMediaRejected("invalid previewImageUrl"))

    outcome = await deliver(
        client, to=GROUP, messages=[video_message("u", "p")],
        fallback_text="see the link", retry_key="tok-2",
    )

    assert outcome == "media-rejected-text-only"
    assert len(client.sent) == 2


@pytest.mark.asyncio
async def test_the_retry_key_differs_between_the_media_send_and_the_fallback() -> None:
    """Same key would make LINE treat the text fallback as a duplicate of the
    media send it is replacing, and drop it."""
    client = FakePushClient(fail_first=LineQuotaExhausted("gone"))

    await deliver(
        client, to=GROUP, messages=[image_message("u", "p")],
        fallback_text="see the link", retry_key="tok-3",
    )

    assert client.sent[0][2] == "tok-3"
    assert client.sent[1][2] != "tok-3"


@pytest.mark.asyncio
async def test_a_fallback_that_also_fails_is_reported_not_swallowed() -> None:
    """The one outcome that must never be indistinguishable from success."""

    class AlwaysFails:
        async def push(self, to: str, messages: list[dict[str, Any]], **kw: Any) -> None:
            raise LinePushError("everything is down")

    outcome = await deliver(
        AlwaysFails(), to=GROUP, messages=[text_message("hi")], fallback_text="hi"
    )

    assert outcome.endswith("-and-silent")


@pytest.mark.asyncio
async def test_the_null_client_records_instead_of_sending() -> None:
    """Running without credentials must not mean crashing, and must not look
    like a successful send either."""
    client = NullPushClient()

    await client.push(GROUP, [text_message("hi")], retry_key="tok-4")

    assert client.sent == [(GROUP, [{"type": "text", "text": "hi"}], "tok-4")]
