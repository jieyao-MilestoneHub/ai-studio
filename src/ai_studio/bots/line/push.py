"""LINE Push API client: send the finished media back into the group.

This module reverses a decision that `reply.py` documents at length, so the
reasoning it reverses is worth stating rather than quietly contradicting.

Reply is free and push is billed **per recipient** — one push into a
twenty-person group costs twenty messages against the monthly quota. That is
why delivery was a link in a free reply, and on cost alone it is still the
better answer. What it is not is the *product*: the goal is a clip that lands
in the group where the person asked for it, addressed to them. A link to a
status page is a receipt for a thing that happened somewhere else.

So the cost is now something to manage rather than avoid, and the two things
that manage it are: this is one private group, and running out of quota is
handled explicitly below rather than discovered as silence.

Three hard requirements LINE puts on media message objects, all of which the
old link-based delivery got to ignore:

- **video**: `originalContentUrl` (mp4, <=200MB) + `previewImageUrl` (<=1MB,
  matching aspect ratio)
- **image**: `originalContentUrl` (JPEG/PNG, <=10MB) + `previewImageUrl` (<=1MB)
- the host must answer **HTTP range requests** (`api.main`'s `/files/{name}`
  does — `FileResponse` returns 206 with a correct `Content-Range`, and there
  is a test pinning it, because a video message fails in a very hard-to-trace
  way when it does not)

References:
https://developers.line.biz/en/reference/messaging-api/#send-push-message
https://developers.line.biz/en/reference/messaging-api/#video-message
https://developers.line.biz/en/docs/messaging-api/pricing/
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol

import httpx

PUSH_ENDPOINT = "https://api.line.me/v2/bot/message/push"

MAX_TEXT_CHARS = 5000
MAX_MESSAGES_PER_PUSH = 5

"""Delivery replies to the request instead of @-mentioning the requester.

`quoteToken` on the caption makes LINE render it as a reply to the original
message -- the quoted card a person gets when someone answers them -- which
is what the group asked for after an @-mention shipped as the literal text
"@你". Only text and sticker objects can carry a quote, so it lives on the
caption; the media object goes first and unquoted. A token LINE no longer
honours is not a delivery failure: the caption just arrives unquoted.
"""

_log = logging.getLogger("ai_studio.push")


class LinePushError(Exception):
    """The Push API refused. Carries LINE's own message, which is specific."""


class LineQuotaExhausted(LinePushError):
    """429 — the monthly message allowance is gone.

    Its own type because the response to it is specific and must not be
    silence: fall back to a free-shaped plain text message so the user is told
    where their video is, even though the video itself cannot be pushed.
    """


class LineMediaRejected(LinePushError):
    """400 — LINE would not take the media object.

    Over a size ceiling, an unreachable URL, a poster whose aspect ratio does
    not match, a host that will not answer a range request. Distinct from a
    quota problem because the fix is completely different, and because
    retrying it unchanged will fail identically.
    """


class PushClientLike(Protocol):
    async def push(
        self, to: str, messages: list[dict[str, Any]], *, retry_key: str | None = None
    ) -> None: ...


# ------------------------------------------------------------------ builders


def text_message(text: str, *, quote_token: str | None = None) -> dict[str, Any]:
    """A text message object, optionally as a reply to `quote_token`'s message.

    No token means a plain message rather than an error: LINE omits the
    quote token on some message kinds, and losing the whole delivery over a
    missing quote is a far worse trade than delivering it unquoted.
    """
    message: dict[str, Any] = {"type": "text", "text": text[:MAX_TEXT_CHARS]}
    if quote_token:
        message["quoteToken"] = quote_token
    return message


def video_message(url: str, preview_url: str) -> dict[str, Any]:
    return {
        "type": "video",
        "originalContentUrl": url,
        "previewImageUrl": preview_url,
    }


def image_message(url: str, preview_url: str) -> dict[str, Any]:
    return {
        "type": "image",
        "originalContentUrl": url,
        "previewImageUrl": preview_url,
    }


# -------------------------------------------------------------------- client


class LinePushClient:
    """Sends push messages. Billed per recipient — see the module docstring."""

    def __init__(self, access_token: str, *, timeout_s: float = 10.0) -> None:
        self._token = access_token
        self._timeout = timeout_s

    async def push(
        self, to: str, messages: list[dict[str, Any]], *, retry_key: str | None = None
    ) -> None:
        """Push into a group or to a user. Raises a typed `LinePushError`.

        `retry_key` becomes `X-Line-Retry-Key`, which is what makes a retry
        after a timeout safe: LINE treats a repeat of the same key as the same
        send rather than a second one. Every caller's natural value for this
        (the job token, "preflight", a token with "-text" appended) is a
        short, non-UUID string -- and LINE rejects the header outright unless
        it is shaped like one. `uuid5` derives a stable UUID from whatever
        string the caller passes, so the same logical retry always maps to
        the same key without every call site needing to know LINE's format
        requirement. Observed live: a real push 400'd on this before the fix
        existed, with the media object never even evaluated.
        """
        if not messages:
            return
        if len(messages) > MAX_MESSAGES_PER_PUSH:
            raise LinePushError(f"at most {MAX_MESSAGES_PER_PUSH} messages per push")

        headers = {"Authorization": f"Bearer {self._token}"}
        if retry_key:
            headers["X-Line-Retry-Key"] = str(uuid.uuid5(uuid.NAMESPACE_URL, retry_key))

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    PUSH_ENDPOINT, json={"to": to, "messages": messages}, headers=headers
                )
            except httpx.HTTPError as exc:
                raise LinePushError(f"could not reach the LINE Push API: {exc}") from exc

        if response.status_code == 429:
            raise LineQuotaExhausted(f"push quota exhausted: {response.text[:300]}")
        if response.status_code == 400:
            raise LineMediaRejected(f"push rejected (400): {response.text[:500]}")
        if response.status_code >= 400:
            raise LinePushError(f"push failed ({response.status_code}): {response.text[:500]}")


class NullPushClient:
    """Records instead of sending. For tests and for running without credentials.

    Deliberately not a silent no-op: `sent` is what a test asserts against, and
    the log line is what tells an operator running without a token that
    delivery is configured but inert.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, list[dict[str, Any]], str | None]] = []

    async def push(
        self, to: str, messages: list[dict[str, Any]], *, retry_key: str | None = None
    ) -> None:
        self.sent.append((to, messages, retry_key))
        _log.info("push (not sent, no credentials): to=%s %d message(s)", to, len(messages))


# -------------------------------------------------------------------- policy


def delivered_messages(
    *,
    media_url: str,
    preview_url: str,
    status_url: str,
    is_video: bool,
    prompt: str,
    quote_token: str | None,
) -> list[dict[str, Any]]:
    """The two message objects a finished request produces: media, then text.

    Media first so the group sees the thing itself before the caption, and the
    quote lives on the text object because a media object cannot carry one.
    """
    media = video_message(media_url, preview_url) if is_video else image_message(
        media_url, preview_url
    )
    caption = text_message(f"{prompt[:60]} 完成了\n{status_url}", quote_token=quote_token)
    return [media, caption]


def understood_messages(
    *, result_text: str, status_url: str, quote_token: str | None
) -> list[dict[str, Any]]:
    """What an understanding job (/說圖 /說音 /說影) delivers: the description
    itself, as a single text message. There is no media object -- an
    understanding job produces text, not a file, so there is nothing to
    attach a caption to."""
    return [text_message(f"{result_text}\n{status_url}", quote_token=quote_token)]


def failed_messages(
    *, reason: str, status_url: str, prompt: str, quote_token: str | None
) -> list[dict[str, Any]]:
    """What a failed request says. More important than the success path.

    On success the user eventually sees something appear. On failure, silence
    is indistinguishable from a broken bot, and they will simply ask again —
    which costs another conversion and another GPU slot.
    """
    return [
        text_message(
            f"{prompt[:40]} 失敗了:{reason[:60]}\n{status_url}",
            quote_token=quote_token,
        )
    ]


async def deliver(
    client: PushClientLike,
    *,
    to: str,
    messages: list[dict[str, Any]],
    fallback_text: str,
    retry_key: str | None = None,
    quote_token: str | None = None,
) -> str:
    """Push `messages`, degrading to plain text rather than failing silently.

    Returns what happened, for the log and for the caller's own record.

    The two degradations are separated on purpose. Quota exhaustion is a
    *money* condition that will recur for the rest of the month, and the fall
    back to one text message is both cheaper and the only thing that still
    works. A rejected media object is a *bug* — an oversized poster, a
    mismatched aspect ratio, a host that would not answer a range request —
    and it will fail identically until someone fixes it, so it is logged at
    WARNING with LINE's own message rather than retried.

    What must never happen is either of them ending as silence: the user is
    waiting, and a delivery that fails without a word is indistinguishable
    from a bot that is broken.
    """
    try:
        await client.push(to, messages, retry_key=retry_key)
        return "pushed"
    except LineQuotaExhausted as exc:
        _log.warning("push quota exhausted, falling back to text: %s", exc)
        outcome = "quota-exhausted"
    except LineMediaRejected as exc:
        _log.warning("LINE refused the media object, falling back to text: %s", exc)
        outcome = "media-rejected"
    except LinePushError as exc:
        # A 500, an expired token, a network that came back mid-request. Worth
        # one cheaper attempt: a two-object media push and a single text
        # message are not equally likely to succeed, and if this one fails too
        # the caller finds out, which is the part that matters.
        _log.warning("push failed, falling back to text: %s", exc)
        outcome = "push-failed"

    try:
        await client.push(
            to,
            [text_message(fallback_text, quote_token=quote_token)],
            retry_key=f"{retry_key}-text" if retry_key else None,
        )
    except LinePushError as exc:
        _log.error("fallback text push also failed; the user was told nothing: %s", exc)
        return f"{outcome}-and-silent"
    return f"{outcome}-text-only"
