"""LINE Reply API client.

Reply is the **only** send method LINE does not count against the monthly
message quota — push, multicast, broadcast and narrowcast all do, and they are
counted *per recipient*, so one push into a twenty-person group costs twenty
messages. Delivering results as a link in a free reply is what makes this bot
cost nothing to run at any volume.

The constraint that shapes everything: a reply token is **single-use and valid
for about one minute**. It cannot carry the finished video, because the video
does not exist for another few minutes at best and possibly not until tomorrow's
window. So the reply carries a link to a status page, and the video is never
pushed at all.

Reference:
https://developers.line.biz/en/docs/messaging-api/pricing/
https://developers.line.biz/en/reference/messaging-api/#send-reply-message
"""

from __future__ import annotations

import httpx

REPLY_ENDPOINT = "https://api.line.me/v2/bot/message/reply"

MAX_TEXT_CHARS = 5000
"""LINE's limit for a text message object."""

MAX_MESSAGES_PER_REPLY = 5


class LineReplyError(Exception):
    """The Reply API refused. Carries LINE's own message, which is specific."""


class LineReplyClient:
    """Sends free reply messages."""

    def __init__(self, access_token: str, *, timeout_s: float = 5.0) -> None:
        self._token = access_token
        self._timeout = timeout_s

    async def reply_text(self, reply_token: str, *texts: str) -> None:
        """Reply with one or more text messages.

        Kept to text on purpose: a link in text has none of the constraints a
        video message object carries (mp4 only, <=200MB, HTTPS with TLS 1.2+,
        a <=1MB poster at a matching aspect ratio, and a host that supports HTTP
        range requests). Dropping those is most of why delivery is a link.
        """
        if not texts:
            return
        if len(texts) > MAX_MESSAGES_PER_REPLY:
            raise LineReplyError(f"at most {MAX_MESSAGES_PER_REPLY} messages per reply")

        messages = [{"type": "text", "text": t[:MAX_TEXT_CHARS]} for t in texts]
        payload = {"replyToken": reply_token, "messages": messages}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    REPLY_ENDPOINT,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._token}"},
                )
            except httpx.HTTPError as exc:
                raise LineReplyError(f"could not reach the LINE Reply API: {exc}") from exc

        if response.status_code >= 400:
            raise LineReplyError(
                f"reply rejected ({response.status_code}): {response.text[:500]}"
            )


class NullReplyClient:
    """A no-op client for tests and for running without LINE credentials."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, tuple[str, ...]]] = []

    async def reply_text(self, reply_token: str, *texts: str) -> None:
        self.sent.append((reply_token, texts))
