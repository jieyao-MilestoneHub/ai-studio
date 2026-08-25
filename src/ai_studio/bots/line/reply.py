"""LINE Reply API client — the acknowledgement, not the delivery.

Reply is the **only** send method LINE does not count against the monthly
message quota — push, multicast, broadcast and narrowcast all do, and they are
counted *per recipient*, so one push into a twenty-person group costs twenty
messages.

The constraint that shapes this module: a reply token is **single-use and valid
for about one minute**. It cannot carry the finished video, because the video
does not exist for another few minutes at best and possibly not until the next
window. So a reply is always and only the immediate "got it, you are number
three in line" — free, instant, and worth sending for every request.

**What changed:** this file used to end by concluding that the video is
therefore *never pushed at all*, and that delivery is a link to a status page.
That conclusion has been reversed, deliberately and with the cost understood —
see `push.py`. The short version: on price the link was the better answer and
still is, but the product is a clip that lands in the group where somebody
asked for it, addressed to them, and a link to a status page is a receipt for
a thing that happened somewhere else. The quota is now something managed (one
private group; explicit 429 handling that degrades to text rather than to
silence) rather than something avoided.

So: **reply acknowledges, push delivers.** Both exist, and neither does the
other's job.

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

        Text only, and not because the media constraints are hard — `push.py`
        meets them now. Because there is nothing to send yet: this fires within
        two seconds of the request, and the clip is minutes away at best. A
        media message here would have no media.
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
