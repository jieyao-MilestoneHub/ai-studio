"""LINE Content API client: download a message's attached media.

A separate host from the rest of the Messaging API (`api-data.line.me`, not
`api.line.me`) -- LINE splits binary content onto its own domain, and hitting
the wrong one gets a 404 rather than a redirect. And it is `.me`, not `.biz`:
the docs live on developers.line.biz, the API does not, and `api-data.line.biz`
does not resolve at all. That typo shipped once and every photo sent to the
bot was dropped with "Name or service not known" -- hence the test that pins
the host.

Reference: https://developers.line.biz/en/reference/messaging-api/#get-content
"""

from __future__ import annotations

from typing import Protocol

import httpx

CONTENT_ENDPOINT = "https://api-data.line.me/v2/bot/message/{message_id}/content"


class LineContentError(Exception):
    """The Content API refused, or the message has no content to fetch."""


class ContentClient(Protocol):
    async def fetch(self, message_id: str) -> bytes: ...


class LineContentClient:
    """Downloads the bytes behind an image (or other media) message."""

    def __init__(self, access_token: str, *, timeout_s: float = 10.0) -> None:
        self._token = access_token
        self._timeout = timeout_s

    async def fetch(self, message_id: str) -> bytes:
        url = CONTENT_ENDPOINT.format(message_id=message_id)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(
                    url, headers={"Authorization": f"Bearer {self._token}"}
                )
            except httpx.HTTPError as exc:
                raise LineContentError(f"could not reach the LINE Content API: {exc}") from exc

        if response.status_code >= 400:
            raise LineContentError(
                f"content fetch rejected ({response.status_code}): {response.text[:300]}"
            )
        return response.content


class NullContentClient:
    """Returns canned bytes. For tests and for running without credentials."""

    def __init__(self, *replies: bytes) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    async def fetch(self, message_id: str) -> bytes:
        self.calls.append(message_id)
        if not self._replies:
            raise LineContentError("no scripted content left")
        return self._replies.pop(0)
