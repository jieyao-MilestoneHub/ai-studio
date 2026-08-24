"""LINE webhook handling: verify, filter, enqueue, acknowledge.

**LINE requires a 200 within two seconds.** Everything on this path is therefore
bounded and local: an HMAC, a few string comparisons, one SQLite insert, and one
HTTP reply. The LLM conversion and the GPU render both happen after the response
has gone out — they are not, and must never become, part of this function.

Deliberately framework-agnostic. `handle()` takes raw bytes and returns
decisions, so the two-second path can be tested without starting a web server,
and so the FastAPI route stays a thin adapter.

Reference: https://developers.line.biz/en/docs/messaging-api/receiving-messages/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from videogen.bots.line.verify import verify
from videogen.pipeline.queue import Job, JobQueue, JobState

DEFAULT_TRIGGER = "生成"
"""Messages must start with this. Everything else is ignored *silently* — no
reply at all — so the bot is invisible during ordinary group conversation."""

STATUS_WORDS = ("好了嗎", "好了没", "進度", "status", "查詢")
"""A free way to ask again. Replies are not billed, so a user can poll by
asking as often as they like and it costs nothing."""


class Replier(Protocol):
    async def reply_text(self, reply_token: str, *texts: str) -> None: ...


class InvalidSignature(Exception):
    """The request is not from LINE. The route maps this to 400."""


@dataclass(frozen=True)
class Outcome:
    """What the handler did with one event, for logging and for tests."""

    action: str  # accepted | duplicate | status | ignored | standby | wrong_group
    job: Job | None = None
    detail: str = ""


class WebhookHandler:
    def __init__(
        self,
        queue: JobQueue,
        replier: Replier,
        *,
        channel_secret: str,
        allowed_group_id: str | None,
        base_url: str,
        trigger: str = DEFAULT_TRIGGER,
    ) -> None:
        self.queue = queue
        self.replier = replier
        self.channel_secret = channel_secret
        self.allowed_group_id = allowed_group_id
        self.base_url = base_url.rstrip("/")
        self.trigger = trigger

    # --------------------------------------------------------------- entry

    async def handle(self, body: bytes, signature: str | None) -> list[Outcome]:
        """Verify and process a webhook delivery. Raises `InvalidSignature`.

        Returns one outcome per event. An empty list is a perfectly normal
        result — LINE's "Verify" button and its periodic connectivity checks
        both send `{"destination": ..., "events": []}`, and a handler that
        indexes `events[0]` fails verification.
        """
        if not verify(body, signature, self.channel_secret):
            raise InvalidSignature("x-line-signature did not match the request body")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidSignature(f"body was signed but is not JSON: {exc}") from exc

        events = payload.get("events") or []
        return [await self._event(e) for e in events if isinstance(e, dict)]

    # --------------------------------------------------------------- events

    async def _event(self, event: dict[str, Any]) -> Outcome:
        # A standby event means another channel owns the conversation. LINE is
        # explicit that a bot must not send anything in that mode.
        if event.get("mode") == "standby":
            return Outcome("standby")

        if event.get("type") != "message":
            return Outcome("ignored", detail=f"event type {event.get('type')}")

        message = event.get("message") or {}
        if message.get("type") != "text":
            return Outcome("ignored", detail=f"message type {message.get('type')}")

        source = event.get("source") or {}
        group_id = source.get("groupId")
        text = str(message.get("text") or "").strip()
        reply_token = event.get("replyToken") or ""

        # Capture mode. There is no API to list the groups an account belongs to
        # — LINE says so explicitly — so the only way to learn a groupId is to
        # read it off an event. Until one is configured this bot reports the id
        # and accepts nothing: an unset allowlist must not mean "serve everyone".
        if not self.allowed_group_id:
            return await self._capture(source, text, reply_token)

        # Allowlist before anything that costs. This is what stops the bot
        # working for any group that happens to add it.
        if group_id != self.allowed_group_id:
            return Outcome("wrong_group", detail=str(group_id))

        if self._is_status_query(text):
            return await self._status(group_id or "", reply_token)

        prompt = self._strip_trigger(text)
        if prompt is None:
            return Outcome("ignored", detail="no trigger word")
        if not prompt:
            await self._safe_reply(reply_token, f"用法:{self.trigger} <想看的畫面>")
            return Outcome("ignored", detail="empty prompt")

        return await self._accept(event, group_id or "", source.get("userId"), prompt, reply_token)

    # -------------------------------------------------------------- actions

    async def _accept(
        self,
        event: dict[str, Any],
        group_id: str,
        user_id: str | None,
        prompt: str,
        reply_token: str,
    ) -> Outcome:
        # webhookEventId is LINE's own idempotency key. Using it means a
        # redelivery cannot enqueue — and pay for — the same clip twice.
        event_id = str(event.get("webhookEventId") or f"{group_id}:{event.get('timestamp')}")
        job, created = self.queue.enqueue(event_id, group_id, prompt, user_id=user_id)

        if not created:
            await self._safe_reply(reply_token, self._status_line(job))
            return Outcome("duplicate", job=job)

        position = self.queue.position(job.token) or 1
        await self._safe_reply(
            reply_token,
            f"收到 ✓ 排隊第 {position} 位,正在解析你的描述\n"
            f"進度與下載 → {self._link(job)}",
        )
        return Outcome("accepted", job=job)

    async def _capture(
        self, source: dict[str, Any], text: str, reply_token: str
    ) -> Outcome:
        """Report the chat's own id so it can be put in LINE_ALLOWED_GROUP_ID.

        Only answers a trigger or status word, so a group is not spammed while
        someone is setting the bot up. Never enqueues: capture mode is
        deliberately inert beyond telling you where you are.
        """
        if self._strip_trigger(text) is None and not self._is_status_query(text):
            return Outcome("ignored", detail="capture mode, no trigger word")

        group_id = source.get("groupId")
        if not group_id:
            await self._safe_reply(
                reply_token,
                "這個 bot 只在群組裡運作。請把它加進群組後再試一次。",
            )
            return Outcome("capture", detail="not a group")

        await self._safe_reply(
            reply_token,
            "尚未設定服務的群組。這個群組的 ID 是:\n"
            f"{group_id}\n"
            "把它填進 .env 的 LINE_ALLOWED_GROUP_ID 再重啟服務即可開始接單。",
        )
        return Outcome("capture", detail=str(group_id))

    async def _status(self, group_id: str, reply_token: str) -> Outcome:
        """Answer 'is it done yet' from the queue. Free, so ask as often as you like."""
        pending = [j for j in self.queue.pending() if j.group_id == group_id]
        recent_done = [
            j for j in self.queue.recent(10) if j.group_id == group_id and j.state is JobState.DONE
        ]

        if not pending and not recent_done:
            await self._safe_reply(reply_token, f"目前沒有排隊中的工作。用「{self.trigger} …」開始。")
            return Outcome("status")

        lines = [f"排隊中 {len(pending)} 件"]
        lines += [f"  · {j.text[:18]} — {self._state_zh(j)}" for j in pending[:5]]
        lines += [f"  ✓ {j.text[:18]} → {self._link(j)}" for j in recent_done[:3]]
        await self._safe_reply(reply_token, "\n".join(lines))
        return Outcome("status")

    # -------------------------------------------------------------- helpers

    def _strip_trigger(self, text: str) -> str | None:
        """The prompt after the trigger word, or None if it is not a request."""
        for prefix in (self.trigger, f"/{self.trigger}", "/gen"):
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
        return None

    def _is_status_query(self, text: str) -> bool:
        return any(text.startswith(w) for w in STATUS_WORDS)

    def _link(self, job: Job) -> str:
        return f"{self.base_url}/q/{job.token}"

    def _status_line(self, job: Job) -> str:
        return f"這個請求已經在處理中({self._state_zh(job)})\n{self._link(job)}"

    def _state_zh(self, job: Job) -> str:
        return {
            JobState.QUEUED: "解析中",
            JobState.PARSED: "等待生成",
            JobState.RUNNING: "生成中",
            JobState.DONE: "完成",
            JobState.FAILED: f"失敗:{(job.error or '')[:60]}",
        }[job.state]

    async def _safe_reply(self, reply_token: str, text: str) -> None:
        """Reply, swallowing failures.

        A reply that does not go out is a cosmetic loss; a webhook that returns
        non-2xx makes LINE redeliver the whole event, and repeated failures make
        LINE suspend delivery to this bot. The 200 matters more than the message.
        """
        if not reply_token:
            return
        try:
            await self.replier.reply_text(reply_token, text)
        except Exception:  # see docstring: the 200 matters more
            return
