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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from ai_studio.bots.line.verify import verify
from ai_studio.core.enums import MediaKind
from ai_studio.pipeline.queue import Job, JobQueue, JobState
from ai_studio.runtime import hours

DEFAULT_TRIGGER = "生成"
"""Messages must start with this. Everything else is ignored *silently* — no
reply at all — so the bot is invisible during ordinary group conversation."""

DEFAULT_IMAGE_TRIGGER = "畫圖"
"""The image-generation counterpart to `DEFAULT_TRIGGER`. Same silent-ignore
rule applies — an unmatched message gets no reply at all."""

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

    action: str  # accepted | duplicate | status | ignored | standby | capture
    #             | wrong_group | wrong_user | rate_limited
    #             | memberJoined | memberLeft
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
        allowed_user_ids: Iterable[str] = (),
        trigger: str = DEFAULT_TRIGGER,
        image_trigger: str = DEFAULT_IMAGE_TRIGGER,
        max_jobs_per_user_per_day: int = 0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.queue = queue
        self.replier = replier
        self.channel_secret = channel_secret
        self.allowed_group_id = allowed_group_id
        self.allowed_user_ids = frozenset(allowed_user_ids)
        self.base_url = base_url.rstrip("/")
        self.trigger = trigger
        self.image_trigger = image_trigger
        self.max_jobs_per_user_per_day = max_jobs_per_user_per_day
        # Injected so a test can stand at 03:00 without the machine having to.
        # `bots` is L6 and `runtime` is L5, so reaching down for the business
        # calendar is allowed; reaching back up never is.
        self.clock = clock or (lambda: datetime.now(timezone.utc))

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

        # Who is in the group is who can spend money, so a membership change
        # is not something to drop on the floor with the other event types.
        if event.get("type") in ("memberJoined", "memberLeft"):
            return self._membership(event)

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

        stripped = self._strip_trigger(text)
        if stripped is None:
            return Outcome("ignored", detail="no trigger word")
        prompt, media_kind = stripped
        if not prompt:
            trigger = self.trigger if media_kind is MediaKind.VIDEO else self.image_trigger
            await self._safe_reply(reply_token, f"用法:{trigger} <想看的畫面>")
            return Outcome("ignored", detail="empty prompt")

        user_id = source.get("userId")
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            # Fail closed. LINE omits `source.userId` for a user who has not
            # accepted the Official Account terms, so "we could not tell who
            # this was" arrives as None -- and it must not resolve to "let them
            # spend a GPU-hour". The id is put in the detail so the log line is
            # enough to add someone deliberately.
            await self._safe_reply(reply_token, "這個 BOT 只回應已授權的成員。")
            return Outcome("wrong_user", detail=str(user_id))

        # Before the insert, deliberately. Accepting and then refusing would
        # still have spent an LLM conversion on a request that never runs.
        if self._over_daily_cap(user_id):
            await self._safe_reply(
                reply_token,
                f"你今天已經送出 {self.max_jobs_per_user_per_day} 個請求,"
                "達到每日上限。明天 11:00 之後再試。",
            )
            return Outcome("rate_limited", detail=str(user_id))

        return await self._accept(
            event, group_id or "", user_id, prompt, reply_token, media_kind=media_kind
        )

    def _membership(self, event: dict[str, Any]) -> Outcome:
        """Report a membership change. Deliberately does not reply.

        A join is the moment the set of people who can spend GPU time changes,
        and nothing here polls the roster -- so this event is the only notice a
        change produces. If the user allowlist is empty the newcomer can trigger
        a render immediately, which is why the report is worth a log line even
        though there is nothing to reply to.
        """
        kind = str(event.get("type"))
        # Any account can add this bot to any group. Only the roster of the
        # group we actually serve is worth a line in the log.
        group_id = (event.get("source") or {}).get("groupId")
        if self.allowed_group_id and group_id != self.allowed_group_id:
            return Outcome("ignored", detail=f"{kind} in {group_id}")

        members = (event.get("joined") or event.get("left") or {}).get("members") or []
        ids = [str(m.get("userId")) for m in members if isinstance(m, dict)]
        return Outcome(kind, detail=" ".join(ids) or "unknown")

    # -------------------------------------------------------------- actions

    async def _accept(
        self,
        event: dict[str, Any],
        group_id: str,
        user_id: str | None,
        prompt: str,
        reply_token: str,
        *,
        media_kind: MediaKind = MediaKind.VIDEO,
    ) -> Outcome:
        # webhookEventId is LINE's own idempotency key. Using it means a
        # redelivery cannot enqueue — and pay for — the same clip twice.
        event_id = str(event.get("webhookEventId") or f"{group_id}:{event.get('timestamp')}")
        job, created = self.queue.enqueue(
            event_id, group_id, prompt, user_id=user_id, media_kind=media_kind
        )

        if not created:
            await self._safe_reply(reply_token, self._status_line(job))
            return Outcome("duplicate", job=job)

        position = self.queue.position(job.token) or 1
        await self._safe_reply(reply_token, self._accepted_line(job, position))
        return Outcome("accepted", job=job)

    def _accepted_line(self, job: Job, position: int) -> str:
        """What a newly accepted request is told.

        Out of hours the request is still accepted — it waits in the queue for
        the next window. Refusing it instead would mean whatever someone
        thought of at midnight is simply lost, which is a worse outcome than
        waiting until eleven. What changes is only what they are told: the
        place in the line is true either way, but on its own at 03:00 it reads
        as "shortly" and is off by eight hours, so out of hours it is followed
        by when "shortly" actually is.
        """
        head = f"收到 ✓ 排隊第 {position} 位"
        link = f"進度與下載 → {self._link(job)}"
        if hours.is_open(self.clock()):
            return f"{head},正在解析你的描述\n{link}"
        opens = hours.next_open(self.clock()).astimezone(hours.TZ)
        return (
            f"{head}\n"
            f"營業時間 {hours.OPEN_LOCAL:%H:%M}-{hours.CLOSE_LOCAL:%H:%M},"
            f"已排入下一個時段(約 {opens:%m/%d %H:%M}),完成後會在群組通知你\n"
            f"{link}"
        )

    def _over_daily_cap(self, user_id: str | None) -> bool:
        """Has this user used up today's allowance?

        Skipped entirely when the cap is 0 (off) or the id is unknown — LINE
        omits `source.userId` for a user who has not accepted the Official
        Account terms, and there is no per-user budget to enforce against a
        user we cannot name. The group and user allowlists are what stand
        between that case and an open bar.
        """
        if not self.max_jobs_per_user_per_day or not user_id:
            return False
        return self.queue.accepted_today(user_id) >= self.max_jobs_per_user_per_day

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
            await self._safe_reply(
                reply_token,
                f"目前沒有排隊中的工作。用「{self.trigger} …」生成影片,"
                f"或「{self.image_trigger} …」生成圖片。",
            )
            return Outcome("status")

        lines = [f"排隊中 {len(pending)} 件"]
        lines += [
            f"  · {'🖼' if j.media_kind is MediaKind.IMAGE else '🎬'} "
            f"{j.text[:18]} — {self._state_zh(j)}"
            for j in pending[:5]
        ]
        lines += [f"  ✓ {j.text[:18]} → {self._link(j)}" for j in recent_done[:3]]
        await self._safe_reply(reply_token, "\n".join(lines))
        return Outcome("status")

    # -------------------------------------------------------------- helpers

    def _strip_trigger(self, text: str) -> tuple[str, MediaKind] | None:
        """The prompt after the trigger word and which kind it asked for, or
        None if it is not a request at all."""
        for prefix in (self.trigger, f"/{self.trigger}", "/gen"):
            if text.startswith(prefix):
                return text[len(prefix) :].strip(), MediaKind.VIDEO
        for prefix in (self.image_trigger, f"/{self.image_trigger}", "/img"):
            if text.startswith(prefix):
                return text[len(prefix) :].strip(), MediaKind.IMAGE
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
