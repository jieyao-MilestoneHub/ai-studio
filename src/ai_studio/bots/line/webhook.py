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
import logging
import re
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ai_studio.bots.line.content import ContentClient, LineContentError
from ai_studio.bots.line.verify import verify
from ai_studio.core.enums import MediaKind
from ai_studio.pipeline.queue import Job, JobQueue, JobState

_log = logging.getLogger("ai_studio.webhook")

DEFAULT_TRIGGER = "/影片"
"""Text-to-video. Messages must start with this — exactly this, no bare
Chinese word and no English alias. Everything else is ignored *silently*, no
reply at all, so the bot is invisible during ordinary group conversation.

One spelling per trigger on purpose: 生成 used to work bare, and a bare word
that is also ordinary Chinese ("生成" appears in normal sentences) is a
request nobody meant, paid for in GPU-minutes. A leading slash is not a word
anyone types by accident."""

DEFAULT_IMAGE_TRIGGER = "/圖片"
"""Text-to-image (Flux). Same one-spelling, silent-ignore rule."""

DEFAULT_I2V_TRIGGER = "/圖影"
"""Image-to-video: the photo this sender posted within `IMAGE_PAIRING_TTL_S`
becomes the H3 first frame."""

DEFAULT_I2I_TRIGGER = "/圖圖"
"""Image-to-image: that same cached photo is re-rendered by Flux under the
prompt.

`/圖影` and `/圖圖` are the ONLY triggers that claim a cached photo. `/影片`
after a photo is still text-to-video and `/圖片` is still text-to-image; the
photo stays cached for the photo-trigger that follows. A photo must never be
silently consumed by a request that did not ask for it, and a request that
did ask must never silently run without one — it gets a reply saying what to
do instead."""

IMAGE_PAIRING_TTL_S = 300.0
"""How long a photo waits for the /圖影 that turns it into a first frame.

Five minutes: long enough for someone to send a photo, think for a moment,
and type a description, short enough that a photo from an unrelated part of
the conversation an hour ago cannot surface as a first frame nobody meant."""

_LENGTH_RE = re.compile(r"\s*(\d{1,2})\s*(?:s|秒)", re.IGNORECASE)
"""Matches a `15s` / `15秒` length flush against a video trigger."""

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
        i2v_trigger: str = DEFAULT_I2V_TRIGGER,
        i2i_trigger: str = DEFAULT_I2I_TRIGGER,
        max_jobs_per_user_per_day: int = 0,
        clock: Callable[[], datetime] | None = None,
        content: ContentClient | None = None,
        incoming_dir: Path | str = Path("incoming"),
        is_warm: Callable[[], bool] | None = None,
    ) -> None:
        self.queue = queue
        self.replier = replier
        self.channel_secret = channel_secret
        self.allowed_group_id = allowed_group_id
        self.allowed_user_ids = frozenset(allowed_user_ids)
        self.base_url = base_url.rstrip("/")
        self.trigger = trigger
        self.image_trigger = image_trigger
        self.i2v_trigger = i2v_trigger
        self.i2i_trigger = i2i_trigger
        self.max_jobs_per_user_per_day = max_jobs_per_user_per_day
        # Injected so a test can stand at 03:00 without the machine having to.
        # `bots` is L6 and `runtime` is L5, so reaching down for the business
        # calendar is allowed; reaching back up never is.
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        # None (not NullContentClient) is the real "image-to-video is off"
        # state: a channel access token is required to fetch LINE's own
        # content, and without one there is nothing that could be fetched.
        self.content = content
        self.incoming_dir = Path(incoming_dir)
        # Whether a pod is already open, so the acknowledgement can quote the
        # right wait: a warm pod renders an image in ~30 s, a cold one has to
        # boot first. Injected; the default assumes cold, which is the
        # honest default when nobody told us otherwise.
        self.is_warm = is_warm or (lambda: False)
        # (group_id, user_id) -> (saved path, received-at). In-process and
        # short-lived on purpose: this is a "send a photo, then say /圖影"
        # pairing window, not durable state -- a restart within the window
        # just means resending the photo, which costs nothing.
        self._pending_images: dict[tuple[str, str], tuple[str, float]] = {}

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
        if message.get("type") == "image":
            return await self._remember_image(event, message)
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
        prompt, media_kind, wants_photo, requested_seconds = stripped
        if not prompt:
            await self._safe_reply(reply_token, self._usage(media_kind, wants_photo))
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
                "達到每日上限。明天再試。",
            )
            return Outcome("rate_limited", detail=str(user_id))

        # Only the photo-triggers touch the photo cache. Popping it for any
        # request would let /影片 or /圖片 silently eat a photo meant for the
        # /圖影 or /圖圖 after them. And a photo-trigger with nothing cached is
        # told so rather than quietly rendered from text: the user asked for
        # their picture, and a picture of something else is not that.
        first_frame_path: str | None = None
        if wants_photo:
            first_frame_path = self._take_pending_image(group_id or "", user_id)
            if first_frame_path is None:
                trigger = self._photo_trigger(media_kind)
                await self._safe_reply(
                    reply_token,
                    f"找不到你的照片。先傳一張照片到群組,再說「{trigger} <想看的畫面>」"
                    f"(照片 {int(IMAGE_PAIRING_TTL_S // 60)} 分鐘內有效)。",
                )
                return Outcome("ignored", detail="no pending photo")

        return await self._accept(
            event,
            group_id or "",
            user_id,
            prompt,
            reply_token,
            media_kind=media_kind,
            first_frame_path=first_frame_path,
            requested_seconds=requested_seconds,
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

    async def _remember_image(self, event: dict[str, Any], message: dict[str, Any]) -> Outcome:
        """Download and cache a photo, waiting to see if a /圖影 claims it.

        Deliberately silent either way -- someone sharing a photo in ordinary
        group conversation must not get a bot reply for it, the same rule
        `DEFAULT_TRIGGER` states for any other message the bot does not act
        on. A photo is only ever *evidence of intent* once the next message is
        the trigger word; on its own it is just a photo in a group chat.
        """
        source = event.get("source") or {}
        group_id = source.get("groupId")
        user_id = source.get("userId")
        if self.content is None or not group_id or not user_id:
            return Outcome("ignored", detail="image, no content client or no user")
        if self.allowed_group_id and group_id != self.allowed_group_id:
            return Outcome("ignored", detail=f"image in {group_id}")

        message_id = str(message.get("id") or "")
        try:
            data = await self.content.fetch(message_id)
        except LineContentError as exc:
            _log.warning("could not fetch image %s: %s", message_id, exc)
            return Outcome("ignored", detail=f"image fetch failed: {exc}")

        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        dest = self.incoming_dir / f"{secrets.token_urlsafe(8)}.jpg"
        dest.write_bytes(data)
        self._pending_images[(group_id, user_id)] = (str(dest), self.clock().timestamp())
        return Outcome("image", detail=str(dest))

    def _take_pending_image(self, group_id: str, user_id: str | None) -> str | None:
        """Pop a still-fresh cached photo for this sender, if there is one.

        Popped rather than peeked: a photo is a first frame for exactly one
        request, not a standing instruction applied to everything the sender
        says next.
        """
        if user_id is None:
            return None
        key = (group_id, user_id)
        cached = self._pending_images.pop(key, None)
        if cached is None:
            return None
        path, received_at = cached
        if self.clock().timestamp() - received_at > IMAGE_PAIRING_TTL_S:
            return None
        return path

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
        first_frame_path: str | None = None,
        requested_seconds: float | None = None,
    ) -> Outcome:
        # webhookEventId is LINE's own idempotency key. Using it means a
        # redelivery cannot enqueue — and pay for — the same clip twice.
        event_id = str(event.get("webhookEventId") or f"{group_id}:{event.get('timestamp')}")
        # The request's quote token rides along so the finished media can be
        # delivered as a *reply to that message* -- the quoted-message card a
        # person gets when someone answers them -- rather than an @-mention.
        quote_token = (event.get("message") or {}).get("quoteToken") or None
        job, created = self.queue.enqueue(
            event_id, group_id, prompt, user_id=user_id, media_kind=media_kind,
            first_frame_path=first_frame_path, quote_token=quote_token,
            requested_seconds=requested_seconds,
        )

        if not created:
            await self._safe_reply(reply_token, self._status_line(job))
            return Outcome("duplicate", job=job)

        await self._safe_reply(reply_token, self._accepted_line(job))
        return Outcome("accepted", job=job)

    def _accepted_line(self, job: Job) -> str:
        """What a newly accepted request is told: an honest wait.

        The wait is the pod's state, not the clock. Warm: the model is in
        VRAM and an image is ~30 s, a clip ~2 min. Cold: the pod has to be
        created and ComfyUI restarted from the network volume first, which
        adds a couple of minutes. Both figures are 📏 from 2026-08-26 on an
        RTX 4090 and rounded up, so the bot under-promises.
        """
        eta = "圖約 30 秒、影片約 2 分鐘" if self.is_warm() else "暖機中:圖約 3 分鐘、影片約 5 分鐘"
        return f"收到,{eta},想查進度可以看{self._link(job)}"

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
                f"「{self.image_trigger} …」生成圖片,"
                f"或先傳照片再「{self.i2v_trigger} …」讓照片動起來、"
                f"「{self.i2i_trigger} …」重畫照片。",
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

    def _strip_trigger(self, text: str) -> tuple[str, MediaKind, bool, float | None] | None:
        """The prompt after the trigger, which kind it asked for, whether it
        wants the cached photo as a first frame, and any requested length in
        seconds — or None if it is not a request at all. Exactly one spelling
        per trigger, no aliases.

        A length rides on the trigger itself: `/影片15s` or `/圖影15秒`. It is
        read only off video triggers (an image has no length), and only when
        it sits flush against the trigger, so an ordinary prompt that happens
        to start with a number is untouched. The value is clamped to the
        model's range at conversion; here it is just the number the user typed.
        """
        for prefix, kind, wants_photo in (
            (self.i2v_trigger, MediaKind.VIDEO, True),
            (self.i2i_trigger, MediaKind.IMAGE, True),
            (self.trigger, MediaKind.VIDEO, False),
            (self.image_trigger, MediaKind.IMAGE, False),
        ):
            if text.startswith(prefix):
                rest = text[len(prefix) :]
                seconds: float | None = None
                if kind is MediaKind.VIDEO:
                    match = _LENGTH_RE.match(rest)
                    if match:
                        seconds = float(match.group(1))
                        rest = rest[match.end() :]
                return rest.strip(), kind, wants_photo, seconds
        return None

    def _photo_trigger(self, media_kind: MediaKind) -> str:
        return self.i2v_trigger if media_kind is MediaKind.VIDEO else self.i2i_trigger

    def _usage(self, media_kind: MediaKind, wants_photo: bool) -> str:
        if wants_photo:
            return f"用法:先傳一張照片,再說 {self._photo_trigger(media_kind)} <想看的畫面>"
        trigger = self.trigger if media_kind is MediaKind.VIDEO else self.image_trigger
        return f"用法:{trigger} <想看的畫面>"

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

        Logged rather than truly silent: swallowing the exception is right --
        it must never turn into a non-2xx -- but swallowing it with no trace at
        all meant a user reporting "no reply" was previously undiagnosable even
        after the fact, from our own systems, with no way to tell an expired
        reply token from a LINE outage from a real bug.
        """
        if not reply_token:
            return
        try:
            await self.replier.reply_text(reply_token, text)
        except Exception as exc:  # see docstring: the 200 matters more
            _log.warning("reply failed for token %s: %s", reply_token, exc)
            return
