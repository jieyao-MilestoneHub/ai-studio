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
from typing import Any, ClassVar, Protocol

from fun_workflow.bots.line.content import ContentClient, LineContentError
from fun_workflow.bots.line.verify import verify
from fun_workflow.core.kinds import JobKind
from fun_workflow.pipeline.queue import Job, JobQueue, JobState

_log = logging.getLogger("fun_workflow.webhook")

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

`/圖影` and `/圖圖` are the ONLY generation triggers that claim a cached
photo. `/影片` after a photo is still text-to-video and `/圖片` is still
text-to-image; the photo stays cached for the photo-trigger that follows. A
photo must never be silently consumed by a request that did not ask for it,
and a request that did ask must never silently run without one — it gets a
reply saying what to do instead."""

DEFAULT_DESCRIBE_IMAGE_TRIGGER = "/說圖"
"""Describe a photo: understanding, the reverse of generation. Claims a
cached photo the same way /圖影 and /圖圖 do, but takes no trailing text --
none of the three describe triggers do, for consistency across all three
regardless of which underlying model could technically take a prompt."""

DEFAULT_DESCRIBE_AUDIO_TRIGGER = "/說音"
"""Describe/transcribe an audio clip. Backed by Qwen3-Omni-Captioner, which
rejects a text prompt outright -- one more reason none of the three describe
triggers accept one."""

DEFAULT_DESCRIBE_VIDEO_TRIGGER = "/說影"
"""Describe a video clip. Backed by Tarsier2."""

DEFAULT_CHAT_TRIGGER = "/himonkey"
DEFAULT_EXTRACT_AUDIO_TRIGGER = "/影音"
"""Video in, its audio track out as an M4A -- pure ffmpeg on the host, no
GPU, no pod, no queue, answered in the free reply. The first trigger that
spends no money; that is why it is handled inline here rather than enqueued
(the worker would open a $0.74/hr pod for an ffmpeg call)."""
DEFAULT_DRAMA_TRIGGER = "/短劇"
"""A one-minute, six-shot drama with one recurring lead, from a one-line
premise. No photo, no length suffix (the length is the six shots), and a
group-wide daily cap on top of the per-user one: a drama is 15-30
GPU-minutes, so `max_jobs_per_user_per_day` alone would let one afternoon
spend the month. See docs/drama.md."""
DEFAULT_SHOW_TRIGGER = "讓我看看"
"""The pull trigger: hand over finished results in a free *reply* instead of
a metered push. Exists for the month in which the push quota is gone -- the
worker then finishes renders nobody is told about. Since a reply token
only lives ~1 minute after the message, the user has to ask *after* the
work is done: they keep their own clock, and the bot says so on accept.

Two shapes. Sent as a *quote-reply* to an earlier message (the request, or
the bot's「收到」answer to it), it hands over exactly that one job -- or
says plainly that it is still running, failed, or is not a request at all.
Sent bare, it hands over everything finished and not yet delivered. A
leading "/" (or the IME's fullwidth solidus, U+FF0F) is tolerated like the status
words, not a second spelling."""
MAX_SHOW_JOBS = 4
"""LINE allows 5 messages per reply: up to four media objects plus one
summary text carrying the links and any text-only results."""
"""Plain-text LLM reply, backed by gpt-oss-20b on the shared pod. Same
one-spelling, no-aliases, silent-ignore-otherwise rule as every other
trigger. Unlike the four generation triggers it claims no cached photo, and
unlike the three describe triggers it *requires* trailing text -- there is
no attached media to fall back on."""

IMAGE_PAIRING_TTL_S = 300.0
"""How long a photo/audio/video clip waits for the trigger that claims it.

Five minutes: long enough for someone to send it, think for a moment, and
type a description (or, for the three describe triggers, just the trigger
word), short enough that media from an unrelated part of the conversation an
hour ago cannot surface as input nobody meant. Shared by all three pending
caches -- image, audio, video -- on purpose; nothing so far needs them to
diverge."""

MAX_AUDIO_UNDERSTAND_S = 30.0
"""Default ceiling for /說音: Qwen3-Omni-Captioner's own stated limit.
Overridable via the constructor so `config.settings` can drive it."""

MAX_VIDEO_UNDERSTAND_S = 120.0
"""[speculative] default ceiling for /說影 -- no source has measured what
Tarsier2 actually tolerates or costs per second of dense video understanding
on this hardware. Generous rather than tight until benchmarked."""

_LENGTH_RE = re.compile(r"\s*(\d{1,2})\s*(?:s|秒)", re.IGNORECASE)
"""Matches a `15s` / `15秒` length flush against a video trigger."""

STATUS_WORDS = ("好了嗎", "好了没", "進度", "status", "查詢")
"""A free way to ask again. Replies are not billed, so a user can poll by
asking as often as they like and it costs nothing."""


class Replier(Protocol):
    async def reply_text(self, reply_token: str, *texts: str) -> Any: ...

    async def reply_messages(self, reply_token: str, messages: list[dict[str, Any]]) -> Any: ...


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
        describe_image_trigger: str = DEFAULT_DESCRIBE_IMAGE_TRIGGER,
        describe_audio_trigger: str = DEFAULT_DESCRIBE_AUDIO_TRIGGER,
        describe_video_trigger: str = DEFAULT_DESCRIBE_VIDEO_TRIGGER,
        chat_trigger: str = DEFAULT_CHAT_TRIGGER,
        drama_trigger: str = DEFAULT_DRAMA_TRIGGER,
        show_trigger: str = DEFAULT_SHOW_TRIGGER,
        extract_audio_trigger: str = DEFAULT_EXTRACT_AUDIO_TRIGGER,
        files_dir: Path | None = None,
        max_audio_understand_s: float = MAX_AUDIO_UNDERSTAND_S,
        max_video_understand_s: float = MAX_VIDEO_UNDERSTAND_S,
        max_jobs_per_user_per_day: int = 0,
        max_chat_messages_per_user_per_day: int = 0,
        max_dramas_per_day: int = 0,
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
        self.describe_image_trigger = describe_image_trigger
        self.describe_audio_trigger = describe_audio_trigger
        self.describe_video_trigger = describe_video_trigger
        self.chat_trigger = chat_trigger
        self.drama_trigger = drama_trigger
        self.show_trigger = show_trigger
        self.extract_audio_trigger = extract_audio_trigger
        self.files_dir = Path(files_dir) if files_dir is not None else None
        self.max_audio_understand_s = max_audio_understand_s
        self.max_video_understand_s = max_video_understand_s
        self.max_jobs_per_user_per_day = max_jobs_per_user_per_day
        self.max_chat_messages_per_user_per_day = max_chat_messages_per_user_per_day
        self.max_dramas_per_day = max_dramas_per_day
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
        # just means resending the photo, which costs nothing. Same shape,
        # same reasoning, for an audio clip waiting on /說音 and a video clip
        # waiting on /說影.
        self._pending_images: dict[tuple[str, str], tuple[str, float]] = {}
        self._pending_audio: dict[tuple[str, str], tuple[str, float]] = {}
        self._pending_video: dict[tuple[str, str], tuple[str, float]] = {}

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
        msg_type = message.get("type")
        if msg_type == "image":
            return await self._remember_image(event, message)
        if msg_type == "audio":
            return await self._remember_audio(event, message)
        if msg_type == "video":
            return await self._remember_video(event, message)
        if msg_type != "text":
            return Outcome("ignored", detail=f"message type {msg_type}")

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
        if self._is_show_request(text):
            quoted = str(message.get("quotedMessageId") or "") or None
            return await self._show(group_id or "", reply_token, quoted_message_id=quoted)
        if self._is_extract_audio_request(text):
            return await self._extract_audio(group_id or "", source.get("userId"), reply_token)

        stripped = self._strip_trigger(text)
        if stripped is None:
            return Outcome("ignored", detail="no trigger word")
        prompt, media_kind, wants_media, requested_seconds = stripped

        # The describe triggers take *optional* text since 2026-08-27: bare,
        # the model gets its engineered default question; with text, that
        # question is rewritten into the model's best form on the pod
        # (prompts/understanding.py). Generation and chat still need words.
        if not media_kind.is_understanding and not prompt:
            await self._safe_reply(reply_token, self._usage(media_kind, wants_media))
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
        if self._over_daily_cap(user_id, media_kind):
            limit = (
                self.max_chat_messages_per_user_per_day
                if media_kind is JobKind.CHAT
                else self.max_jobs_per_user_per_day
            )
            await self._safe_reply(
                reply_token,
                f"你今天已經送出 {limit} 個請求,"
                "達到每日上限。明天再試。",
            )
            _log.warning("refused: daily cap", extra={"user": user_id, "kind": media_kind.value, "reason": "daily cap"})
            return Outcome("rate_limited", detail=str(user_id))

        if media_kind is JobKind.DRAMA and self._over_drama_cap():
            await self._safe_reply(
                reply_token,
                f"今天的短劇額度({self.max_dramas_per_day} 部)已經用完,明天再來。"
                f"其他功能({self.trigger}、{self.image_trigger}…)不受影響。",
            )
            _log.warning("refused: drama cap", extra={"user": user_id, "kind": media_kind.value, "reason": "drama cap"})
            return Outcome("rate_limited", detail="drama cap")

        # Only a trigger that asked for cached media touches a cache, and only
        # the matching one: /影片 or /圖片 must never silently eat a photo
        # meant for /圖影 or /圖圖, and /說音 must never claim a photo or video
        # clip meant for /說圖 or /說影. A trigger that did ask for media with
        # nothing cached is told so rather than silently running without it.
        first_frame_path: str | None = None
        input_media_path: str | None = None
        if wants_media is not None:
            path = self._take_pending(wants_media, group_id or "", user_id)
            if path is None:
                await self._safe_reply(reply_token, self._no_pending_media(media_kind, wants_media))
                return Outcome("ignored", detail=f"no pending {wants_media}")
            if media_kind.is_understanding:
                input_media_path = path
            else:
                first_frame_path = path

        return await self._accept(
            event,
            group_id or "",
            user_id,
            prompt,
            reply_token,
            media_kind=media_kind,
            first_frame_path=first_frame_path,
            requested_seconds=requested_seconds,
            input_media_path=input_media_path,
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

    async def _remember_audio(self, event: dict[str, Any], message: dict[str, Any]) -> Outcome:
        """Download and cache an audio clip, waiting to see if /說音 claims it.

        LINE's own `audio` message object carries `duration` (milliseconds)
        in the webhook payload itself, so an over-long clip is rejected
        *before any bandwidth is spent fetching it* -- not merely before any
        GPU is spent on it. Silently, matching `_remember_image`'s rule: an
        audio message shared in ordinary group conversation must not get a
        bot reply just for existing.
        """
        source = event.get("source") or {}
        group_id = source.get("groupId")
        user_id = source.get("userId")
        if self.content is None or not group_id or not user_id:
            return Outcome("ignored", detail="audio, no content client or no user")
        if self.allowed_group_id and group_id != self.allowed_group_id:
            return Outcome("ignored", detail=f"audio in {group_id}")

        duration_ms = message.get("duration")
        if isinstance(duration_ms, int | float) and duration_ms > self.max_audio_understand_s * 1000:
            return Outcome(
                "ignored",
                detail=f"audio too long for {self.describe_audio_trigger}: {duration_ms / 1000:.0f}s",
            )

        message_id = str(message.get("id") or "")
        try:
            data = await self.content.fetch(message_id)
        except LineContentError as exc:
            _log.warning("could not fetch audio %s: %s", message_id, exc)
            return Outcome("ignored", detail=f"audio fetch failed: {exc}")

        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        dest = self.incoming_dir / f"{secrets.token_urlsafe(8)}.m4a"
        dest.write_bytes(data)
        self._pending_audio[(group_id, user_id)] = (str(dest), self.clock().timestamp())
        return Outcome("audio", detail=str(dest))

    async def _remember_video(self, event: dict[str, Any], message: dict[str, Any]) -> Outcome:
        """Download and cache a video clip, waiting to see if /說影 claims it.
        See `_remember_audio` for why this is silent either way, and for why
        the length check runs before the fetch."""
        source = event.get("source") or {}
        group_id = source.get("groupId")
        user_id = source.get("userId")
        if self.content is None or not group_id or not user_id:
            return Outcome("ignored", detail="video, no content client or no user")
        if self.allowed_group_id and group_id != self.allowed_group_id:
            return Outcome("ignored", detail=f"video in {group_id}")

        duration_ms = message.get("duration")
        if isinstance(duration_ms, int | float) and duration_ms > self.max_video_understand_s * 1000:
            return Outcome(
                "ignored",
                detail=f"video too long for {self.describe_video_trigger}: {duration_ms / 1000:.0f}s",
            )

        message_id = str(message.get("id") or "")
        try:
            data = await self.content.fetch(message_id)
        except LineContentError as exc:
            _log.warning("could not fetch video %s: %s", message_id, exc)
            return Outcome("ignored", detail=f"video fetch failed: {exc}")

        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        dest = self.incoming_dir / f"{secrets.token_urlsafe(8)}.mp4"
        dest.write_bytes(data)
        self._pending_video[(group_id, user_id)] = (str(dest), self.clock().timestamp())
        return Outcome("video", detail=str(dest))

    def _take_pending_image(self, group_id: str, user_id: str | None) -> str | None:
        """Pop a still-fresh cached photo for this sender, if there is one.

        Popped rather than peeked: a photo is a first frame (or, for /說圖,
        the thing to describe) for exactly one request, not a standing
        instruction applied to everything the sender says next.
        """
        return self._pop_pending(self._pending_images, group_id, user_id)

    def _take_pending_audio(self, group_id: str, user_id: str | None) -> str | None:
        """Pop a still-fresh cached audio clip. See `_take_pending_image`."""
        return self._pop_pending(self._pending_audio, group_id, user_id)

    def _take_pending_video(self, group_id: str, user_id: str | None) -> str | None:
        """Pop a still-fresh cached video clip. See `_take_pending_image`."""
        return self._pop_pending(self._pending_video, group_id, user_id)

    def _take_pending(self, kind: str, group_id: str, user_id: str | None) -> str | None:
        """Dispatch to the cache matching `wants_media` ("image"/"audio"/"video")."""
        if kind == "image":
            return self._take_pending_image(group_id, user_id)
        if kind == "audio":
            return self._take_pending_audio(group_id, user_id)
        return self._take_pending_video(group_id, user_id)

    def _pop_pending(
        self, cache: dict[tuple[str, str], tuple[str, float]], group_id: str, user_id: str | None
    ) -> str | None:
        if user_id is None:
            return None
        cached = cache.pop((group_id, user_id), None)
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
        media_kind: JobKind = JobKind.VIDEO,
        first_frame_path: str | None = None,
        requested_seconds: float | None = None,
        input_media_path: str | None = None,
    ) -> Outcome:
        # webhookEventId is LINE's own idempotency key. Using it means a
        # redelivery cannot enqueue — and pay for — the same clip twice.
        event_id = str(event.get("webhookEventId") or f"{group_id}:{event.get('timestamp')}")
        # The request's quote token rides along so the finished media can be
        # delivered as a *reply to that message* -- the quoted-message card a
        # person gets when someone answers them -- rather than an @-mention.
        quote_token = (event.get("message") or {}).get("quoteToken") or None
        message_id = str((event.get("message") or {}).get("id") or "") or None
        job, created = self.queue.enqueue(
            event_id, group_id, prompt, user_id=user_id, media_kind=media_kind,
            first_frame_path=first_frame_path, quote_token=quote_token,
            requested_seconds=requested_seconds, input_media_path=input_media_path,
            message_id=message_id,
        )

        if not created:
            await self._safe_reply(reply_token, self._status_line(job))
            return Outcome("duplicate", job=job)

        sent = await self._safe_reply(reply_token, self._accepted_line(job))
        if sent:
            # So quoting the bot's own「收到」names this request too.
            self.queue.set_reply_message_id(job.id, sent[0])
        _log.info(
            "accepted", extra={"job_id": job.id, "token": job.token, "kind": media_kind.value,
                               "user": user_id, "message_id": message_id, "stage": "accept"},
        )
        return Outcome("accepted", job=job)

    def _accepted_line(self, job: Job) -> str:
        """What a newly accepted request is told: an honest wait.

        The wait is the pod's state, not the clock. Warm: the model is in
        VRAM and an image is ~30 s, a clip ~2 min. Cold: the pod has to be
        created and ComfyUI restarted from the network volume first, which
        adds a couple of minutes. Both figures are 📏 from 2026-08-26 on an
        RTX 4090 and rounded up, so the bot under-promises.

        Chat gets its own honest wait rather than reusing the generation
        line: a warm gpt-oss-20b reply is fast, but nothing here has
        measured a cold load yet, and pretending chat is always instant
        would be the same silent optimism this method exists to avoid.
        """
        if job.media_kind is JobKind.CHAT:
            eta = "已經在線上,幾秒內回覆" if self.is_warm() else "暖機中,第一句回覆可能要等幾分鐘"
        elif job.media_kind is JobKind.DRAMA:
            # [speculative] until the first real drama: 3 screenwriter calls,
            # 8 Flux stills, 6 H3 clips, then a CPU concat -- see docs/drama.md.
            eta = "短劇要慢慢做,約 25 到 40 分鐘" + ("" if self.is_warm() else "(含暖機)") + ",完成後會貼到群組"
        elif job.media_kind.is_understanding:
            # 📏 2026-08-27: 27-65 s to load the model, seconds to answer.
            eta = "辨識約 1 到 2 分鐘(含載入模型)"
            if job.media_kind is JobKind.IMAGE_UNDERSTAND:
                eta += ",這個看圖模型只會用英文回答"
        else:
            eta = "圖約 30 秒、影片約 2 分鐘" if self.is_warm() else "暖機中:圖約 3 分鐘、影片約 5 分鐘"
        line = f"收到,{eta},想查進度可以看\n{self._link(job)}"
        if self.queue.push_quota_exhausted():
            # Push is gone for the month: the result will not arrive on its
            # own. Say so now, while there is a reply token to say it with.
            line += f"\n本月推播額度已用完,完成後請自己說「{self.show_trigger}」來領取。"
        return line

    def _over_drama_cap(self) -> bool:
        """Has the *group* used today's dramas? Group-wide by design: the
        cost is the pod's, not one member's. 0 disables."""
        if not self.max_dramas_per_day:
            return False
        return self.queue.accepted_kind_today(JobKind.DRAMA) >= self.max_dramas_per_day

    def _over_daily_cap(self, user_id: str | None, media_kind: JobKind) -> bool:
        """Has this user used up today's allowance?

        Chat has its own separate counter
        (`max_chat_messages_per_user_per_day`) rather than sharing
        `max_jobs_per_user_per_day` with video/image/understanding — a
        normal conversation's cadence would otherwise exhaust a user's
        entire daily video/image allowance too (`accepted_today()` excludes
        chat rows for the same reason). Skipped entirely when the relevant
        cap is 0 (off) or the id is unknown — LINE omits `source.userId` for
        a user who has not accepted the Official Account terms, and there is
        no per-user budget to enforce against a user we cannot name. The
        group and user allowlists are what stand between that case and an
        open bar.
        """
        if not user_id:
            return False
        if media_kind is JobKind.CHAT:
            if not self.max_chat_messages_per_user_per_day:
                return False
            return self.queue.accepted_chat_today(user_id) >= self.max_chat_messages_per_user_per_day
        if not self.max_jobs_per_user_per_day:
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
                f"「{self.i2i_trigger} …」重畫照片。也可以先傳照片/語音/影片,"
                f"再說「{self.describe_image_trigger}」/「{self.describe_audio_trigger}」/"
                f"「{self.describe_video_trigger}」聽聽 AI 怎麼形容(後面可以加想問的話)。"
                f"想聊天就用「{self.chat_trigger} …」。"
                f"傳影片再說「{self.extract_audio_trigger}」可以把聲音抽成音檔。"
                f"想看一分鐘有劇情的短劇,說「{self.drama_trigger} <一句故事前提>」(約半小時)。"
                f"做好但還沒送到的成品,說「{self.show_trigger}」領取。",
            )
            return Outcome("status")

        lines = [f"排隊中 {len(pending)} 件"]
        lines += [
            f"  · {self._STATUS_GLYPH.get(j.media_kind, '🎬')} "
            f"{j.text[:18]} — {self._state_zh(j)}"
            for j in pending[:5]
        ]
        lines += [f"  ✓ {j.text[:18]}\n{self._link(j)}" for j in recent_done[:3]]
        await self._safe_reply(reply_token, "\n".join(lines))
        return Outcome("status")

    # -------------------------------------------------------------- helpers

    def _strip_trigger(self, text: str) -> tuple[str, JobKind, str | None, float | None] | None:
        """The prompt after the trigger, which kind it asked for, which
        pending-media cache (if any) it claims, and any requested length in
        seconds — or None if it is not a request at all. Exactly one spelling
        per trigger, no aliases.

        The third element is `"image"`/`"audio"`/`"video"` naming which cache
        `_take_pending` must pop, or `None` for a trigger that touches no
        cache at all. A bare bool stopped being enough once there was more
        than one kind of pending media to consume.

        A length rides on the trigger itself: `/影片15s` or `/圖影15秒`. It is
        read only off video triggers (an image has no length), and only when
        it sits flush against the trigger, so an ordinary prompt that happens
        to start with a number is untouched. The value is clamped to the
        model's range at conversion; here it is just the number the user typed.
        """
        # A Chinese mobile IME turns a leading "/" into the fullwidth solidus
        # (U+FF0F) more often than not, and such a message matches no trigger
        # and is silently ignored -- which reads to the user as a dead bot.
        # Normalise a leading fullwidth solidus to ASCII before matching.
        if text[:1] == "\uff0f":
            text = "/" + text[1:]
        for prefix, kind, wants_media in (
            (self.i2v_trigger, JobKind.VIDEO, "image"),
            (self.i2i_trigger, JobKind.IMAGE, "image"),
            (self.trigger, JobKind.VIDEO, None),
            (self.image_trigger, JobKind.IMAGE, None),
            (self.describe_image_trigger, JobKind.IMAGE_UNDERSTAND, "image"),
            (self.describe_audio_trigger, JobKind.AUDIO_UNDERSTAND, "audio"),
            (self.describe_video_trigger, JobKind.VIDEO_UNDERSTAND, "video"),
            (self.chat_trigger, JobKind.CHAT, None),
            (self.drama_trigger, JobKind.DRAMA, None),
        ):
            if text.startswith(prefix):
                rest = text[len(prefix) :]
                seconds: float | None = None
                if kind is JobKind.VIDEO:
                    match = _LENGTH_RE.match(rest)
                    if match:
                        seconds = float(match.group(1))
                        rest = rest[match.end() :]
                return rest.strip(), kind, wants_media, seconds
        return None

    _DESCRIBE_EXAMPLE: ClassVar[dict[JobKind, str]] = {
        JobKind.IMAGE_UNDERSTAND: "這是什麼品種的狗",
        JobKind.AUDIO_UNDERSTAND: "他在唱什麼歌",
        JobKind.VIDEO_UNDERSTAND: "他最後做了什麼",
    }

    def _media_trigger(self, media_kind: JobKind, wants_media: str) -> str:
        """Which trigger word claims this pending media -- for refusal/usage text."""
        if media_kind is JobKind.IMAGE_UNDERSTAND:
            return self.describe_image_trigger
        if media_kind is JobKind.AUDIO_UNDERSTAND:
            return self.describe_audio_trigger
        if media_kind is JobKind.VIDEO_UNDERSTAND:
            return self.describe_video_trigger
        return self.i2v_trigger if media_kind is JobKind.VIDEO else self.i2i_trigger

    _MEDIA_NOUN: ClassVar[dict[str, str]] = {"image": "照片", "audio": "語音", "video": "影片"}
    _MEDIA_VERB: ClassVar[dict[str, str]] = {
        "image": "傳一張照片", "audio": "傳一段語音訊息", "video": "傳一段影片",
    }
    _STATUS_GLYPH: ClassVar[dict[JobKind, str]] = {
        JobKind.IMAGE: "🖼",
        JobKind.IMAGE_UNDERSTAND: "🔎🖼",
        JobKind.AUDIO_UNDERSTAND: "🔎🎧",
        JobKind.VIDEO_UNDERSTAND: "🔎🎬",
        JobKind.CHAT: "💬",
        JobKind.DRAMA: "🎭",
    }
    """`_status()`'s per-kind glyph. VIDEO falls back to `.get(..., '🎬')`."""

    def _usage(self, media_kind: JobKind, wants_media: str | None) -> str:
        if media_kind.is_understanding:
            assert wants_media is not None  # every understanding kind wants media
            trigger = self._media_trigger(media_kind, wants_media)
            example = self._DESCRIBE_EXAMPLE[media_kind]
            return (
                f"用法:先{self._MEDIA_VERB[wants_media]},再輸入 {trigger};"
                f"後面可以加想問的話,例如「{trigger} {example}」"
            )
        if media_kind is JobKind.CHAT:
            return f"用法:{self.chat_trigger} <想說的話>"
        if media_kind is JobKind.DRAMA:
            return f"用法:{self.drama_trigger} <一句故事前提>,例如「{self.drama_trigger} 一個夜市老闆娘發現攤位下藏著一封信」"
        if wants_media:
            return f"用法:先傳一張照片,再說 {self._media_trigger(media_kind, wants_media)} <想看的畫面>"
        trigger = self.trigger if media_kind is JobKind.VIDEO else self.image_trigger
        return f"用法:{trigger} <想看的畫面>"

    def _no_pending_media(self, media_kind: JobKind, wants_media: str) -> str:
        trigger = self._media_trigger(media_kind, wants_media)
        noun = self._MEDIA_NOUN[wants_media]
        verb = self._MEDIA_VERB[wants_media]
        tail = "(可加一句想問的)" if media_kind.is_understanding else " <想看的畫面>"
        return (
            f"找不到你的{noun}。先{verb}到群組,再說「{trigger}{tail}」"
            f"({int(IMAGE_PAIRING_TTL_S // 60)} 分鐘內有效)。"
        )

    def _is_status_query(self, text: str) -> bool:
        return any(text.startswith(w) for w in STATUS_WORDS)

    def _is_extract_audio_request(self, text: str) -> bool:
        if text[:1] == "\uff0f":
            text = "/" + text[1:]
        return text.split(maxsplit=1)[:1] == [self.extract_audio_trigger] or text == self.extract_audio_trigger

    async def _extract_audio(self, group_id: str, user_id: str | None, reply_token: str) -> Outcome:
        """`/影音`: the audio track of the sender's last video, as an M4A.

        Inline, not enqueued: ffmpeg on this host, no GPU. The clip must be
        the sender's own cached one (same rule as `/說影`). Delivered as a
        LINE audio message in the free reply -- no push quota -- with the
        link as a text fallback, and every failure said in words: no clip,
        no audio track, or ffmpeg refusing the file.
        """
        path = self._take_pending_video(group_id, user_id)
        if path is None:
            await self._safe_reply(reply_token, self._no_pending_media_for_extract())
            return Outcome("ignored", detail="extract-audio: no pending video")
        if self.files_dir is None:
            await self._safe_reply(reply_token, "這台主機沒有設定檔案目錄,無法轉出音檔。")
            return Outcome("ignored", detail="extract-audio: no files_dir")

        from ai_studio.media import FFmpegError, extract_audio

        dest = self.files_dir / f"{Path(path).stem}_audio.m4a"
        try:
            out, duration_ms = extract_audio(Path(path), dest)
        except FFmpegError as exc:
            reason = "這段影片沒有聲音軌" if "no audio track" in str(exc) else f"轉檔失敗:{str(exc)[:80]}"
            await self._safe_reply(reply_token, f"{reason}\n(影片本身不會被保存)")
            _log.warning("extract-audio failed for %s: %s", path, exc)
            return Outcome("extract_audio", detail="failed")

        url = f"{self.base_url}/files/{out.name}"
        messages: list[dict[str, Any]] = [
            {"type": "audio", "originalContentUrl": url, "duration": max(duration_ms, 1)},
            {"type": "text", "text": f"音檔轉好了({duration_ms / 1000:.1f} 秒,m4a)\n{url}"},
        ]
        try:
            await self.replier.reply_messages(reply_token, messages)
        except Exception as exc:  # the 200 matters more; see _safe_reply
            _log.warning("extract-audio reply failed: %s", exc)
            return Outcome("extract_audio", detail="reply failed")
        return Outcome("extract_audio", detail=out.name)

    def _no_pending_media_for_extract(self) -> str:
        return (
            f"找不到你的影片。先傳一段影片到群組,再說「{self.extract_audio_trigger}」"
            f"({int(IMAGE_PAIRING_TTL_S // 60)} 分鐘內有效)。"
        )

    def _is_show_request(self, text: str) -> bool:
        if text[:1] in ("/", "\uff0f"):
            text = text[1:]
        return text.startswith(self.show_trigger)

    async def _show(
        self, group_id: str, reply_token: str, *, quoted_message_id: str | None = None
    ) -> Outcome:
        """「讓我看看」: hand over finished results as a reply.

        A reply is free where a push is metered, so this is the delivery path
        for the month in which the push quota is gone (`pipeline.worker.
        _deliver` leaves such jobs undelivered on purpose).

        Quoting an earlier message names one job: that one is handed over
        whatever its delivery state (asking to see it again is free), and a
        job that is not done says so explicitly -- still running, failed with
        its reason, or the quoted message is not a request at all. Silence
        here would read as a broken bot.

        Bare, it hands over everything finished and not yet delivered: up to
        four media objects and one summary text per call (LINE's five-message
        reply limit), oldest first. Only what actually went out is marked
        delivered, so a rejected reply is retried by asking again.
        """
        if quoted_message_id:
            return await self._show_one(group_id, reply_token, quoted_message_id)

        waiting = [j for j in self.queue.undelivered(50) if j.group_id == group_id]
        if not waiting:
            pending = [j for j in self.queue.pending() if j.group_id == group_id]
            tail = f"還有 {len(pending)} 件在處理中,晚點再說一次。" if pending else "目前沒有在處理的工作。"
            await self._safe_reply(reply_token, f"沒有等著給你看的成品。{tail}")
            return Outcome("show")

        batch = waiting[:MAX_SHOW_JOBS]
        messages: list[dict[str, Any]] = []
        lines: list[str] = []
        for job in batch:
            media = self._media_message(job)
            if media is not None:
                messages.append(media)
            lines.append(self._show_line(job))
        rest = len(waiting) - len(batch)
        if rest:
            lines.append(f"還有 {rest} 件,再說一次「{self.show_trigger}」。")
        messages.append({"type": "text", "text": "\n".join(lines)[:5000]})
        try:
            await self.replier.reply_messages(reply_token, messages)
        except Exception as exc:  # the 200 matters more; see _safe_reply
            _log.warning("show reply failed, nothing marked delivered: %s", exc)
            return Outcome("show", detail="reply failed")
        for job in batch:
            self.queue.mark_delivered(job.id)
        for job in batch:
            _log.info("pulled", extra={"job_id": job.id, "token": job.token, "kind": job.media_kind.value,
                                       "stage": "deliver", "outcome": "pulled"})
        return Outcome("show", detail=f"{len(batch)} handed over")

    async def _show_one(self, group_id: str, reply_token: str, quoted_message_id: str) -> Outcome:
        job = self.queue.by_quoted_message(group_id, quoted_message_id)
        if job is None:
            await self._safe_reply(
                reply_token, "你引用的那則訊息不是這個 BOT 收下的請求,沒有對應的成品。"
            )
            return Outcome("show", detail="quoted message unknown")
        if job.state is JobState.FAILED:
            await self._safe_reply(
                reply_token, f"那件失敗了:{(job.error or 'unknown')[:80]}\n{self._link(job)}"
            )
            return Outcome("show", job=job, detail="failed")
        if job.state is not JobState.DONE:
            await self._safe_reply(
                reply_token,
                f"那件還沒好({self._state_zh(job)}),晚點再引用一次。\n{self._link(job)}",
            )
            return Outcome("show", job=job, detail="not done")

        messages: list[dict[str, Any]] = []
        media = self._media_message(job)
        if media is not None:
            messages.append(media)
        elif job.output_path:
            # The file exists but nothing can preview it (no poster): say so
            # and hand over the link, rather than a summary that hides it.
            messages.append({
                "type": "text",
                "text": f"{job.text[:18]} 完成了,但預覽圖不在,直接開:\n{self._link(job)}",
            })
        if not messages or job.result_text:
            messages.append({"type": "text", "text": self._show_line(job)[:5000]})
        try:
            await self.replier.reply_messages(reply_token, messages)
        except Exception as exc:  # the 200 matters more; see _safe_reply
            _log.warning("show reply failed for job %d: %s", job.id, exc)
            return Outcome("show", job=job, detail="reply failed")
        if job.delivered_at is None:
            self.queue.mark_delivered(job.id)
        _log.info("pulled", extra={"job_id": job.id, "token": job.token, "kind": job.media_kind.value,
                                   "stage": "deliver", "outcome": "pulled"})
        return Outcome("show", job=job, detail="handed over")

    def _media_message(self, job: Job) -> dict[str, Any] | None:
        """The media object for a finished generation job, or None when there
        is no file (understanding/chat/failed) or no poster to preview it with
        -- the link in the summary text still gets it to the user."""
        if job.state is not JobState.DONE or not job.output_path:
            return None
        name = Path(job.output_path).name
        url = f"{self.base_url}/files/{name}"
        poster = f"{Path(name).stem}_poster.jpg"
        has_poster = self.files_dir is not None and (self.files_dir / poster).is_file()
        if job.media_kind is JobKind.VIDEO:
            if not has_poster:
                return None
            return {"type": "video", "originalContentUrl": url,
                    "previewImageUrl": f"{self.base_url}/files/{poster}"}
        if job.media_kind is JobKind.IMAGE:
            preview = f"{self.base_url}/files/{poster}" if has_poster else url
            return {"type": "image", "originalContentUrl": url, "previewImageUrl": preview}
        return None

    def _show_line(self, job: Job) -> str:
        glyph = self._STATUS_GLYPH.get(job.media_kind, "🎬")
        if job.state is JobState.FAILED:
            return f"{glyph} {job.text[:18]} — 失敗:{(job.error or '')[:60]}"
        if job.result_text:
            return f"{glyph} {job.text[:18]}\n{job.result_text[:400]}"
        return f"{glyph} {job.text[:18]}\n{self._link(job)}"

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

    async def _safe_reply(self, reply_token: str, text: str) -> list[str]:
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
            return []
        try:
            sent = await self.replier.reply_text(reply_token, text)
        except Exception as exc:  # see docstring: the 200 matters more
            _log.warning("reply failed for token %s: %s", reply_token, exc)
            return []
        return [str(i) for i in sent] if isinstance(sent, list | tuple) else []
