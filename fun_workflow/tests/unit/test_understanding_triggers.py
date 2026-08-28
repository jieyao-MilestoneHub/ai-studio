"""Understanding: `/說圖` `/說音` `/說影` describe a photo/audio/video clip
instead of generating one. Same photo-cache mechanism `/圖影`/`/圖圖` use
(`test_image_to_video.py`), extended to audio and video, but reversed: the
media is consumed, not produced, so it lands in `input_media_path`, never
`first_frame_path` -- and none of the three take trailing text, the mirror
image of the four generation triggers, which require it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ai_studio.core.enums import MediaKind

from fun_workflow.bots.line.content import LineContentError, NullContentClient
from fun_workflow.bots.line.reply import NullReplyClient
from fun_workflow.bots.line.verify import sign
from fun_workflow.bots.line.webhook import IMAGE_PAIRING_TTL_S, WebhookHandler
from fun_workflow.pipeline.queue import JobQueue

SECRET = "test-channel-secret"
GROUP = "Cae56f94637c1234567890abcdef12345"
USER = "U" + "1" * 32


def _media_event(kind: str, *, message_id: str, user: str = USER, event_id: str, **extra) -> dict:
    message = {"type": kind, "id": message_id, **extra}
    return {
        "type": "message",
        "mode": "active",
        "webhookEventId": event_id,
        "timestamp": 1700000000000,
        "replyToken": "rt-" + event_id,
        "source": {"type": "group", "groupId": GROUP, "userId": user},
        "message": message,
    }


def _text_event(text: str, *, user: str = USER, event_id: str = "evt-txt") -> dict:
    return {
        "type": "message",
        "mode": "active",
        "webhookEventId": event_id,
        "timestamp": 1700000000000,
        "replyToken": "rt-" + event_id,
        "source": {"type": "group", "groupId": GROUP, "userId": user},
        "message": {"type": "text", "id": "m1", "text": text},
    }


def _body(events: list[dict]) -> bytes:
    return json.dumps({"destination": "U" + "0" * 32, "events": events}).encode("utf-8")


async def _send(handler: WebhookHandler, event: dict):
    body = _body([event])
    return await handler.handle(body, sign(body, SECRET))


def _wired(tmp_path: Path, *, content=None, clock=None):
    queue = JobQueue(tmp_path / "q.sqlite3")
    replier = NullReplyClient()
    handler = WebhookHandler(
        queue,
        replier,
        channel_secret=SECRET,
        allowed_group_id=GROUP,
        base_url="https://vg.example.com/",
        content=content,
        incoming_dir=tmp_path / "incoming",
        clock=clock,
    )
    return handler, queue, replier


# ------------------------------------------------------------------- /說圖


@pytest.mark.asyncio
async def test_a_photo_then_describe_image_becomes_an_understanding_job(tmp_path: Path) -> None:
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(handler, _media_event("image", message_id="img-1", event_id="evt-img"))
    (outcome,) = await _send(handler, _text_event("/說圖"))

    assert outcome.action == "accepted"
    job = outcome.job
    assert job is not None
    assert job.media_kind is MediaKind.IMAGE_UNDERSTAND
    assert job.input_media_path is not None
    assert job.first_frame_path is None
    assert Path(job.input_media_path).read_bytes() == b"fake-jpeg-bytes"
    queue.close()


@pytest.mark.asyncio
async def test_describe_image_without_a_photo_is_refused_naming_its_own_trigger(
    tmp_path: Path,
) -> None:
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"x"))

    (outcome,) = await _send(handler, _text_event("/說圖"))

    assert outcome.action == "ignored" and outcome.job is None
    assert queue.counts() == {}
    reply = replier.sent[0][1][0]
    assert "照片" in reply and "/說圖" in reply and "/圖影" not in reply
    queue.close()


@pytest.mark.asyncio
async def test_describe_image_with_trailing_text_is_accepted_as_the_question(tmp_path: Path) -> None:
    """Since 2026-08-27 the describe triggers take optional text: it is the
    user's question, rewritten on the pod into the model's best form. Kept
    verbatim on the job so the rewrite has the original to work from."""
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, replier = _wired(tmp_path, content=content)

    await _send(handler, _media_event("image", message_id="img-1", event_id="evt-img"))
    (outcome,) = await _send(handler, _text_event("/說圖 這是誰"))

    assert outcome.action == "accepted" and outcome.job is not None
    assert outcome.job.text == "這是誰"
    assert queue.counts() == {"queued": 1}
    assert "只會用英文回答" in replier.sent[-1][1][0]
    queue.close()


@pytest.mark.asyncio
async def test_describe_usage_advertises_the_optional_question(tmp_path: Path) -> None:
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"x"))
    (outcome,) = await _send(handler, _text_event("/說圖"))
    assert outcome.action == "ignored"  # no photo cached
    assert "可加一句想問的" in replier.sent[-1][1][0]
    queue.close()


@pytest.mark.asyncio
async def test_describe_image_does_not_shadow_or_get_shadowed_by_i2v_i2i(tmp_path: Path) -> None:
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(handler, _media_event("image", message_id="img-1", event_id="evt-img"))
    (outcome,) = await _send(handler, _text_event("/說圖"))
    assert outcome.job is not None and outcome.job.media_kind is MediaKind.IMAGE_UNDERSTAND

    # The photo was already claimed -- a second photo-consuming trigger finds
    # nothing cached.
    (second,) = await _send(handler, _text_event("/圖影 貓", event_id="evt-2"))
    assert second.action == "ignored"
    queue.close()


# ------------------------------------------------------------------- /說音


@pytest.mark.asyncio
async def test_an_audio_clip_then_describe_audio_becomes_an_understanding_job(
    tmp_path: Path,
) -> None:
    content = NullContentClient(b"fake-audio-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(
        handler,
        _media_event("audio", message_id="aud-1", event_id="evt-aud", duration=8000),
    )
    (outcome,) = await _send(handler, _text_event("/說音"))

    assert outcome.action == "accepted"
    job = outcome.job
    assert job is not None
    assert job.media_kind is MediaKind.AUDIO_UNDERSTAND
    assert job.input_media_path is not None
    assert Path(job.input_media_path).read_bytes() == b"fake-audio-bytes"
    queue.close()


@pytest.mark.asyncio
async def test_audio_over_the_duration_cap_is_never_even_fetched(tmp_path: Path) -> None:
    """The cap is read off LINE's own `duration` field in the webhook payload
    itself, so an over-long clip costs zero bandwidth, not just zero GPU."""
    content = NullContentClient(b"fake-audio-bytes")
    handler, queue, replier = _wired(tmp_path, content=content)

    (image_outcome,) = await _send(
        handler,
        _media_event("audio", message_id="aud-1", event_id="evt-aud", duration=45_000),
    )

    assert image_outcome.action == "ignored"
    assert content.calls == [], "an over-long clip must not be downloaded at all"
    assert not replier.sent, "passive media gets no reply, over cap or not"

    (outcome,) = await _send(handler, _text_event("/說音", event_id="evt-2"))
    assert outcome.action == "ignored", "nothing was cached, so /說音 finds nothing"
    queue.close()


@pytest.mark.asyncio
async def test_describe_audio_without_a_clip_is_refused_naming_its_own_trigger(
    tmp_path: Path,
) -> None:
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"x"))

    (outcome,) = await _send(handler, _text_event("/說音"))

    assert outcome.action == "ignored" and outcome.job is None
    reply = replier.sent[0][1][0]
    assert "語音" in reply and "/說音" in reply
    queue.close()


@pytest.mark.asyncio
async def test_a_photo_is_not_claimed_by_describe_audio(tmp_path: Path) -> None:
    """/說音 must never claim a photo meant for /說圖 or /圖影."""
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(handler, _media_event("image", message_id="img-1", event_id="evt-img"))
    (outcome,) = await _send(handler, _text_event("/說音"))

    assert outcome.action == "ignored"
    queue.close()


# ------------------------------------------------------------------- /說影


@pytest.mark.asyncio
async def test_a_video_clip_then_describe_video_becomes_an_understanding_job(
    tmp_path: Path,
) -> None:
    content = NullContentClient(b"fake-video-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(
        handler,
        _media_event("video", message_id="vid-1", event_id="evt-vid", duration=20_000),
    )
    (outcome,) = await _send(handler, _text_event("/說影"))

    assert outcome.action == "accepted"
    job = outcome.job
    assert job is not None
    assert job.media_kind is MediaKind.VIDEO_UNDERSTAND
    assert job.input_media_path is not None
    assert Path(job.input_media_path).read_bytes() == b"fake-video-bytes"
    queue.close()


@pytest.mark.asyncio
async def test_describe_video_without_a_clip_is_refused_naming_its_own_trigger(
    tmp_path: Path,
) -> None:
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"x"))

    (outcome,) = await _send(handler, _text_event("/說影"))

    assert outcome.action == "ignored" and outcome.job is None
    reply = replier.sent[0][1][0]
    assert "影片" in reply and "/說影" in reply
    queue.close()


# --------------------------------------------------------------- shared rules


@pytest.mark.asyncio
async def test_a_stale_audio_clip_is_not_used(tmp_path: Path) -> None:
    clock_value = [0.0]

    def clock():
        from datetime import datetime, timezone

        return datetime.fromtimestamp(clock_value[0], tz=timezone.utc)

    content = NullContentClient(b"fake-audio-bytes")
    handler, queue, _ = _wired(tmp_path, content=content, clock=clock)

    await _send(handler, _media_event("audio", message_id="aud-1", event_id="evt-aud", duration=1000))
    clock_value[0] += IMAGE_PAIRING_TTL_S + 1
    (outcome,) = await _send(handler, _text_event("/說音"))

    assert outcome.action == "ignored" and outcome.job is None
    queue.close()


@pytest.mark.asyncio
async def test_no_content_client_means_audio_and_video_are_silently_ignored(
    tmp_path: Path,
) -> None:
    handler, queue, replier = _wired(tmp_path, content=None)

    (audio_outcome,) = await _send(
        handler, _media_event("audio", message_id="a1", event_id="evt-a")
    )
    (video_outcome,) = await _send(
        handler, _media_event("video", message_id="v1", event_id="evt-v")
    )

    assert audio_outcome.action == "ignored"
    assert video_outcome.action == "ignored"
    assert not replier.sent
    queue.close()


@pytest.mark.asyncio
async def test_a_failed_audio_fetch_is_logged_and_ignored_not_raised(tmp_path: Path) -> None:
    class _Failing:
        async def fetch(self, message_id: str) -> bytes:
            raise LineContentError("boom")

    handler, queue, _ = _wired(tmp_path, content=_Failing())

    (outcome,) = await _send(
        handler, _media_event("audio", message_id="a1", event_id="evt-a")
    )
    assert outcome.action == "ignored"
    queue.close()


# ------------------------------------------------------------ /影音 (inline)


@pytest.mark.asyncio
async def test_extract_audio_replies_with_an_audio_message_and_no_job(tmp_path: Path, monkeypatch) -> None:
    """`/影音` never touches the queue or the GPU: the sender's cached clip
    goes through ffmpeg on the host and comes back in the free reply."""
    from ai_studio import media

    calls: list[tuple[Path, Path]] = []

    def fake_extract(src: Path, dest: Path, **_):
        calls.append((Path(src), Path(dest)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"m4a")
        return dest, 4321

    monkeypatch.setattr(media, "extract_audio", fake_extract)
    content = NullContentClient(b"fake-mp4-bytes")
    handler, queue, replier = _wired(tmp_path, content=content)
    handler.files_dir = tmp_path / "files"

    await _send(handler, _media_event("video", message_id="vid-1", event_id="evt-vid", duration=4000))
    (outcome,) = await _send(handler, _text_event("/影音"))

    assert outcome.action == "extract_audio" and outcome.detail.endswith("_audio.m4a")
    assert queue.counts() == {}, "no job, no pod"
    assert calls and calls[0][1].name.endswith("_audio.m4a")
    _, messages = replier.sent_messages[-1]
    assert messages[0]["type"] == "audio"
    assert messages[0]["duration"] == 4321
    assert messages[0]["originalContentUrl"].startswith("https://vg.example.com/files/")
    assert "4.3 秒" in messages[1]["text"]
    queue.close()


@pytest.mark.asyncio
async def test_extract_audio_without_a_clip_says_so(tmp_path: Path) -> None:
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"x"))
    handler.files_dir = tmp_path / "files"
    (outcome,) = await _send(handler, _text_event("/影音"))
    assert outcome.action == "ignored"
    assert "找不到你的影片" in replier.sent[-1][1][0]
    queue.close()


@pytest.mark.asyncio
async def test_extract_audio_reports_a_silent_clip_in_words(tmp_path: Path, monkeypatch) -> None:
    from ai_studio import media

    def no_track(src: Path, dest: Path, **_):
        raise media.FFmpegError(f"{Path(src).name} has no audio track to extract")

    monkeypatch.setattr(media, "extract_audio", no_track)
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"fake-mp4-bytes"))
    handler.files_dir = tmp_path / "files"
    await _send(handler, _media_event("video", message_id="vid-2", event_id="evt-vid2", duration=4000))
    (outcome,) = await _send(handler, _text_event("\uff0f影音"))  # the IME's fullwidth solidus
    assert outcome.action == "extract_audio" and outcome.detail == "failed"
    assert "沒有聲音軌" in replier.sent[-1][1][0]
    queue.close()
