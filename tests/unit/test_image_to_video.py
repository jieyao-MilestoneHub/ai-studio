"""Image-to-video: a photo, then `/圖影`, becomes the H3 first frame.

Three layers, tested separately:

- the queue carries `first_frame_path` from enqueue to render, since a
  request may not actually render until the next business window
- the webhook pairs a cached photo with the next `/圖影` from the same
  sender, and only that: `/影片` and `/圖片` leave the photo alone, and a
  `/圖影` with no photo is told so instead of rendering from text
- the ComfyUI provider uploads that file and switches to the
  image-conditioned sibling workflow, since one static graph cannot both
  wire and not-wire a `LoadImage` node depending on the request
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_studio.bots.line.content import LineContentError, NullContentClient
from ai_studio.bots.line.reply import NullReplyClient
from ai_studio.bots.line.verify import sign
from ai_studio.bots.line.webhook import IMAGE_PAIRING_TTL_S, WebhookHandler
from ai_studio.core.enums import MediaKind
from ai_studio.core.errors import ProviderSubmitError
from ai_studio.core.provider_spec import ClipRequest
from ai_studio.pipeline.queue import JobQueue
from ai_studio.providers.comfyui import ComfyUIProvider

SECRET = "test-channel-secret"
GROUP = "Cae56f94637c1234567890abcdef12345"
USER = "U" + "1" * 32
REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------- queue


def test_first_frame_path_round_trips_through_the_queue(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "q.sqlite3")
    try:
        job, created = queue.enqueue(
            "evt-1", GROUP, "一隻貓在窗邊", user_id=USER,
            first_frame_path="/incoming/abc123.jpg",
        )
        assert created
        assert job.first_frame_path == "/incoming/abc123.jpg"
        assert queue.by_id(job.id).first_frame_path == "/incoming/abc123.jpg"
    finally:
        queue.close()


def test_first_frame_path_defaults_to_none(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "q.sqlite3")
    try:
        job, _ = queue.enqueue("evt-1", GROUP, "一隻貓在窗邊", user_id=USER)
        assert job.first_frame_path is None
    finally:
        queue.close()


# ------------------------------------------------------------------- webhook


def _body(events: list[dict]) -> bytes:
    return json.dumps({"destination": "U" + "0" * 32, "events": events}).encode("utf-8")


def _image_event(*, message_id: str = "img-1", user: str = USER, event_id: str = "evt-img") -> dict:
    return {
        "type": "message",
        "mode": "active",
        "webhookEventId": event_id,
        "timestamp": 1700000000000,
        "replyToken": "rt-" + event_id,
        "source": {"type": "group", "groupId": GROUP, "userId": user},
        "message": {"type": "image", "id": message_id},
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


@pytest.mark.asyncio
async def test_a_photo_then_a_trigger_becomes_the_first_frame(tmp_path: Path) -> None:
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, replier = _wired(tmp_path, content=content)

    (image_outcome,) = await _send(handler, _image_event())
    assert image_outcome.action == "image"
    assert not replier.sent, "a bare photo must get no reply"

    (accept_outcome,) = await _send(handler, _text_event("/圖影 貓在下雨的窗邊"))
    assert accept_outcome.action == "accepted"
    job = accept_outcome.job
    assert job is not None and job.first_frame_path is not None
    assert Path(job.first_frame_path).read_bytes() == b"fake-jpeg-bytes"
    queue.close()


@pytest.mark.asyncio
async def test_a_photo_is_not_reused_after_being_claimed(tmp_path: Path) -> None:
    """A photo is a first frame for exactly one request, not a standing
    instruction applied to everything the sender says next."""
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(handler, _image_event())
    (first,) = await _send(handler, _text_event("/圖影 貓", event_id="evt-1"))
    assert first.job is not None and first.job.first_frame_path is not None
    (second,) = await _send(handler, _text_event("/圖影 狗", event_id="evt-2"))

    assert second.action == "ignored" and second.job is None
    assert second.detail == "no pending image"
    queue.close()


@pytest.mark.asyncio
async def test_other_triggers_do_not_consume_a_pending_photo(tmp_path: Path) -> None:
    """Only /圖影 claims a photo. Flux (/圖片) has no first-frame concept, and
    /影片 is text-to-video by definition -- a photo meant for the /圖影 after
    them must not be silently eaten by either."""
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(handler, _image_event())
    (img_job_outcome,) = await _send(handler, _text_event("/圖片 一隻貓", event_id="evt-1"))
    assert img_job_outcome.job is not None
    assert img_job_outcome.job.media_kind is MediaKind.IMAGE
    assert img_job_outcome.job.first_frame_path is None

    (t2v_outcome,) = await _send(handler, _text_event("/影片 一隻貓", event_id="evt-2"))
    assert t2v_outcome.job is not None
    assert t2v_outcome.job.media_kind is MediaKind.VIDEO
    assert t2v_outcome.job.first_frame_path is None

    (i2v_outcome,) = await _send(handler, _text_event("/圖影 一隻貓", event_id="evt-3"))
    assert i2v_outcome.job is not None and i2v_outcome.job.first_frame_path is not None
    queue.close()


@pytest.mark.asyncio
async def test_i2i_claims_the_photo_as_an_image_job(tmp_path: Path) -> None:
    """/圖圖 is the image counterpart of /圖影: same cache, same one-shot
    claim, but the job is a Flux image whose source is the photo."""
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(handler, _image_event())
    (outcome,) = await _send(handler, _text_event("/圖圖 變成油畫"))

    assert outcome.action == "accepted"
    job = outcome.job
    assert job is not None and job.media_kind is MediaKind.IMAGE
    assert job.first_frame_path is not None
    assert Path(job.first_frame_path).read_bytes() == b"fake-jpeg-bytes"
    assert job.text == "變成油畫"
    queue.close()


@pytest.mark.asyncio
async def test_i2i_without_a_photo_is_refused_naming_its_own_trigger(tmp_path: Path) -> None:
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"x"))

    (outcome,) = await _send(handler, _text_event("/圖圖 變成油畫"))

    assert outcome.action == "ignored" and outcome.job is None
    assert queue.counts() == {}
    reply = replier.sent[0][1][0]
    assert "照片" in reply and "/圖圖" in reply and "/圖影" not in reply


@pytest.mark.asyncio
async def test_i2i_and_i2v_do_not_shadow_each_other_or_the_plain_triggers(tmp_path: Path) -> None:
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(handler, _image_event())
    (plain,) = await _send(handler, _text_event("/圖片 一隻貓", event_id="e1"))
    assert plain.job is not None and plain.job.first_frame_path is None
    (i2i,) = await _send(handler, _text_event("/圖圖 一隻貓", event_id="e2"))
    assert i2i.job is not None and i2i.job.first_frame_path is not None
    assert i2i.job.media_kind is MediaKind.IMAGE
    (i2v,) = await _send(handler, _text_event("/圖影 一隻貓", event_id="e3"))
    assert i2v.action == "ignored", "the one photo was already claimed by /圖圖"
    queue.close()


@pytest.mark.asyncio
async def test_i2v_without_a_photo_is_refused_with_instructions(tmp_path: Path) -> None:
    """The user asked for their picture to move; a clip of something else is
    not that. Nothing is queued, and the reply says what to do."""
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"x"))

    (outcome,) = await _send(handler, _text_event("/圖影 一隻貓"))

    assert outcome.action == "ignored" and outcome.job is None
    assert queue.counts() == {}
    reply = replier.sent[0][1][0]
    assert "照片" in reply and "/圖影" in reply


@pytest.mark.asyncio
async def test_a_bare_i2v_trigger_gets_its_own_usage_line(tmp_path: Path) -> None:
    handler, queue, replier = _wired(tmp_path, content=NullContentClient(b"x"))
    (outcome,) = await _send(handler, _text_event("/圖影"))
    assert outcome.action == "ignored"
    assert queue.counts() == {}
    assert "先傳一張照片" in replier.sent[0][1][0]


@pytest.mark.asyncio
async def test_a_stale_photo_is_not_used(tmp_path: Path) -> None:
    clock_value = [0.0]

    def clock():
        from datetime import datetime, timezone
        return datetime.fromtimestamp(clock_value[0], tz=timezone.utc)

    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content, clock=clock)

    await _send(handler, _image_event())
    clock_value[0] += IMAGE_PAIRING_TTL_S + 1
    (outcome,) = await _send(handler, _text_event("/圖影 貓"))

    assert outcome.action == "ignored" and outcome.job is None
    queue.close()


@pytest.mark.asyncio
async def test_a_photo_from_one_user_is_not_claimed_by_another(tmp_path: Path) -> None:
    content = NullContentClient(b"fake-jpeg-bytes")
    handler, queue, _ = _wired(tmp_path, content=content)

    await _send(handler, _image_event(user="U" + "2" * 32))
    (outcome,) = await _send(handler, _text_event("/圖影 貓", user=USER))

    assert outcome.action == "ignored" and outcome.job is None
    queue.close()


@pytest.mark.asyncio
async def test_no_content_client_means_images_are_silently_ignored(tmp_path: Path) -> None:
    handler, queue, replier = _wired(tmp_path, content=None)
    (outcome,) = await _send(handler, _image_event())
    assert outcome.action == "ignored"
    assert not replier.sent
    queue.close()


@pytest.mark.asyncio
async def test_a_failed_content_fetch_is_logged_and_ignored_not_raised(tmp_path: Path) -> None:
    class _Failing:
        async def fetch(self, message_id: str) -> bytes:
            raise LineContentError("boom")

    handler, queue, replier = _wired(tmp_path, content=_Failing())
    (outcome,) = await _send(handler, _image_event())
    assert outcome.action == "ignored"
    assert not replier.sent
    queue.close()


# ------------------------------------------------------------------ provider


@pytest.fixture
def provider() -> ComfyUIProvider:
    return ComfyUIProvider(
        REPO / "workflows" / "h3_fl2va_turbo.json", base_url="http://fake-pod:8188"
    )


def test_the_i2va_sibling_is_found_next_to_the_base_workflow(provider: ComfyUIProvider) -> None:
    assert provider._i2va_workflow is not None
    assert "first_frame" in provider._i2va_workflow.bindings


def _copy_workflow_under(tmp_path: Path, name: str) -> Path:
    """The base H3 workflow's content under a name with no "fl2va" in it, so
    `_load_i2va_sibling`'s filename substitution has nothing to find."""
    dest = tmp_path / name
    dest.write_text(
        (REPO / "workflows" / "h3_fl2va_turbo.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return dest


def test_a_workflow_with_no_i2va_sibling_has_none(tmp_path: Path) -> None:
    provider = ComfyUIProvider(
        _copy_workflow_under(tmp_path, "custom.json"), base_url="http://fake-pod:8188"
    )
    assert provider._i2va_workflow is None


@pytest.mark.asyncio
async def test_submit_with_a_first_frame_uploads_and_switches_workflow(
    provider: ComfyUIProvider, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    async def fake_upload_image(data: bytes, filename: str) -> str:
        captured["uploaded_bytes"] = data
        captured["uploaded_filename"] = filename
        return "server_side_name.png"

    async def fake_queue_prompt(graph: dict[str, Any]) -> str:
        captured["graph"] = graph
        return "prompt-1"

    provider.client.upload_image = fake_upload_image  # type: ignore[method-assign]
    provider.client.queue_prompt = fake_queue_prompt  # type: ignore[method-assign]

    source = tmp_path / "photo.jpg"
    source.write_bytes(b"real-photo-bytes")
    request = ClipRequest(
        shot_id="job1", prompt="a cat", width=864, height=480,
        duration_s=5.0, fps=24, seed=7, first_frame_path=str(source),
    )

    job = await provider.submit(request)

    assert job.job_id == "prompt-1"
    assert captured["uploaded_bytes"] == b"real-photo-bytes"
    assert captured["uploaded_filename"] == "photo.jpg"
    # node 140 is the i2va sibling's LoadImage node -- confirms the i2va
    # graph was used, not the base text-only one, which has no such node.
    assert captured["graph"]["140"]["inputs"]["image"] == "server_side_name.png"


@pytest.mark.asyncio
async def test_submit_without_a_first_frame_never_touches_upload(
    provider: ComfyUIProvider,
) -> None:
    calls: list[Any] = []

    async def fake_upload_image(data: bytes, filename: str) -> str:
        calls.append((data, filename))
        return "unused"

    async def fake_queue_prompt(graph: dict[str, Any]) -> str:
        return "prompt-1"

    provider.client.upload_image = fake_upload_image  # type: ignore[method-assign]
    provider.client.queue_prompt = fake_queue_prompt  # type: ignore[method-assign]

    request = ClipRequest(
        shot_id="job1", prompt="a cat", width=864, height=480, duration_s=5.0, fps=24,
    )
    await provider.submit(request)

    assert calls == []


@pytest.mark.asyncio
async def test_a_first_frame_with_no_i2va_sibling_raises_clearly(tmp_path: Path) -> None:
    provider = ComfyUIProvider(
        _copy_workflow_under(tmp_path, "custom.json"), base_url="http://fake-pod:8188"
    )
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")
    request = ClipRequest(
        shot_id="job1", prompt="a cat", width=864, height=480,
        duration_s=5.0, fps=24, first_frame_path=str(source),
    )

    with pytest.raises(ProviderSubmitError, match="no image-conditioned sibling"):
        await provider.submit(request)


@pytest.mark.asyncio
async def test_a_missing_first_frame_file_raises_a_provider_error(
    provider: ComfyUIProvider,
) -> None:
    request = ClipRequest(
        shot_id="job1", prompt="a cat", width=864, height=480,
        duration_s=5.0, fps=24, first_frame_path="/does/not/exist.jpg",
    )
    with pytest.raises(ProviderSubmitError, match="could not read"):
        await provider.submit(request)


# ------------------------------------------------------------------- content


def test_the_content_api_host_is_the_one_line_actually_serves() -> None:
    """`api-data.line.biz` was shipped once. developers.line.biz hosts the docs;
    the API is on `.me`, and the `.biz` host does not resolve, so every photo
    was dropped at DNS with "Name or service not known" and /圖影 replied
    "找不到你的照片" to a user who had just sent one."""
    from urllib.parse import urlsplit

    from ai_studio.bots.line.content import CONTENT_ENDPOINT

    assert urlsplit(CONTENT_ENDPOINT).hostname == "api-data.line.me"
    assert CONTENT_ENDPOINT.endswith("/v2/bot/message/{message_id}/content")
