"""The always-on HTTP surface.

Dependencies are injected so these run with no LINE credentials and no network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fun_workflow.api.main import create_app
from fun_workflow.bots.line.reply import NullReplyClient
from fun_workflow.bots.line.verify import sign
from fun_workflow.bots.line.webhook import WebhookHandler
from fun_workflow.pipeline.queue import JobQueue

SECRET = "test-channel-secret"
GROUP = "Cae56f94637c1234567890abcdef12345"
PROMPT = {"shots": [{"index": 1, "description": "一隻橘貓走在雨中"}]}


def _build(tmp_path: Path, *, allowed_group: str | None = GROUP):
    queue = JobQueue(tmp_path / "q.sqlite3")
    replier = NullReplyClient()
    handler = WebhookHandler(
        queue,
        replier,
        channel_secret=SECRET,
        allowed_group_id=allowed_group,
        base_url="https://vg.example.com",
    )
    app = create_app(queue=queue, handler=handler, files_dir=tmp_path / "files")
    return app, queue, replier


@pytest.fixture
def client(tmp_path: Path):
    app, queue, replier = _build(tmp_path)
    with TestClient(app) as c:
        yield c, queue, replier
    queue.close()


def _post(c: TestClient, events: list[dict], *, secret: str = SECRET):
    body = json.dumps({"destination": "U" + "0" * 32, "events": events}).encode("utf-8")
    return c.post(
        "/callback",
        content=body,
        headers={"x-line-signature": sign(body, secret), "content-type": "application/json"},
    )


def _event(text: str, event_id: str = "evt-1", group: str = GROUP) -> dict:
    return {
        "type": "message",
        "mode": "active",
        "webhookEventId": event_id,
        "timestamp": 1700000000000,
        "replyToken": "rt-" + event_id,
        "source": {"type": "group", "groupId": group, "userId": "U" + "1" * 32},
        "message": {"type": "text", "id": "m1", "text": text},
    }


# ---------------------------------------------------------------- /callback


def test_a_bad_signature_is_rejected_with_400(client) -> None:
    c, _, _ = client
    body = json.dumps({"events": []}).encode("utf-8")
    response = c.post("/callback", content=body, headers={"x-line-signature": "wrong"})
    assert response.status_code == 400


def test_a_missing_signature_header_is_rejected(client) -> None:
    c, _, _ = client
    response = c.post("/callback", content=b'{"events":[]}')
    assert response.status_code == 400


def test_the_verify_button_gets_a_200(client) -> None:
    """LINE's Verify sends events: [] and requires 200, or setup fails."""
    c, _, _ = client
    assert _post(c, []).status_code == 200


def test_a_trigger_message_is_accepted_and_left_queued_for_the_worker(client) -> None:
    """The webhook only enqueues. Conversion needs the pod (the rewriter is
    gpt-oss-20b on it), so it happens in the worker's prepare phase -- this
    process must never try, and the row stays `queued`, not `parsed`."""
    c, queue, replier = client
    assert _post(c, [_event("/影片 一隻橘貓走在雨中")]).status_code == 200
    assert queue.counts() == {"queued": 1}
    assert "想查進度可以看" in replier.sent[0][1][0]
    assert queue.recent()[0].prompt is None


# --------------------------------------------------------------- status page


def test_the_status_page_renders_for_a_known_token(client) -> None:
    c, queue, _ = client
    _post(c, [_event("/影片 一隻貓")])
    token = queue.recent()[0].token

    response = c.get(f"/q/{token}")
    assert response.status_code == 200
    assert "等待生成" in response.text
    assert "整理成模型看得懂的提示" in response.text, "queued wording says what the pod will do first"
    assert "一隻貓" in response.text


def test_an_unknown_token_is_404_not_an_error_page(client) -> None:
    c, _, _ = client
    assert c.get("/q/does-not-exist").status_code == 404


def test_a_finished_job_shows_a_download_link_and_the_shot_breakdown(client) -> None:
    c, queue, _ = client
    _post(c, [_event("/影片 一隻貓")])
    job = queue.recent()[0]
    queue.set_parsed(job.id, PROMPT)
    queue.claim_next(gpu_tier="L40S/COMMUNITY")
    queue.complete(job.id, "files/out.mp4")

    body = c.get(f"/q/{job.token}").text
    assert "下載影片" in body
    assert "/files/out.mp4" in body
    assert "L40S/COMMUNITY" in body, "which tier served it must be visible"
    assert "一隻橘貓走在雨中" in body or "解析出的分鏡" in body


def test_a_finished_image_job_shows_an_img_tag_not_a_broken_video_tag(client) -> None:
    c, queue, _ = client
    _post(c, [_event("/圖片 一隻貓")])
    job = queue.recent()[0]
    queue.set_parsed(job.id, {"_rendered": "a cat"})
    queue.claim_next(gpu_tier="L40S/COMMUNITY")
    queue.complete(job.id, "files/out.png")

    body = c.get(f"/q/{job.token}").text
    assert "下載圖片" in body
    assert "<img" in body
    assert "<video" not in body
    assert "/files/out.png" in body


def test_the_page_escapes_user_text(client) -> None:
    """Prompts come from strangers in a group chat."""
    c, queue, _ = client
    _post(c, [_event("/影片 <script>alert(1)</script>")])
    token = queue.recent()[0].token

    body = c.get(f"/q/{token}").text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


# ------------------------------------------------------------------- /files


def test_a_stored_file_downloads(client, tmp_path: Path) -> None:
    c, _, _ = client
    (tmp_path / "files").mkdir(exist_ok=True)
    (tmp_path / "files" / "clip.mp4").write_bytes(b"\x00\x00\x00 ftypisom")

    response = c.get("/files/clip.mp4")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"


def test_a_stored_image_downloads_with_the_right_content_type(client, tmp_path: Path) -> None:
    """Regression: this route used to hardcode video/mp4 for every file, which
    would mislabel an image the moment Flux jobs landed in the same directory."""
    c, _, _ = client
    (tmp_path / "files").mkdir(exist_ok=True)
    (tmp_path / "files" / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    response = c.get("/files/pic.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    "name", ["../.env", "..\\.env", "sub/dir.mp4", ".hidden", "../../etc/passwd"]
)
def test_path_traversal_is_refused(client, name: str) -> None:
    c, _, _ = client
    assert c.get(f"/files/{name}").status_code in (400, 404)


def test_a_missing_file_is_404(client) -> None:
    c, _, _ = client
    assert c.get("/files/nope.mp4").status_code == 404


# ------------------------------------------------------------------ healthz


def test_healthz_reports_the_queue(client) -> None:
    c, _, _ = client
    _post(c, [_event("/影片 一隻貓")])

    payload = c.get("/healthz").json()
    assert payload["ok"] is True
    assert payload["queue"] == {"queued": 1}  # conversion happens in the worker now


# ------------------------------------------------------------- capture mode


def test_capture_mode_reports_the_group_id_and_queues_nothing(tmp_path: Path) -> None:
    """The shipping default: no allowlist yet, so accept no work."""
    app, queue, replier = _build(tmp_path, allowed_group=None)
    with TestClient(app) as c:
        assert _post(c, [_event("/影片 一隻貓")]).status_code == 200
    assert queue.counts() == {}
    assert GROUP in replier.sent[-1][1][0]
    queue.close()


# ----------------------------------------------------------------- logging


def test_a_rejected_signature_is_logged_at_warning(client, caplog) -> None:
    """The one failure that must never be silent.

    LINE suspends delivery to a bot that keeps failing, so a wrong channel
    secret has to be visible in `journalctl -u ai-studio` without a debugger.
    """
    c, _, _ = client
    with caplog.at_level(logging.WARNING, logger="ai_studio.webhook"):
        assert c.post(
            "/callback",
            content=b'{"events":[]}',
            headers={"x-line-signature": "wrong"},
        ).status_code == 400
    assert any("REJECTED" in r.message for r in caplog.records)


def test_an_accepted_event_logs_its_token(client, caplog) -> None:
    """The status-page token is the only handle on a request, so the log line
    has to carry it: without it a support question cannot be traced to a job."""
    c, _, replier = client
    with caplog.at_level(logging.INFO, logger="ai_studio.webhook"):
        _post(c, [_event("/影片 一隻貓")])

    # The reply carries the status URL, so the token it ends with is the same
    # handle the user holds -- the log line has to name that exact one.
    token = replier.sent[-1][1][0].rstrip("/").rsplit("/", 1)[-1]
    line = " ".join(r.getMessage() for r in caplog.records)
    assert "accepted" in line
    assert token and token in line
    # documented in docs/line-bot.md as the way to build LINE_ALLOWED_USER_IDS
    assert "user=" in line


def test_the_verify_ping_is_logged_as_such(client, caplog) -> None:
    """`events: []` is the console's Verify button, not an error."""
    c, _, _ = client
    with caplog.at_level(logging.INFO, logger="ai_studio.webhook"):
        _post(c, [])
    assert any("no events" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------- file delivery

# LINE's video message object requires the host to answer HTTP range requests.
# Nothing about the failure says so: the object is accepted, and the video
# simply never plays. These pin the behaviour so a Starlette upgrade cannot
# take it away quietly.


@pytest.fixture
def files_client(tmp_path: Path):
    """An app with no webhook wiring — these routes need neither."""
    files_dir = tmp_path / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    queue = JobQueue(tmp_path / "files.sqlite3")
    with TestClient(create_app(queue=queue, files_dir=files_dir)) as c:
        yield c, files_dir
    queue.close()


def test_files_answers_a_range_request_with_206(files_client) -> None:
    """A LINE video message fails in a very hard-to-trace way without this."""
    client, files_dir = files_client
    (files_dir / "clip.mp4").write_bytes(bytes(range(256)) * 4)

    response = client.get("/files/clip.mp4", headers={"Range": "bytes=0-99"})

    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 0-99/1024"
    assert len(response.content) == 100


def test_files_advertises_range_support(files_client) -> None:
    client, files_dir = files_client
    (files_dir / "clip.mp4").write_bytes(b"0" * 512)

    response = client.get("/files/clip.mp4")

    assert response.status_code == 200
    assert response.headers.get("accept-ranges") == "bytes"


def test_a_range_past_the_end_is_refused_not_truncated(files_client) -> None:
    """416 rather than a short 206: a player handed silently truncated data
    reports a corrupt file, which is a much longer trail to follow."""
    client, files_dir = files_client
    (files_dir / "clip.mp4").write_bytes(b"0" * 100)

    response = client.get("/files/clip.mp4", headers={"Range": "bytes=500-999"})

    assert response.status_code == 416


def test_a_poster_is_served_with_an_image_content_type(files_client) -> None:
    """`previewImageUrl` has to arrive as an image; the mp4 and its poster live
    in the same directory and are told apart only by extension."""
    client, files_dir = files_client
    (files_dir / "clip_poster.jpg").write_bytes(b"\xff\xd8\xff\xe0")

    response = client.get("/files/clip_poster.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


# ------------------------------------------------------------ memberJoined

# The second WARNING line ("no LINE_ALLOWED_USER_IDS set: they can trigger a
# render now") lives in the FastAPI route, not in `WebhookHandler` — so the
# handler-level tests in `test_line_webhook.py` cannot reach it, and it went
# unasserted. A join is the moment the set of people who can spend GPU time
# changes, and nothing here polls the roster, so this log line is the only
# notice that change produces.


def _member_joined(group: str = GROUP, event_id: str = "evt-join") -> dict:
    return {
        "type": "memberJoined",
        "mode": "active",
        "webhookEventId": event_id,
        "timestamp": 1700000000000,
        "source": {"type": "group", "groupId": group},
        "joined": {"members": [{"type": "user", "userId": "U" + "7" * 32}]},
    }


def _app_with_users(tmp_path: Path, users):
    queue = JobQueue(tmp_path / "join.sqlite3")
    handler = WebhookHandler(
        queue,
        NullReplyClient(),
        channel_secret=SECRET,
        allowed_group_id=GROUP,
        allowed_user_ids=users,
        base_url="https://vg.example.com",
    )
    return create_app(queue=queue, handler=handler, files_dir=tmp_path / "files"), queue


def test_an_open_allowlist_warns_twice_when_someone_joins(
    tmp_path: Path, caplog
) -> None:
    """With no user allowlist, the newcomer can trigger a render immediately.
    That is the whole reason the second line exists."""
    app, queue = _app_with_users(tmp_path, ())
    with TestClient(app) as c, caplog.at_level(logging.WARNING, logger="ai_studio.webhook"):
        assert _post(c, [_member_joined()]).status_code == 200

    messages = [r.getMessage() for r in caplog.records]
    assert any("JOINED" in m for m in messages)
    assert any("LINE_ALLOWED_USER_IDS" in m for m in messages)
    queue.close()


def test_a_closed_allowlist_warns_once(tmp_path: Path, caplog) -> None:
    """The join is still worth a line — the roster changed — but the newcomer
    cannot spend anything, so the second line would be false."""
    app, queue = _app_with_users(tmp_path, ("U" + "1" * 32,))
    with TestClient(app) as c, caplog.at_level(logging.WARNING, logger="ai_studio.webhook"):
        assert _post(c, [_member_joined()]).status_code == 200

    messages = [r.getMessage() for r in caplog.records]
    assert any("JOINED" in m for m in messages)
    assert not any("LINE_ALLOWED_USER_IDS" in m for m in messages)
    queue.close()


def test_a_join_in_another_group_warns_about_nothing(tmp_path: Path, caplog) -> None:
    """Any account can add this bot to any group. Only the roster of the group
    actually served is worth a line."""
    app, queue = _app_with_users(tmp_path, ())
    with TestClient(app) as c, caplog.at_level(logging.WARNING, logger="ai_studio.webhook"):
        assert _post(c, [_member_joined(group="C" + "f" * 32)]).status_code == 200

    messages = [r.getMessage() for r in caplog.records]
    assert not any("JOINED" in m for m in messages)
    queue.close()


def test_the_page_names_the_open_model_with_a_link(client) -> None:
    """Asked 2026-08-27: every job page says which open model produced it and
    links to its repo. Generators come from a fixed table, understanding and
    chat from the provider's capabilities, so a model swap shows up here."""
    from fun_workflow.api.main import model_for
    from fun_workflow.core.kinds import JobKind

    c, queue, _ = client
    _post(c, [_event("/影片 一隻貓")])
    job = queue.recent()[0]
    body = c.get(f"/q/{job.token}").text
    assert "開源模型" in body
    assert 'href="https://huggingface.co/Comfy-Org/MiniMax-H3"' in body
    assert "專案 REPO" in body
    assert 'href="https://github.com/jieyao-MilestoneHub/ai-studio"' in body

    for kind in JobKind:
        name, url = model_for(kind)
        assert name and url.startswith("https://huggingface.co/")
    assert model_for(JobKind.CHAT)[1].endswith("/openai/gpt-oss-20b")
    assert "Qwen2-Audio" in model_for(JobKind.AUDIO_UNDERSTAND)[0]


def test_the_running_wording_matches_the_kind_of_job(client) -> None:
    """A chat or a photo-description page must not say 正在算圖 (asked 2026-08-27)."""
    from fun_workflow.api.main import state_text
    from fun_workflow.core.kinds import JobKind
    from fun_workflow.pipeline.queue import Job, JobState

    def running(kind):
        return Job(
            id=1, token="t", event_id="e", group_id="g", user_id=None, text="x",
            state=JobState.RUNNING, media_kind=kind, first_frame_path=None, quote_token=None,
            message_id=None, reply_message_id=None, requested_seconds=None,
            input_media_path=None, prompt_json=None, output_path=None, result_text=None,
            cost_usd=None, error=None, gpu_tier=None, gpu_usd_per_hr=None, created_at=0.0, parsed_at=None,
            started_at=0.0, finished_at=None, delivered_at=None, attempts=1,
        )

    seen = set()
    for kind in JobKind:
        label, note = state_text(running(kind))
        assert label and note
        seen.add(note)
    assert len(seen) == len(list(JobKind)), "every kind gets its own sentence"
    assert "算圖" not in state_text(running(JobKind.CHAT))[1]
    assert "算圖" not in state_text(running(JobKind.IMAGE_UNDERSTAND))[1]
    assert "算影片" in state_text(running(JobKind.VIDEO))[1]
