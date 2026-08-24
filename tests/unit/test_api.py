"""The always-on HTTP surface.

Dependencies are injected so these run with no LINE credentials and no network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videogen.api.main import create_app
from videogen.bots.line.reply import NullReplyClient
from videogen.bots.line.verify import sign
from videogen.bots.line.webhook import WebhookHandler
from videogen.pipeline.queue import JobQueue

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


def test_a_trigger_message_is_accepted_and_converted_in_the_background(client) -> None:
    """The 200 goes out first; conversion runs after, so it reaches `parsed`.

    With no LLM configured that conversion is the template fallback, which is
    exactly the behaviour that keeps the pipeline runnable with no LLM at all.
    """
    c, queue, replier = client
    assert _post(c, [_event("生成 一隻橘貓走在雨中")]).status_code == 200
    assert queue.counts() == {"parsed": 1}
    assert "排隊第 1 位" in replier.sent[0][1][0]

    job = queue.recent()[0]
    assert job.prompt is not None
    assert job.prompt["_built_by"] == "template"
    assert job.prompt["_rendered"].startswith("integrated_multimodal_description:")


# --------------------------------------------------------------- status page


def test_the_status_page_renders_for_a_known_token(client) -> None:
    c, queue, _ = client
    _post(c, [_event("生成 一隻貓")])
    token = queue.recent()[0].token

    response = c.get(f"/q/{token}")
    assert response.status_code == 200
    assert "等待生成" in response.text, "conversion has already run"
    assert "一隻貓" in response.text


def test_an_unknown_token_is_404_not_an_error_page(client) -> None:
    c, _, _ = client
    assert c.get("/q/does-not-exist").status_code == 404


def test_a_finished_job_shows_a_download_link_and_the_shot_breakdown(client) -> None:
    c, queue, _ = client
    _post(c, [_event("生成 一隻貓")])
    job = queue.recent()[0]
    queue.set_parsed(job.id, PROMPT)
    queue.claim_next(gpu_tier="L40S/COMMUNITY")
    queue.complete(job.id, "files/out.mp4")

    body = c.get(f"/q/{job.token}").text
    assert "下載影片" in body
    assert "/files/out.mp4" in body
    assert "L40S/COMMUNITY" in body, "which tier served it must be visible"
    assert "一隻橘貓走在雨中" in body or "解析出的分鏡" in body


def test_the_page_escapes_user_text(client) -> None:
    """Prompts come from strangers in a group chat."""
    c, queue, _ = client
    _post(c, [_event("生成 <script>alert(1)</script>")])
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
    _post(c, [_event("生成 一隻貓")])

    payload = c.get("/healthz").json()
    assert payload["ok"] is True
    assert payload["queue"] == {"parsed": 1}


# ------------------------------------------------------------- capture mode


def test_capture_mode_reports_the_group_id_and_queues_nothing(tmp_path: Path) -> None:
    """The shipping default: no allowlist yet, so accept no work."""
    app, queue, replier = _build(tmp_path, allowed_group=None)
    with TestClient(app) as c:
        assert _post(c, [_event("生成 一隻貓")]).status_code == 200
    assert queue.counts() == {}
    assert GROUP in replier.sent[-1][1][0]
    queue.close()


# ----------------------------------------------------------------- logging


def test_a_rejected_signature_is_logged_at_warning(client, caplog) -> None:
    """The one failure that must never be silent.

    LINE suspends delivery to a bot that keeps failing, so a wrong channel
    secret has to be visible in `journalctl -u videogen` without a debugger.
    """
    c, _, _ = client
    with caplog.at_level(logging.WARNING, logger="videogen.webhook"):
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
    with caplog.at_level(logging.INFO, logger="videogen.webhook"):
        _post(c, [_event("生成 一隻貓")])

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
    with caplog.at_level(logging.INFO, logger="videogen.webhook"):
        _post(c, [])
    assert any("no events" in r.getMessage() for r in caplog.records)
