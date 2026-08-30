"""The request side's pre-launch checklist: everything about taking a
request and answering it that can be proved with no pod and no money.

Built on `ai_studio.checks` -- same result type, same rules (a check
that cannot run is SKIP, never PASS; green means green) -- with the LINE
half of the list: the deployed secret, dedupe, the queue -> rewrite path,
accept-and-hold, the push client, and range requests on /files. The GPU
half (graphs, poster, placement) is `ai-studio preflight`.

Only check 5 sends anything to a real person, which is why it is opt-in
behind a flag rather than merely credential-gated.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_studio import paths
from ai_studio.checks import (
    CheckResult,
    check_offline_suite,
    fail,
    passed,
    run_checks,
    skip,
)

from fun_workflow.config.settings import get_fun_settings

SYNTHETIC_SECRET = "preflight-synthetic-channel-secret"
"""Used only where the subject of the check is not the credential itself.

Check 2 exists to prove the *deployed* secret works, so it skips without one.
Check 4 is about the clock, so a synthetic secret there proves what it claims
to prove.
"""


# ------------------------------------------------------------------ helpers


def _test_client(handler: Any = None, files_dir: Path | None = None) -> Any:
    """An in-process app. Raises `ImportError` if the web extra is absent.

    In-process on purpose: these checks must work before the service is up, and
    a check that needs the thing it is checking already running is not a
    pre-launch check.
    """
    from fastapi.testclient import TestClient

    from fun_workflow.api.main import create_app
    from fun_workflow.pipeline.queue import JobQueue

    root = Path(tempfile.mkdtemp(prefix="ai-studio-preflight-"))
    queue = JobQueue(root / "queue.sqlite3")
    app = create_app(
        queue=queue, handler=handler, files_dir=files_dir or (root / "files")
    )
    return TestClient(app), queue, root


def _handler(queue: Any, *, secret: str, clock: Any = None) -> Any:
    from fun_workflow.bots.line.reply import NullReplyClient
    from fun_workflow.bots.line.webhook import WebhookHandler

    settings = get_fun_settings()
    return WebhookHandler(
        queue,
        NullReplyClient(),
        channel_secret=secret,
        allowed_group_id=settings.line_allowed_group_id or "Cpreflight",
        base_url=settings.public_base_url,
        clock=clock,
    )


def _event(text: str, *, event_id: str, group: str, user: str = "U" + "1" * 32) -> dict[str, Any]:
    return {
        "type": "message",
        "mode": "active",
        "webhookEventId": event_id,
        "timestamp": 1700000000000,
        "replyToken": "rt-" + event_id,
        "source": {"type": "group", "groupId": group, "userId": user},
        "message": {"type": "text", "id": "m1", "text": text},
    }


def _signed(events: list[dict[str, Any]], secret: str) -> tuple[bytes, str]:
    from fun_workflow.bots.line.verify import sign

    body = json.dumps({"destination": "U" + "0" * 32, "events": events}).encode("utf-8")
    return body, sign(body, secret)


# ------------------------------------------------------------------- checks


def check_signature_and_dedupe() -> CheckResult:
    """2. A signed event is accepted, a tampered one is not, a replay is not billed.

    Skips without a configured secret rather than falling back to a synthetic
    one: the thing under test here is the credential the service will actually
    run with. If this is broken, nothing a user types in Phase 7 reaches the
    queue at all.
    """
    name = "webhook signature and event dedupe"
    settings = get_fun_settings()
    secret = settings.line_channel_secret
    if secret is None:
        return skip(2, name, "LINE_CHANNEL_SECRET is not set")

    try:
        client, queue, _ = _test_client()
    except ImportError as exc:
        return skip(2, name, f"the web extra is not installed: {exc}")

    try:
        value = secret.get_secret_value()
        group = settings.line_allowed_group_id or "Cpreflight"
        client.app.state.handler = _handler(queue, secret=value)

        body, signature = _signed([_event("/影片 preflight", event_id="pf-1", group=group)], value)
        good = client.post("/callback", content=body, headers={"x-line-signature": signature})
        if good.status_code != 200:
            return fail(2, name, f"a correctly signed event got {good.status_code}")

        tampered = client.post(
            "/callback", content=body + b" ", headers={"x-line-signature": signature}
        )
        if tampered.status_code != 400:
            return fail(2, name, f"a tampered body got {tampered.status_code}, expected 400")

        client.post("/callback", content=body, headers={"x-line-signature": signature})
        jobs = queue.counts()
        total = sum(jobs.values())
        if total != 1:
            return fail(2, name, f"a redelivered event produced {total} jobs, expected 1")
        return passed(2, name, "200 signed, 400 tampered, redelivery deduped")
    finally:
        queue.close()


def check_queue_and_conversion() -> CheckResult:
    """3. A queued Chinese request becomes a claimable English prompt.

    Offline on purpose: the real rewriter is gpt-oss-20b on the pod, which
    does not exist before a window opens, so this proves the queue -> prompt
    -> `_rendered` path with a scripted reply and no network. The live check
    is `ai-studio rewrite --kind image "..."` against an open pod.
    """
    name = "queue and prompt conversion (offline)"
    import asyncio

    from ai_studio.llm.scripted import ScriptedLlmClient

    from fun_workflow.pipeline.convert_worker import convert_job
    from fun_workflow.pipeline.queue import JobQueue

    root = Path(tempfile.mkdtemp(prefix="ai-studio-preflight-"))
    queue = JobQueue(root / "queue.sqlite3")
    try:
        job, _ = queue.enqueue(
            "pf-convert", "Cpreflight", "一隻橘貓坐在窗邊看雨",
            user_id="Upreflight", media_kind=_image_kind(),
        )
        client = ScriptedLlmClient(
            '{"prompt": "An orange tabby cat sits on a windowsill watching the rain, '
            'soft grey daylight from the window, wet glass, wooden frame"}'
        )
        how = asyncio.run(convert_job(queue, job.id, client, prompt_mode="structured"))
        stored = queue.by_id(job.id)
        if stored is None or stored.prompt is None:
            return fail(3, name, "the job was never parsed")
        rendered = str(stored.prompt.get("_rendered", ""))
        if not rendered.isascii():
            return fail(3, name, f"the prompt is not English ({how}): {rendered[:60]}")
        return passed(3, name, f"built_by={how}, rendered={rendered[:60]}")
    except Exception as exc:
        return fail(3, name, f"{type(exc).__name__}: {exc}")
    finally:
        queue.close()


def _image_kind() -> Any:
    from fun_workflow.core.kinds import JobKind

    return JobKind.IMAGE


def check_out_of_hours() -> CheckResult:
    """4. A request at any hour is accepted, answered, and opens nothing itself.

    There are no business hours any more; what this pins is that the *web*
    path never creates a pod. Only the worker does, on its own tick, behind
    the budget guard and the daily cap -- so a webhook that is up while the
    worker is down accepts and holds, and nothing bills.
    """
    name = "any hour: accepted, answered, nothing opened by the webhook"
    try:
        client, queue, _ = _test_client()
    except ImportError as exc:
        return skip(4, name, f"the web extra is not installed: {exc}")

    from ai_studio.runtime import session as sess

    try:
        settings = get_fun_settings()
        group = settings.line_allowed_group_id or "Cpreflight"
        night = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
        state_before = sess.load_state()

        handler = _handler(queue, secret=SYNTHETIC_SECRET, clock=lambda: night)
        client.app.state.handler = handler
        body, signature = _signed(
            [_event("/影片 preflight", event_id="pf-shut", group=group)], SYNTHETIC_SECRET
        )
        response = client.post("/callback", content=body, headers={"x-line-signature": signature})
        if response.status_code != 200:
            return fail(4, name, f"the request got {response.status_code}")

        if sum(queue.counts().values()) != 1:
            return fail(4, name, "the request was refused instead of held for the worker")

        sent = handler.replier.sent
        if not sent:
            return fail(4, name, "the request was accepted with no reply at all")
        text = sent[-1][1][0]
        if "/q/" not in text:
            return fail(4, name, f"the reply carries no status-page link: {text[:80]}")

        state_after = sess.load_state()
        if (state_after is not None) != (state_before is not None) or (
            state_after is not None and state_before is not None
            and state_after.pod_id != state_before.pod_id
        ):
            return fail(4, name, "the webhook path changed the pod session state")
    except Exception as exc:
        return fail(4, name, f"{type(exc).__name__}: {exc}")
    return passed(4, name, "accepted and held (1 waiting), reply carries a status link, no pod created")


def check_push(*, send: bool = False) -> CheckResult:
    """5. A real push reaches the real group, and the mention resolves.

    Opt-in behind `--push`, not merely credential-gated: this is the one check
    that sends a message to actual people and spends actual quota. Note how
    many messages it consumes — that number is the open input in PLAN.md §4.
    """
    name = "push into the real group (with mention)"
    settings = get_fun_settings()
    if not send:
        return skip(5, name, "not attempted; pass --push to send a real message")
    token = settings.line_channel_access_token
    if token is None:
        return skip(5, name, "LINE_CHANNEL_ACCESS_TOKEN is not set")
    if not settings.line_allowed_group_id:
        return skip(5, name, "LINE_ALLOWED_GROUP_ID is not set")

    import asyncio

    from fun_workflow.bots.line.push import LinePushClient, text_message

    client = LinePushClient(token.get_secret_value())
    # Plain text: real deliveries quote the request message, and preflight
    # has no request to quote.
    message = text_message("preflight check 5: if you can see this, push delivery works.")
    try:
        asyncio.run(client.push(settings.line_allowed_group_id, [message], retry_key="preflight"))
    except Exception as exc:
        return fail(5, name, f"{type(exc).__name__}: {exc}")
    return passed(
        5, name,
        "sent 1 text message"
        " -- now read the quota consumed off LINE Official Account Manager",
    )


def check_files_range() -> CheckResult:
    """6. `/files` answers a Range request with 206.

    LINE's video message object requires it, and nothing about the failure says
    so: the object is accepted and the video simply never plays.
    """
    name = "/files answers Range with 206"
    root = Path(tempfile.mkdtemp(prefix="ai-studio-preflight-"))
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    (files / "preflight.mp4").write_bytes(bytes(range(256)) * 4)

    try:
        client, queue, _ = _test_client(files_dir=files)
    except ImportError as exc:
        return skip(6, name, f"the web extra is not installed: {exc}")

    try:
        response = client.get("/files/preflight.mp4", headers={"Range": "bytes=0-99"})
        if response.status_code != 206:
            return fail(6, name, f"got {response.status_code}, expected 206")
        content_range = response.headers.get("content-range", "")
        if content_range != "bytes 0-99/1024":
            return fail(6, name, f"Content-Range was {content_range!r}")
        return passed(6, name, f"206 with {content_range}")
    finally:
        queue.close()




def check_caption_font() -> CheckResult:
    """7. The drama caption font resolves to a CJK face on this host.

    libass never fails on a missing font: it substitutes, and Mandarin
    captions come out as boxes in a file that has already been pushed. The
    check asks fontconfig what it would actually use.
    """
    fun = get_fun_settings()
    name = f"caption font '{fun.drama_font_name}' resolves to a CJK face"
    if fun.drama_fonts_dir is not None:
        files = [p.name for p in Path(fun.drama_fonts_dir).glob("*") if p.suffix.lower() in (".ttf", ".ttc", ".otf")]
        if not files:
            return fail(7, name, f"AI_STUDIO_DRAMA_FONTS_DIR={fun.drama_fonts_dir} holds no font files")
        return passed(7, name, f"fontsdir with {len(files)} file(s): {', '.join(files[:3])}")
    if shutil.which("fc-match") is None:
        return skip(7, name, "fc-match not on PATH (no fontconfig); set AI_STUDIO_DRAMA_FONTS_DIR")
    proc = subprocess.run(
        ["fc-match", "-f", "%{family}|%{file}", fun.drama_font_name],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30.0, check=False,
    )
    if proc.returncode != 0:
        return fail(7, name, (proc.stderr or proc.stdout).strip()[:200])
    family, _, file = proc.stdout.strip().partition("|")
    if "cjk" not in family.lower() and "cjk" not in file.lower():
        return fail(7, name, f"fontconfig substitutes {family!r} ({file}); captions would render as boxes")
    return passed(7, name, f"{family} ({Path(file).name})")


def run_all(*, run_suite: bool = True, send_push: bool = False) -> list[CheckResult]:
    """The request side's checklist. Numbered as a person reads it."""
    return run_checks([
        lambda: check_offline_suite(run=run_suite, cwd=paths.repo_root() / "fun_workflow"),
        check_signature_and_dedupe,
        check_queue_and_conversion,
        check_out_of_hours,
        lambda: check_push(send=send_push),
        check_files_range,
        check_caption_font,
    ])
