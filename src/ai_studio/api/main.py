"""The always-on service: LINE webhook, status page, file delivery.

This runs 24/7 on a small VPS while the GPU pod exists for two hours a day. It
cannot scale to zero, because **LINE requires a 200 within two seconds** and a
cold start does not reliably fit in that budget.

Route budget discipline:

- `POST /callback` does an HMAC, a few comparisons, one SQLite insert and one
  reply. Nothing else. The LLM conversion runs in a background task *after* the
  response is sent; the GPU render happens hours later in the window.
- `GET /q/{token}` and `GET /files/{name}` are plain reads.

Because results are delivered as a link rather than a LINE video message, the
file route has none of the constraints a video message object imposes — no HTTP
range-request support, no 200MB ceiling, no matching-aspect poster, no codec
requirements. A 5-second clip measured 0.99MB, so ~50 a month is ~50MB and a
directory on this box is the whole storage layer.

Uvicorn's own access log is off (`access_log=False` in the CLI) and replaced by
one structured line per callback, because an access log answers "was there a
request" while the only operational question worth asking is "what did we decide
about it". A rejected signature is logged at WARNING: repeated 400s mean the
channel secret is wrong, and LINE suspends delivery to a bot that keeps failing.
"""

from __future__ import annotations

import html
import logging
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from ai_studio.bots.line.content import LineContentClient
from ai_studio.bots.line.reply import LineReplyClient, NullReplyClient
from ai_studio.bots.line.webhook import InvalidSignature, WebhookHandler
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import MediaKind
from ai_studio.llm.endpoint import RunpodLlmClient
from ai_studio.pipeline.queue import Job, JobQueue, JobState

_log = logging.getLogger("ai_studio.webhook")

_LLM_UNSET = object()
"""Distinguishes "no override passed" from "explicitly no LLM" for `llm=`.

`None` is itself a meaningful, valid value for `llm` -- it is what turns on the
template fallback -- so it cannot double as the "use settings" default without
making that fallback impossible to select on purpose from a test.
"""


def _pod_is_warm() -> bool:
    """A pod is open and inside its lease. Read per request, not cached: the
    reaper can close it between two messages."""
    from ai_studio.runtime import session as sess

    live = sess.load_state()
    return live is not None and not live.past_window()


def create_app(
    *,
    queue: JobQueue | None = None,
    handler: WebhookHandler | None = None,
    files_dir: Path | None = None,
    llm: RunpodLlmClient | None = _LLM_UNSET,  # type: ignore[assignment]
) -> FastAPI:
    """Build the app. Dependencies are injectable so tests need no credentials."""
    settings = get_settings()
    app = FastAPI(title="ai-studio", docs_url=None, redoc_url=None)

    app.state.queue = queue or JobQueue()
    # settings.files_dir, not a literal: drain writes the finished mp4 to
    # settings.files_dir, and if these two disagree the clip renders fine and
    # then 404s at the moment of delivery. AI_STUDIO_FILES_DIR must move both.
    app.state.files_dir = Path(files_dir or settings.files_dir)
    app.state.files_dir.mkdir(parents=True, exist_ok=True)

    # `_convert_later` reads `app.state.llm`; unset, it falls back to the
    # template prompt (a working path, just the 26.0-quality one, not the
    # 367.6 structured one). Both credentials have to be present for a call
    # that can actually authenticate.
    if llm is _LLM_UNSET:
        llm = (
            RunpodLlmClient(
                settings.llm_endpoint_id,
                settings.runpod_api_key.get_secret_value(),
                model=settings.llm_model,
            )
            if settings.llm_endpoint_id and settings.runpod_api_key
            else None
        )
    app.state.llm = llm

    if handler is None:
        secret = settings.line_channel_secret
        token = settings.line_channel_access_token
        replier: Any = (
            LineReplyClient(token.get_secret_value()) if token else NullReplyClient()
        )
        # Same token as the reply/push clients -- LINE's Content API is just
        # another endpoint under the one channel access token, not a
        # separate credential. None (not a null client) is "image-to-video is
        # off": a photo with nothing able to fetch it behind it must not
        # pretend to cache one.
        content = LineContentClient(token.get_secret_value()) if token else None
        handler = WebhookHandler(
            app.state.queue,
            replier,
            channel_secret=secret.get_secret_value() if secret else "",
            allowed_group_id=settings.line_allowed_group_id,
            allowed_user_ids=settings.allowed_users,
            base_url=settings.public_base_url,
            # A cap the handler defaults to *off*: this is the composition
            # root, so a guard that is configured but never passed in here is
            # a guard that does not exist. See tests/unit/test_drain_wiring.py
            # for the last time that distinction cost this project something.
            max_jobs_per_user_per_day=settings.max_jobs_per_user_per_day,
            max_audio_understand_s=settings.max_audio_understand_s,
            max_video_understand_s=settings.max_video_understand_s,
            content=content,
            incoming_dir=Path("incoming"),
            is_warm=_pod_is_warm,
        )
    app.state.handler = handler

    # ------------------------------------------------------------- webhook

    @app.post("/callback")
    async def callback(request: Request, background: BackgroundTasks) -> PlainTextResponse:
        # Raw bytes: the signature is over exactly what was sent. Parsing first
        # and re-encoding would change the body and never verify.
        body = await request.body()
        signature = request.headers.get("x-line-signature")

        try:
            outcomes = await app.state.handler.handle(body, signature)
        except InvalidSignature as exc:
            # Loud on purpose. LINE stops delivering to a bot that keeps
            # rejecting, so a silent 400 is the one failure you must not miss.
            _log.warning("callback REJECTED: %s (%d bytes)", exc, len(body))
            raise HTTPException(status_code=400, detail=str(exc)) from None

        if not outcomes:
            # The console's Verify button sends {"destination":..,"events":[]}.
            _log.info("callback ok: no events (verify ping or non-message)")
        else:
            _log.info("callback ok: %s", ", ".join(_summarise(o) for o in outcomes))

        # Conversion runs after the response goes out, so it cannot eat the
        # two-second budget however slow the LLM endpoint's cold start is.
        for outcome in outcomes:
            if outcome.action == "accepted" and outcome.job is not None:
                background.add_task(_convert_later, app, outcome.job.id)
            elif outcome.action == "memberJoined":
                # The set of people who can spend GPU time just changed.
                # Nothing here polls the roster, so this is the only notice.
                _log.warning("member(s) JOINED the group: %s", outcome.detail)
                if not app.state.handler.allowed_user_ids:
                    _log.warning(
                        "  no LINE_ALLOWED_USER_IDS set: they can trigger a render now"
                    )
            elif outcome.action == "capture" and outcome.detail.startswith("C"):
                # There is no API to list a bot's groups, so this line is the
                # only way to learn the id. Make it impossible to miss.
                print(
                    "\n" + "=" * 62
                    + f"\n  GROUP ID: {outcome.detail}\n"
                    + "  Put this in .env as LINE_ALLOWED_GROUP_ID and restart.\n"
                    + "=" * 62 + "\n",
                    flush=True,
                )

        return PlainTextResponse("OK")

    # --------------------------------------------------------- status page

    @app.get("/q/{token}", response_class=HTMLResponse)
    async def status_page(token: str) -> HTMLResponse:
        job = app.state.queue.by_token(token)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown request")
        position = app.state.queue.position(token)
        return HTMLResponse(_render(job, position, settings.public_base_url))

    @app.get("/files/{name}")
    async def download(name: str) -> FileResponse:
        # Reject anything that could climb out of the directory.
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="bad filename")
        path = app.state.files_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        # Both mp4 clips and png images live in this one directory now.
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=name)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "queue": app.state.queue.counts()}

    return app


def _summarise(outcome: Any) -> str:
    """One outcome as a short ASCII phrase. The Windows console is cp950 and
    throws on anything else, so this stays ASCII even though prompts are not."""
    parts = [outcome.action]
    if outcome.job is not None:
        parts.append(f"job={outcome.job.id} token={outcome.job.token}")
        # The id is here so the log alone is enough to build
        # LINE_ALLOWED_USER_IDS from people who have actually used the bot.
        parts.append(f"user={outcome.job.user_id}")
    if outcome.detail:
        parts.append(f"({outcome.detail})")
    return " ".join(parts)


async def _convert_later(app: FastAPI, job_id: int) -> None:
    """Turn a queued request into a validated H3 prompt.

    Imported lazily and failure-tolerant: a conversion problem must leave the
    request in the queue for a retry, not take down the web process.
    """
    try:
        from ai_studio.pipeline.convert_worker import convert_job

        await convert_job(app.state.queue, job_id, getattr(app.state, "llm", None))
    except Exception:  # background task: the job stays queued for a retry
        return


# ----------------------------------------------------------------- rendering

_STATE_ZH = {
    JobState.QUEUED: ("解析中", "正在把你的描述轉成模型看得懂的分鏡"),
    JobState.PARSED: ("等待生成", "已排入佇列,GPU 開機中或排隊中"),
    JobState.RUNNING: ("生成中", "正在算圖,約 5 分鐘"),
    JobState.DONE: ("完成", ""),
    JobState.FAILED: ("失敗", ""),
}


def _render(job: Job, position: int | None, base_url: str) -> str:
    label, note = _STATE_ZH[job.state]
    rows = [("狀態", label), ("你的描述", job.text)]
    if position:
        rows.append(("佇列位次", f"第 {position} 位"))
    if job.gpu_tier:
        rows.append(("使用的 GPU", job.gpu_tier))
    if job.error:
        rows.append(("錯誤", job.error[:300]))

    body = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )

    action = ""
    if job.state is JobState.DONE and job.output_path:
        name = Path(job.output_path).name
        url = f"{base_url.rstrip('/')}/files/{name}"
        if job.media_kind is MediaKind.IMAGE:
            action = (
                f'<p class="cta"><a href="{html.escape(url)}" download>下載圖片</a></p>'
                f'<img src="{html.escape(url)}" alt="生成結果">'
            )
        else:
            action = (
                f'<p class="cta"><a href="{html.escape(url)}" download>下載影片 (mp4)</a></p>'
                f'<video controls preload="metadata" src="{html.escape(url)}"></video>'
            )
    elif job.state is JobState.DONE and job.result_text:
        # An understanding job has no output file at all -- the result is
        # text, rendered directly rather than assumed to be a media link.
        action = f'<p class="result">{html.escape(job.result_text)}</p>'
    elif note:
        action = f'<p class="note">{html.escape(note)}</p>'

    plan = job.prompt or {}
    shots = plan.get("shots") or []
    breakdown = ""
    if shots:
        items = "".join(
            f"<li><b>鏡頭 {s.get('index', i + 1)}</b> {html.escape(str(s.get('description', ''))[:180])}</li>"
            for i, s in enumerate(shots)
        )
        breakdown = f"<h2>解析出的分鏡</h2><ol>{items}</ol>"

    return f"""<!doctype html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ai-studio · {html.escape(label)}</title>
<style>
:root{{color-scheme:light dark}}
body{{font:16px/1.6 system-ui,-apple-system,"Noto Sans TC",sans-serif;
     max-width:44rem;margin:0 auto;padding:1.5rem}}
h1{{font-size:1.3rem;margin:0 0 1rem}}
table{{width:100%;border-collapse:collapse;margin-bottom:1rem}}
th,td{{text-align:left;padding:.5rem .25rem;border-bottom:1px solid color-mix(in srgb,currentColor 15%,transparent);vertical-align:top}}
th{{width:8rem;font-weight:600;opacity:.7}}
.cta a{{display:inline-block;padding:.7rem 1.2rem;border-radius:.5rem;
       background:#2563eb;color:#fff;text-decoration:none;font-weight:600}}
video,img{{width:100%;border-radius:.5rem;margin-top:1rem;background:#000}}
.note{{opacity:.7}}
ol{{padding-left:1.2rem}} li{{margin:.4rem 0}}
</style></head>
<body>
<h1>ai_studio · {html.escape(label)}</h1>
<table>{body}</table>
{action}
{breakdown}
</body></html>"""
