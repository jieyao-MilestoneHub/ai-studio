"""The always-on service: LINE webhook, status page, file delivery.

This runs 24/7 on a small VPS while the GPU pod exists for two hours a day. It
cannot scale to zero, because **LINE requires a 200 within two seconds** and a
cold start does not reliably fit in that budget.

Route budget discipline:

- `POST /callback` does an HMAC, a few comparisons, one SQLite insert and one
  reply. Nothing else. Prompt rewriting and the GPU render both happen in the
  worker process, on the pod, when a job is claimed.
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from ai_studio.bots.line.content import LineContentClient
from ai_studio.bots.line.reply import LineReplyClient, NullReplyClient
from ai_studio.bots.line.webhook import InvalidSignature, WebhookHandler
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import MediaKind
from ai_studio.pipeline.queue import Job, JobQueue, JobState

_log = logging.getLogger("ai_studio.webhook")

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

    # No prompt conversion here any more: the rewriter is gpt-oss-20b on the
    # pod, and the GPU worker converts each request in its prepare phase
    # (pipeline/worker.py). This process only enqueues.

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
            max_chat_messages_per_user_per_day=settings.max_chat_messages_per_user_per_day,
            max_dramas_per_day=settings.max_dramas_per_day,
            max_audio_understand_s=settings.max_audio_understand_s,
            max_video_understand_s=settings.max_video_understand_s,
            content=content,
            incoming_dir=Path("incoming"),
            is_warm=_pod_is_warm,
            # So「讓我看看」can attach the poster the worker already rendered
            # next to the clip -- same directory drain writes to.
            files_dir=app.state.files_dir,
        )
    app.state.handler = handler

    # ------------------------------------------------------------- webhook

    @app.post("/callback")
    async def callback(request: Request) -> PlainTextResponse:
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

        for outcome in outcomes:
            if outcome.action == "memberJoined":
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


# ----------------------------------------------------------------- rendering

_STATE_ZH = {
    JobState.QUEUED: ("等待生成", "已排入佇列;GPU 開機後會先把你的描述整理成模型看得懂的提示"),
    JobState.PARSED: ("等待生成", "已排入佇列,GPU 開機中或排隊中"),
    JobState.RUNNING: ("生成中", "正在算圖,約 5 分鐘"),
    JobState.DONE: ("完成", ""),
    JobState.FAILED: ("失敗", ""),
}
"""The generic (video) wording; `state_text()` specialises it per kind."""

_RUNNING_ZH: dict[MediaKind, tuple[str, str]] = {
    # (label, note) while the job is on the GPU. Durations are 📏 from the
    # RTX 4090 on 2026-08-27, cold load included, rounded up so the page
    # under-promises: image ~30 s warm; a 10 s clip 2-5 min; the three
    # understanding models and chat each take ~1 min to load if another
    # model is resident, then seconds to answer.
    MediaKind.VIDEO: ("生成中", "正在算影片,約 2 到 5 分鐘"),
    MediaKind.IMAGE: ("生成中", "正在算圖,約 30 秒到 1 分鐘"),
    MediaKind.IMAGE_UNDERSTAND: ("辨識中", "正在看這張照片,約 1 分鐘(含載入模型)"),
    MediaKind.AUDIO_UNDERSTAND: ("辨識中", "正在聽這段聲音,約 1 分鐘(含載入模型)"),
    MediaKind.VIDEO_UNDERSTAND: ("辨識中", "正在看這段影片,約 1 到 2 分鐘(含載入模型)"),
    MediaKind.CHAT: ("回覆中", "正在想怎麼回你,約 1 分鐘(含載入模型)"),
    MediaKind.DRAMA: ("製作中", "六個鏡頭慢慢做:角色定裝、每鏡首幀、每鏡影片,約 25 到 40 分鐘"),
}

_WAITING_ZH: dict[MediaKind, tuple[str, str]] = {
    MediaKind.VIDEO: ("等待生成", "已排入佇列,GPU 開機中或排隊中"),
    MediaKind.IMAGE: ("等待生成", "已排入佇列,GPU 開機中或排隊中"),
    MediaKind.IMAGE_UNDERSTAND: ("等待辨識", "已排入佇列,GPU 開機中或排隊中"),
    MediaKind.AUDIO_UNDERSTAND: ("等待辨識", "已排入佇列,GPU 開機中或排隊中"),
    MediaKind.VIDEO_UNDERSTAND: ("等待辨識", "已排入佇列,GPU 開機中或排隊中"),
    MediaKind.CHAT: ("等待回覆", "已排入佇列,GPU 開機中或排隊中"),
    MediaKind.DRAMA: ("等待製作", "已排入佇列;GPU 開機後先由 gpt-oss-20b 寫劇本,再開始生成"),
}


def state_text(job: Job) -> tuple[str, str]:
    """(label, note) for the page, worded for what the job actually is.

    "正在算圖" on a chat or a video-description page read as the wrong page
    (asked 2026-08-27). Understanding and chat jobs also skip the
    parsing step, so their queued wording never claims to be building
    shots. Unknown kind raises -- a wrong sentence is worse than none.
    """
    kind = job.media_kind
    if job.state is JobState.RUNNING:
        return _RUNNING_ZH[kind]
    if job.state is JobState.PARSED:
        return _WAITING_ZH[kind]
    if job.state is JobState.QUEUED:
        label, _ = _WAITING_ZH[kind]
        return label, _STATE_ZH[JobState.QUEUED][1]
    return _STATE_ZH[job.state]


PROJECT_REPO_URL = "https://github.com/jieyao-MilestoneHub/ai-studio"
"""Shown on every job page so a viewer can find the code that made it."""

_GENERATION_MODELS: dict[MediaKind, tuple[str, str]] = {
    # The two ComfyUI-served generators name their weights by repo, not by a
    # capabilities snapshot (their `model_id` is "<model>@<w>x<h>"), so the
    # public repo is spelled out here. The weights `deploy/pod_setup.sh`
    # actually downloads: Comfy-Org's repackaging of MiniMax H3, and the
    # fp8 Flux.1-dev transformer from Comfy-Org/flux1-dev (ungated, same
    # weights as black-forest-labs/FLUX.1-dev).
    MediaKind.VIDEO: ("MiniMax H3 (MiniMax-H3-fl2va)", "https://huggingface.co/Comfy-Org/MiniMax-H3"),
    MediaKind.IMAGE: ("Flux.1-dev", "https://huggingface.co/black-forest-labs/FLUX.1-dev"),
    # A drama is all three: gpt-oss-20b writes, Flux paints the keyframes, H3
    # animates them. The video model is the one named; the page's screenplay
    # block says the rest.
    MediaKind.DRAMA: (
        "MiniMax H3 + Flux.1-dev + gpt-oss-20b", "https://huggingface.co/Comfy-Org/MiniMax-H3",
    ),
}


def model_for(kind: MediaKind) -> tuple[str, str]:
    """The open model a job of this kind runs on, and where it lives.

    Understanding and chat read the id off the provider's capabilities so
    a model swap (2026-08-27: two of the three understanding models) shows
    up here without a second edit; the two generators are a fixed table.
    Raises on a kind nothing serves -- fail loudly, never a blank row.
    """
    if kind in _GENERATION_MODELS:
        return _GENERATION_MODELS[kind]
    if kind is MediaKind.CHAT:
        from ai_studio.providers.chat import chat_capabilities

        model_id = chat_capabilities().model_id
    elif kind.is_understanding:
        from ai_studio.providers.understanding import understanding_capabilities

        model_id = understanding_capabilities(kind).model_id
    else:
        raise ValueError(f"no model table entry for {kind!r}")
    return model_id, f"https://huggingface.co/{model_id}"


def _drama_block(job: Job, plan: dict[str, Any]) -> str:
    """The screenplay and, while it renders, per-stage progress.

    Progress is read off `runs/drama/<token>/state.json`, which
    `pipeline.drama` rewrites after every fetched artifact -- so the page
    says「首幀 4/6」rather than「生成中」for half an hour. Absent file (not
    started yet): screenplay only.
    """
    from ai_studio.core.drama_spec import SHOT_COUNT
    from ai_studio.pipeline.drama import load_state

    screenplay = plan.get("screenplay") or {}
    parts: list[str] = []
    if screenplay:
        anchor = screenplay.get("anchor") or {}
        parts.append(
            f"<h2>{html.escape(str(screenplay.get('title', '')))}</h2>"
            f"<p class=\"note\">{html.escape(str(screenplay.get('logline', '')))}</p>"
            f"<p><b>主角</b> {html.escape(str(anchor.get('name', '')))} — "
            f"{html.escape(str(anchor.get('appearance', '')))}</p>"
        )
    run_dir = get_settings().runs_dir / "drama" / job.token
    if (run_dir / "state.json").is_file() and job.state is not JobState.FAILED:
        try:
            state = load_state(run_dir)
        except Exception:  # a half-written state file must not 500 the page
            state = None
        if state is not None:
            done = "完成" if state.output else "進行中"
            parts.append(
                f"<p><b>進度</b> 角色 {len(state.character)}/2 · 首幀 {len(state.keyframes)}/{SHOT_COUNT}"
                f" · 影片 {len(state.clips)}/{SHOT_COUNT} · {done}"
                f"<br><b>臉部修復</b> {html.escape(state.face_repair)}"
                f" · <b>GPU 花費</b> ${state.spent_usd:.2f}</p>"
            )
    shots = plan.get("shots") or []
    if shots:
        items = "".join(
            f"<li><b>鏡頭 {s.get('index', i + 1)}</b> {html.escape(str(s.get('description', ''))[:220])}</li>"
            for i, s in enumerate(shots)
        )
        parts.append(f"<h2>分鏡</h2><ol>{items}</ol>")
    return "".join(parts)


def _render(job: Job, position: int | None, base_url: str) -> str:
    label, note = state_text(job)
    rows: list[tuple[str, str]] = [("狀態", html.escape(label)), ("你的描述", html.escape(job.text))]
    if position:
        rows.append(("佇列位次", f"第 {position} 位"))
    if job.gpu_tier:
        rows.append(("使用的 GPU", html.escape(job.gpu_tier)))
    if job.gpu_usd_per_hr:
        rows.append(("GPU 租用價格", f"${job.gpu_usd_per_hr:.3f}/hr"))
    model_name, model_url = model_for(job.media_kind)
    rows.append((
        "開源模型",
        f'<a href="{html.escape(model_url)}" rel="noopener">{html.escape(model_name)}</a>',
    ))
    rows.append((
        "專案 REPO",
        f'<a href="{html.escape(PROJECT_REPO_URL)}" rel="noopener">{html.escape(PROJECT_REPO_URL)}</a>',
    ))
    if job.error:
        rows.append(("錯誤", html.escape(job.error[:300])))

    # Values are already HTML (escaped text, or the anchors above).
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)

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
    if job.media_kind is MediaKind.DRAMA:
        breakdown = _drama_block(job, plan)
    elif shots:
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
