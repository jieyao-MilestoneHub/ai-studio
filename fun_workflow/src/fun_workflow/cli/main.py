"""`funapp`: the LINE group's side of the shop, composed on top of ai-studio.

Everything here either takes requests (`line serve`), does them on a pod
(`worker`, `drain`), decides when that pod may close (`reap`), or keeps the
host tidy (`gc`, `archive`). The pod itself -- opening it, placing it,
paying for it -- is ai-studio's, reached only from this module.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import typer
from ai_studio import media, paths
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import MediaKind
from ai_studio.core.errors import AIStudioError
from ai_studio.providers.registry import get_provider
from rich.console import Console
from rich.table import Table

from fun_workflow import paths as fun_paths
from fun_workflow.bots.line.limits import PREVIEW_IMAGE_MAX_BYTES, UNDERSTANDING_MAX_OUTPUT_CHARS
from fun_workflow.config.settings import get_fun_settings
from fun_workflow.core.kinds import JobKind

app = typer.Typer(
    help="The LINE group's playground: webhook, queue, GPU worker, and the daily chores.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(service: str) -> None:
    """The one place a process turns logging on: stderr for journald plus the
    JSONL trace under the shared log dir. Every command that does work calls
    it first; read-only commands do not."""
    from ai_studio.core.observability import configure_logging

    from fun_workflow.core.observability import EXTRA_FIELDS

    settings = get_settings()
    configure_logging(
        service=service, log_dir=settings.log_dir, level=settings.log_level, extra_fields=EXTRA_FIELDS,
    )


line_app = typer.Typer(help="LINE bot: serve the webhook, or discover a group id.")
app.add_typer(line_app, name="line")


def _pod_is_warm() -> bool:
    """A pod is open and inside its lease. Read per request, not cached: the
    reaper can close it between two messages."""
    from ai_studio.runtime import session as sess

    live = sess.load_state()
    return live is not None and not live.past_window()


def _run_server(host: str, port: int, reload: bool = False) -> None:
    import uvicorn

    from fun_workflow.api.main import create_app

    # log_config=None keeps uvicorn from calling dictConfig, which clears every
    # existing handler and defines no root logger - so a config set up here
    # would be silently discarded and ai-studio's own lines would never appear.
    # Owning the config instead means our INFO lines and uvicorn's both show up,
    # here and under journalctl on the host.
    _setup_logging("webhook")
    uvicorn.run(
        create_app(is_warm=_pod_is_warm),
        host=host,
        port=port,
        reload=reload,
        access_log=False,
        log_config=None,
    )


@line_app.command("serve")
def line_serve(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8000),
) -> None:
    """Run the always-on service: webhook, status pages, file downloads."""
    settings = get_fun_settings()
    if settings.line_channel_secret is None:
        console.print("[red]LINE_CHANNEL_SECRET is unset.[/red] Every webhook will 400.")
        raise typer.Exit(1)

    group = settings.line_allowed_group_id
    console.print(f"[bold]ai-studio[/bold] on {host}:{port}")
    console.print(f"  public base  {settings.public_base_url}")
    console.print(f"  webhook      {settings.public_base_url.rstrip('/')}/callback")
    if group:
        console.print(f"  serving group {group}")
        users = settings.allowed_users
        if users:
            console.print(f"  authorised    {len(users)} user(s)")
        else:
            # Not a warning about a misconfiguration - a warning about a choice.
            console.print(
                "  [yellow]any group member can spend GPU time[/yellow]: "
                "LINE_ALLOWED_USER_IDS is unset, so whoever is invited to the "
                "group next can trigger a render."
            )
    else:
        console.print(
            "  [yellow]capture mode[/yellow]: no LINE_ALLOWED_GROUP_ID set, so no "
            "work is accepted. Say the trigger word in the group and the id "
            "will be printed here."
        )
    _run_server(host, port)


@line_app.command("capture-group")
def line_capture_group(
    port: int = typer.Option(8000),
    host: str = typer.Option("0.0.0.0"),
) -> None:
    """Discover a group's id, which no API exposes.

    LINE documents that a bot cannot list the groups it belongs to, so the id
    has to be read off a live webhook event. This runs the service in capture
    mode: it answers the trigger word with the group id and accepts no work.
    """
    settings = get_fun_settings()
    if settings.line_allowed_group_id:
        console.print(
            f"[yellow]LINE_ALLOWED_GROUP_ID is already set[/yellow] "
            f"({settings.line_allowed_group_id}). "
            "Clear it in .env first, or you will not see capture output."
        )
        raise typer.Exit(1)

    console.print("[bold]capture mode[/bold] - waiting for a group message")
    console.print("  1. add the official account to the group")
    console.print("  2. point the LINE console Webhook URL at "
                  f"{settings.public_base_url.rstrip('/')}/callback")
    # The trigger is CJK and has no ASCII alias any more (one spelling per
    # trigger, see bots.line.webhook). On a cp950 console it may print as
    # mojibake; the docs carry the same string for copy-pasting.
    console.print("  3. say [cyan]/影片 test[/cyan] in the group")
    console.print("  the group id will be printed here, and replied into the chat")
    console.print("")
    _run_server(host, port)


@app.command("worker")
def worker(
    name: str = typer.Option("ai-studio-window", help="Pod name to open and reuse."),
    poll_seconds: float = typer.Option(15.0, help="How often to poll ComfyUI."),
    idle_seconds: float = typer.Option(10.0, help="Queue check interval while open."),
    max_ticks: int | None = typer.Option(None, help="Stop after N passes. For testing."),
) -> None:
    """Serve the queue: open a pod when work arrives, at any hour.

    This is what replaces the `open` and `drain` timers. It runs as a systemd
    service with `Restart=always`; with nothing queued it sleeps, so nothing
    fires when no one has asked for anything.

    Closing is still someone else's job, on purpose: `funapp reap` every
    minute, `ai-studio session close` at the bell, and `--terminate-after` on
    the pod itself. Three independent ways for a machine to stop billing, none of which
    depends on this process still being alive.
    """
    _setup_logging("worker")
    from fun_workflow.pipeline.queue import JobQueue
    from fun_workflow.pipeline.worker import serve

    settings = get_fun_settings()
    host = _RuntimeHost(name=name, poll_seconds=poll_seconds)
    queue = JobQueue()
    console.print(
        f"worker up. opens a pod on demand at any hour, "
        f"files -> {settings.files_dir}"
    )
    try:
        report = asyncio.run(
            serve(
                queue,
                cast(Any, host),
                files_dir=settings.files_dir,
                idle_poll_s=idle_seconds,
                poll_interval_s=poll_seconds,
                max_ticks=max_ticks,
            )
        )
    except KeyboardInterrupt:
        console.print("worker stopped. [dim]Any open pod is still billing: session close[/dim]")
        raise typer.Exit(0) from None
    finally:
        queue.close()
    console.print(report.summary())


@app.command("drain")
def drain(
    max_clips: int | None = typer.Option(None, help="Stop after N clips."),
    poll_seconds: float = typer.Option(15.0, help="How often to poll ComfyUI."),
) -> None:
    """Render queued requests on a pod someone already opened (`ai-studio session open`).

    The manual counterpart of `worker`: it exits immediately and successfully
    when no window is open, and never opens one itself.

    Which workflow runs is decided by the rung that answered, not by preference.
    A 48GB card takes the fp8 graph and applies the LoRA in bypass; a 24GB card
    takes int8 with the LoRA merged, which the node pack itself calls softer.
    """
    _setup_logging("drain")
    from ai_studio.inference.client import InferenceClient
    from ai_studio.pipeline.pod_llm import PodLlmClient
    from ai_studio.runtime import session as sess

    from fun_workflow.pipeline.drain import drain_window
    from fun_workflow.pipeline.queue import JobQueue

    settings = get_settings()
    fun = get_fun_settings()
    session = sess.load_state()
    if session is None:
        console.print("no window is open; nothing to drain")
        return
    if session.past_window():
        console.print("the window has already ended; not claiming new work")
        return

    workflow = paths.workflow("h3_fl2va_turbo.json" if session.low_vram else "h3_fl2va_turbo_fp8.json")
    flux_workflow = paths.workflow("flux_dev.json")
    console.print(
        f"draining on {session.tier_label} ({session.vram_gb}GB, {session.quantisation}, "
        f"lora={'merged' if session.low_vram else 'bypass'}) until {session.window_end}"
    )

    from ai_studio.core.enums import MediaKind

    h3_backend = get_provider(
        "comfyui",
        workflow=workflow,
        base_url=session.comfy_url,
        hourly_usd=session.cost_per_hr,
    )
    flux_backend = get_provider(
        "flux",
        workflow=flux_workflow,
        base_url=session.comfy_url,
        hourly_usd=session.cost_per_hr,
        i2i_face_workflow=fun_paths.workflow("flux_dev_i2i_face.json"),
    )
    # Understanding and chat jobs share this pod's one FIFO queue -- a manual
    # drain that omitted them would KeyError the moment one was claimed.
    understand_backends = {
        kind: get_provider(
            name, base_url=session.inference_url, hourly_usd=session.cost_per_hr,
            max_output_chars=UNDERSTANDING_MAX_OUTPUT_CHARS,
        )
        for kind, name in (
            (MediaKind.IMAGE_UNDERSTAND, "understand-image"),
            (MediaKind.AUDIO_UNDERSTAND, "understand-audio"),
            (MediaKind.VIDEO_UNDERSTAND, "understand-video"),
            (MediaKind.CHAT, "chat"),
        )
    }
    queue = JobQueue()
    try:
        report = asyncio.run(
            drain_window(
                queue,
                {MediaKind.VIDEO: h3_backend, MediaKind.IMAGE: flux_backend, **understand_backends},
                window_end=datetime.fromisoformat(session.window_end),
                files_dir=fun.files_dir,
                gpu_tier=session.tier_label,
                gpu_usd_per_hr=session.cost_per_hr,
                poll_interval_s=poll_seconds,
                max_clips=max_clips,
                # Without this a long render looks like idleness and the reaper
                # closes the window out from under the clip being rendered.
                on_activity=sess.touch_activity,
                # The rewriter on the same pod; queued requests are converted
                # up front, one gpt-oss residency for the batch.
                llm=PodLlmClient(
                    InferenceClient(session.inference_url, timeout_s=settings.inference_timeout_s),
                    job_timeout_s=settings.inference_job_timeout_s,
                ),
            )
        )
    finally:
        queue.close()
    console.print(str(report))


class _RuntimeHost:
    """The composition root's half of `pipeline.worker.WindowHost`.

    `pipeline` may not import ai-studio's pod runtime (import-linter, "Only
    the composition root reaches the pod runtime"), so the worker loop takes
    sessions, providers and delivery by protocol. This is where the two
    packages are joined, which is exactly what a CLI module is for.
    """

    def __init__(self, *, name: str, poll_seconds: float, push: object | None = None) -> None:
        from fun_workflow.bots.line.push import LinePushClient, NullPushClient

        self.name = name
        self.poll_seconds = poll_seconds
        self._providers: dict[str, dict[MediaKind, object]] = {}
        self._llms: dict[str, object] = {}

        settings = get_fun_settings()
        self.files_dir = Path(settings.files_dir)
        self.base_url = settings.public_base_url.rstrip("/")
        token = settings.line_channel_access_token
        # NullPushClient without a token: the worker still renders and still
        # marks jobs delivered, and the log says the push was not sent. Failing
        # to start would make credentials a prerequisite for generating at all.
        self.push: Any = push or (
            LinePushClient(token.get_secret_value()) if token else NullPushClient()
        )

    def now(self) -> datetime:
        from datetime import timezone

        return datetime.now(timezone.utc)

    @staticmethod
    def _live_session(now: datetime | None) -> object | None:
        """The pod already open, if any."""
        from ai_studio.runtime import session as sess

        live = sess.load_state()
        if live is None or live.past_window(now):
            return None
        return live

    def claim_deadline(self, now: datetime | None = None) -> datetime:
        """The bell new work must finish before: the live pod's lease end, or
        the lease a pod opened now would get. This is what stops a render
        being started two minutes before `--terminate-after` throws it away.
        """
        from ai_studio.runtime import hours
        from ai_studio.runtime.session import Session

        live = cast(Session | None, self._live_session(now))
        if live is not None:
            return datetime.fromisoformat(live.window_end)
        return hours.window_end_for(now)

    def ensure_pod(self) -> object:
        from ai_studio.runtime import session as sess

        session = sess.ensure_pod(name=self.name)
        console.print(
            f"[green]window[/green] pod={session.pod_id} {session.tier_label} "
            f"${session.cost_per_hr:.2f}/hr until {session.window_end}"
        )
        return session

    def wait_ready(self, session: object) -> float:
        from ai_studio.runtime import session as sess

        live = cast(sess.Session, session)
        if not live.provisioned:
            # A fresh pod: start deploy/pod_setup.sh on it (the step that used
            # to be a person with a terminal), then wait for the node pack.
            console.print(f"  provisioning {live.pod_id} (pod_setup.sh + pod_setup.d over ssh)")
            sess.provision(live, extras=fun_paths.pod_setup_extras())
            sess.mark_provisioned()
        waited = sess.wait_ready(live)
        console.print(f"  ComfyUI ready after {waited:.0f}s")
        # A second, unrelated process on the same pod: deploy/pod_setup.sh
        # starts both, but they have separate readiness signals (ComfyUI's
        # node-pack probe vs a plain health check), so both are waited on
        # before any job -- including an understanding one -- is claimed.
        understanding_waited = sess.wait_understanding_ready(live)
        console.print(f"  inference server ready after {understanding_waited:.0f}s")
        return waited

    def providers_for(self, session: object) -> dict[MediaKind, object]:
        """Backends bound to this pod, built once per pod and then reused.

        Rebuilding them per job would reopen an HTTP client for every render;
        keying the cache on the pod id is what makes a *new* pod get new ones.
        """
        from ai_studio.runtime.session import Session

        live = cast(Session, session)
        if live.pod_id in self._providers:
            return self._providers[live.pod_id]

        workflow = paths.workflow("h3_fl2va_turbo.json" if live.low_vram else "h3_fl2va_turbo_fp8.json")
        flux_workflow = paths.workflow("flux_dev.json")

        built: dict[MediaKind, object] = {
            MediaKind.VIDEO: get_provider(
                "comfyui", workflow=workflow, base_url=live.comfy_url,
                hourly_usd=live.cost_per_hr,
            ),
            MediaKind.IMAGE: get_provider(
                "flux", workflow=flux_workflow, base_url=live.comfy_url,
                hourly_usd=live.cost_per_hr,
                i2i_face_workflow=fun_paths.workflow("flux_dev_i2i_face.json"),
            ),
            **{
                kind: get_provider(
                    name, base_url=live.inference_url, hourly_usd=live.cost_per_hr,
                    max_output_chars=UNDERSTANDING_MAX_OUTPUT_CHARS,
                )
                for kind, name in (
                    (MediaKind.IMAGE_UNDERSTAND, "understand-image"),
                    (MediaKind.AUDIO_UNDERSTAND, "understand-audio"),
                    (MediaKind.VIDEO_UNDERSTAND, "understand-video"),
                    (MediaKind.CHAT, "chat"),
                )
            },
        }
        self._providers = {live.pod_id: built}
        return built

    def llm_for(self, session: object) -> object:
        """The prompt rewriter on this pod: gpt-oss-20b through the inference
        server, one client per pod like `providers_for`. Never a serverless
        endpoint (decision 2026-08-27) -- see `pipeline.pod_llm`."""
        from ai_studio.inference.client import InferenceClient
        from ai_studio.pipeline.pod_llm import PodLlmClient
        from ai_studio.runtime.session import Session

        live = cast(Session, session)
        if live.pod_id not in self._llms:
            settings = get_settings()
            self._llms = {
                live.pod_id: PodLlmClient(
                    InferenceClient(live.inference_url, timeout_s=settings.inference_timeout_s),
                    job_timeout_s=settings.inference_job_timeout_s,
                )
            }
        return self._llms[live.pod_id]

    def touch_activity(self, kind: JobKind) -> None:
        from ai_studio.runtime import session as sess

        from fun_workflow.pipeline.idle import grace_for

        sess.touch_activity(kind.value, grace_minutes=grace_for(kind))

    async def deliver(self, job: Any, asset: Path | None) -> str:
        """Push the finished media into the group that asked for it, @-ing them.

        The poster is built here rather than in `push.py` because it is an
        ffmpeg call and `bots` has no business shelling out. If it fails, the
        delivery degrades to text and a link rather than being abandoned — a
        thumbnail is not worth losing a clip that cost GPU-minutes over.
        """
        from fun_workflow.bots.line import push as line_push

        status_url = f"{self.base_url}/q/{job.token}"

        if job.result_text:
            # An understanding job (/說圖 /說音 /說影): text, no media object
            # and no poster -- `asset` is always None for these.
            text = job.result_text
            if job.media_kind is JobKind.IMAGE_UNDERSTAND:
                # 📏 moondream3 never writes Chinese. Translating it here
                # would mean another model swap on the one card (~60 s per
                # photo), so the note is the honest, free answer.
                text = f"(moondream3 只能用英文描述)\n{text}"
            messages = line_push.understood_messages(
                result_text=text, status_url=status_url, quote_token=job.quote_token,
            )
            fallback = f"{text[:200]}\n{status_url}"
        elif asset is None:
            messages = line_push.failed_messages(
                reason=job.error or "unknown", status_url=status_url,
                prompt=job.text, quote_token=job.quote_token,
            )
            fallback = f"{job.text[:40]} 失敗了\n{status_url}"
        else:
            messages = []
            try:
                preview = media.poster(
                    asset, self.files_dir / f"{asset.stem}_poster.jpg",
                    max_bytes=PREVIEW_IMAGE_MAX_BYTES,
                )
            except AIStudioError as exc:
                console.print(f"[yellow]no poster for {asset.name}:[/yellow] {exc}")
            else:
                messages = line_push.delivered_messages(
                    media_url=f"{self.base_url}/files/{asset.name}",
                    preview_url=f"{self.base_url}/files/{preview.name}",
                    status_url=status_url,
                    is_video=job.media_kind in (JobKind.VIDEO, JobKind.DRAMA),
                    prompt=job.text,
                    quote_token=job.quote_token,
                    caption=_drama_caption(job),
                )
            fallback = f"{_drama_caption(job) or job.text[:40] + ' 完成了'}\n{status_url}"
            if not messages:
                messages = [
                    line_push.text_message(fallback, quote_token=job.quote_token)
                ]

        return await line_push.deliver(
            self.push,
            to=job.group_id,
            messages=messages,
            fallback_text=fallback,
            # LINE treats a repeat of this key as the same send, so a retry
            # after a timeout cannot bill the group twice.
            retry_key=job.token,
            quote_token=job.quote_token,
        )


def _drama_caption(job: Any) -> str | None:
    """「《title》logline」for a finished drama; None for every other kind,
    so `delivered_messages` keeps its「<prompt> 完成了」default."""
    if job.media_kind is not JobKind.DRAMA:
        return None
    screenplay = (job.prompt or {}).get("screenplay") or {}
    title = str(screenplay.get("title") or job.text[:20])
    logline = str(screenplay.get("logline") or "")
    return f"🎭《{title}》完成了\n{logline[:80]}".rstrip()



@app.command("reap")
def reap() -> None:
    """Close the pod once it has gone quiet. Schedule this every minute.

    `ai-studio session reap` with the one thing only this side knows: whether
    work is waiting in the queue. A pod with a job about to land on it is
    never closed, whatever the clock says. The grace was recorded by the
    worker with its last render (`pipeline.idle`), so nothing is passed here.
    """
    _setup_logging("reap")
    from ai_studio.runtime import session as sess

    from fun_workflow.pipeline.queue import JobQueue

    with JobQueue() as queue:
        hold = bool(queue.pending())
    decision = sess.close_if_idle(hold=hold)
    console.print(str(decision))
    sess.log_reap(decision)


@app.command()
def gc(
    days: float = typer.Option(
        None, help="Delete media older than this. Default: AI_STUDIO_FILES_RETENTION_DAYS."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would go, delete nothing."),
) -> None:
    """Prune old delivered media and received photos. Schedule this daily.

    `files/` gains an mp4/png plus a jpg poster per finished request and
    `incoming/` a jpg per photo sent, and nothing removed either -- on an
    always-on host that is a slow disk leak. A photo a still-live
    image-to-video job points at is protected regardless of age.
    """
    _setup_logging("gc")
    from ai_studio.storage.retention import sweep_old_files

    from fun_workflow.pipeline.queue import DEFAULT_DB, JobQueue
    from fun_workflow.storage.gc import prune_chat_turns, remove_empty_drama_dirs

    settings = get_settings()
    fun = get_fun_settings()
    max_age = fun.files_retention_days if days is None else days
    if max_age <= 0:
        console.print("retention is 0 (disabled); nothing pruned. [dim]The disk will fill.[/dim]")
        return

    protected: set[str] = set()
    try:
        with JobQueue() as queue:
            protected = {
                str(Path(j.first_frame_path).resolve())
                for j in queue.pending()
                if j.first_frame_path
            }
    except Exception as exc:  # a missing queue must not stop a disk sweep
        console.print(f"[yellow]could not read the queue ({exc}); protecting nothing[/yellow]")

    total_removed = total_freed = 0
    # A drama's run directory holds ~15 intermediate stills and clips; sweep
    # it on the same clock, but never one whose job is still pending -- the
    # state file is what lets a requeued drama resume instead of re-paying.
    drama_dirs = sorted(p for p in (settings.runs_dir / "drama").glob("*") if p.is_dir())
    pending_tokens = set()
    try:
        with JobQueue() as queue:
            pending_tokens = {j.token for j in queue.pending()}
    except Exception:  # the queue was already reported unreadable above
        pending_tokens = set()
    # files/index.jsonl maps every delivered file back to its request; it
    # is a file under files/ and would be swept like any other at 7 days.
    from fun_workflow.storage.index import index_path

    protected = protected | {str(index_path(fun.files_dir).resolve())}
    sweep_targets = [("files", fun.files_dir), ("incoming", fun.incoming_dir)]
    # `sweep_old_files` is deliberately flat, so each stage directory of a
    # drama is its own target (state.json and the manifest sit at the top).
    for d in drama_dirs:
        if d.name in pending_tokens:
            continue
        sweep_targets.append((f"runs/drama/{d.name}", d))
        sweep_targets += [(f"runs/drama/{d.name}/{sub.name}", sub) for sub in sorted(d.iterdir()) if sub.is_dir()]
    for label, directory in sweep_targets:
        result = sweep_old_files(
            directory, max_age_days=max_age, dry_run=dry_run, keep=protected
        )
        verb = "would remove" if dry_run else "removed"
        console.print(f"  {label}: {verb} {result.removed}, kept {result.kept} "
                      f"({result.freed_bytes / 1_048_576:.1f} MB)")
        total_removed += result.removed
        total_freed += result.freed_bytes
    removed_dirs = remove_empty_drama_dirs(settings.runs_dir, dry_run=dry_run)
    turns = prune_chat_turns(DEFAULT_DB, dry_run=dry_run)
    console.print(
        f"[green]gc[/green] {'(dry run) ' if dry_run else ''}"
        f"{total_removed} file(s), {total_freed / 1_048_576:.1f} MB, "
        f"older than {max_age:.0f}d; empty drama dirs {removed_dirs}, old chat_turns {turns}"
    )



@app.command("archive")
def archive(
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan and report; write and delete nothing."),
) -> None:
    """`ai-studio archive`, plus what this side keeps. Schedule this daily (03:00 Asia/Taipei).

    Adds to the tar a consistent sqlite backup of the queue, every drama's
    state and render manifest, and files/index.jsonl; then prunes old
    chat_turns rows and empty drama dirs on top of ai-studio's own pruning.
    """
    _setup_logging("archive")
    from ai_studio.storage.archive import run_archive

    from fun_workflow.pipeline.queue import DEFAULT_DB
    from fun_workflow.storage.gc import archive_members, prune_chat_turns, remove_empty_drama_dirs

    settings = get_settings()
    fun = get_fun_settings()
    db = DEFAULT_DB
    result = run_archive(
        root=Path.cwd(),
        log_dir=settings.log_dir,
        runs_dir=settings.runs_dir,
        out_dir=Path("out"),
        archive_dir=settings.archive_dir,
        hot_days=settings.log_hot_days,
        keep_days=settings.archive_keep_days,
        sqlite=db if db.is_file() else None,
        extra_members=archive_members(settings.runs_dir, fun.files_dir),
        dry_run=dry_run,
    )
    removed_dirs = remove_empty_drama_dirs(settings.runs_dir, dry_run=dry_run)
    turns = prune_chat_turns(db, dry_run=dry_run) if db.is_file() else 0
    if dry_run:
        console.print(result.plan.summary())
        for member in result.plan.members[:20]:
            console.print(f"  + {member}")
        if len(result.plan.members) > 20:
            console.print(f"  ... {len(result.plan.members) - 20} more")
    console.print(f"{result.summary()}; drama-dirs={removed_dirs} chat_turns={turns}")


_QUESTION_KINDS = {
    "image": JobKind.IMAGE_UNDERSTAND,
    "audio": JobKind.AUDIO_UNDERSTAND,
    "video": JobKind.VIDEO_UNDERSTAND,
}


@app.command("rewrite-question")
def rewrite_question(
    text: str = typer.Argument("", help="The member's words after the trigger; empty for the bare-trigger default."),
    kind: str = typer.Option(..., "--kind", "-k", help=f"One of: {', '.join(_QUESTION_KINDS)}."),
) -> None:
    """Print the question(s) an understanding job would send, rewritten on an open pod.

    The live smoke test of prompts/understanding.py against gpt-oss-20b --
    no LINE, no render. With empty text it needs no pod at all: the defaults
    are printed as-is.
    """
    try:
        asyncio.run(_rewrite_question(text, kind))
    except AIStudioError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None


async def _rewrite_question(text: str, kind: str) -> None:
    from ai_studio.inference.client import InferenceClient
    from ai_studio.pipeline.pod_llm import PodLlmClient

    from fun_workflow.prompts.understanding import convert_question

    modality = _QUESTION_KINDS.get(kind)
    if modality is None:
        raise AIStudioError(f"--kind must be one of {', '.join(_QUESTION_KINDS)}, got {kind!r}")
    llm: Any = None
    if text.strip():
        settings = get_settings()
        llm = PodLlmClient(
            InferenceClient(settings.inference_url, timeout_s=settings.inference_timeout_s),
            job_timeout_s=settings.inference_job_timeout_s,
        )
    started = time.monotonic()
    try:
        prompt, audio_prompt, how = await convert_question(text, llm, modality=modality)
    finally:
        if llm is not None:
            await llm.aclose()
    console.print(f"[bold]built_by[/bold] {how}   {time.monotonic() - started:.1f}s")
    console.print(prompt or "(no question: caption path)", highlight=False, markup=False)
    if audio_prompt:
        console.print("[bold]audio_prompt[/bold]")
        console.print(audio_prompt, highlight=False, markup=False)


@app.command("preflight")
def preflight_cmd(
    skip_suite: bool = typer.Option(False, "--skip-suite", help="Skip check 1 (pytest/ruff/lint-imports/mypy)."),
    push: bool = typer.Option(False, "--push", help="Check 5 only: actually send a message to the real group."),
) -> None:
    """Run the request-side pre-launch checks. Nothing here opens a pod.

    Signature verification and dedupe against the deployed secret, the
    queue -> rewrite path with a scripted LLM, the accept-and-hold path, the
    push client, and HTTP range requests on /files. The GPU side has its own
    list: `ai-studio preflight`.

    Exits 0 only when every check PASSes. A check that cannot run is SKIP,
    not PASS -- "could not verify" must never read as "verified".

    `--push` sends a real message to the real group and spends real quota. It
    is opt-in rather than merely credential-gated for exactly that reason.
    """
    from datetime import timezone

    from ai_studio.checks import Status, stamp, summarise

    from fun_workflow.cli.preflight import run_all

    results = run_all(run_suite=not skip_suite, send_push=push)

    table = Table(title="preflight (request side)")
    table.add_column("#", justify="right")
    table.add_column("check")
    table.add_column("", justify="center")
    table.add_column("detail", overflow="fold")
    colour = {Status.PASS: "green", Status.FAIL: "red", Status.SKIP: "yellow"}
    for result in results:
        table.add_row(
            str(result.number), result.name,
            f"[{colour[result.status]}]{result.status.value}[/{colour[result.status]}]", result.detail,
        )
    console.print(table)

    green, summary = summarise(results)
    console.print(f"\n{summary}")
    console.print(f"[dim]{stamp(results, when=datetime.now(timezone.utc)).splitlines()[0]}[/dim]")
    if green:
        console.print("[green]all green.[/green]")
        return
    console.print("[yellow]not green.[/yellow] A skip here is an unknown the first real request would discover.")
    raise typer.Exit(1)


DRYRUN_SCREENPLAY: dict[str, Any] = {
    # The canned screenwriter replies `drama-dryrun` feeds through the real
    # `prompts.drama` parser, so the offline run exercises validation too:
    # the beat template, the framing alternation, the one push-in.
    "outline": {
        "title": "夜市的信",
        "logline": "A night-market stall owner finds a letter that says the market closes tomorrow.",
        "style": "Live-action, cinematic",
        "anchor": {
            "name": "阿玲",
            "appearance": "25-year-old Asian woman, oval face, small mole under right eye, "
            "dark chin-length straight hair",
            "wardrobe": "a faded red apron over a white t-shirt",
            "voice": "soft, low, slightly hoarse",
        },
        "world": {
            "location": "a narrow night-market food stall facing a single row of neighbouring stalls",
            "light": "warm tungsten string lights from above left, cool blue dawn behind",
            "signature_prop": "a folded paper letter",
        },
        "beats": {
            "hook": "She lifts the stall's shutter before dawn and an envelope is taped underneath it.",
            "setup": "She reads it between customers: the market closes tomorrow.",
            "conflict": "A regular asks what is wrong; she says nothing is.",
            "turn": "Evening: down the row, every other stall is already packing up for good.",
            "payoff": "She writes a reply on the back of the letter.",
            "cliffhanger": "Dawn again: she tapes her reply where the first one was, and opens as usual.",
        },
        "overall_soundscape": "Sizzling oil, a crowd murmuring, scooters passing on the road behind.",
        "non_diegetic_music": "N/A",
    },
    "shots": [
        {"index": 1, "scene": "the stall before dawn, shutter half up, string lights off", "sub_shots": [
            {"framing": "close-up", "action": "the lead's hand peels an envelope from under the counter", "camera": {"motion": "static_shot"}},
            {"framing": "medium", "action": "the lead stands and turns the envelope over", "camera": {"motion": "tilt_up", "speed": "slow"}},
        ]},
        {"index": 2, "scene": "the same stall mid-evening, steam from the wok, the letter in hand", "sub_shots": [
            {"framing": "wide", "action": "the lead serves a customer with the letter tucked under the till", "camera": {"motion": "static_shot"}},
            {"framing": "close-up", "action": "the lead unfolds the letter and her hands go still", "camera": {"motion": "static_shot"}},
        ]},
        {"index": 3, "scene": "the stall counter, a regular customer's shoulder in the foreground", "sub_shots": [
            {"framing": "over-the-shoulder", "action": "the lead answers with a small shake of the head", "camera": {"motion": "static_shot"},
             "line": "沒事,明天照常開。"},
        ]},
        {"index": 4, "scene": "the market row at night, neighbouring stalls stacking crates", "cut_reason": "time_passing", "sub_shots": [
            {"framing": "wide", "action": "the lead stands at her counter looking down the row", "camera": {"motion": "pan_right", "speed": "slow"}},
            {"framing": "medium close-up", "action": "the lead's face as she sees the empty stalls", "camera": {"motion": "push_in", "amplitude": "small", "speed": "slow"}},
        ]},
        {"index": 5, "scene": "the counter under one work lamp, the letter face down, a pen", "sub_shots": [
            {"framing": "close-up", "action": "the lead writes on the back of the letter", "camera": {"motion": "static_shot"},
             "line": "我不走。"},
            {"framing": "medium", "action": "the lead folds the letter and holds it", "camera": {"motion": "static_shot"}},
        ]},
        {"index": 6, "scene": "the stall before dawn again, shutter going up, first light", "cut_reason": "time_passing", "sub_shots": [
            {"framing": "wide", "action": "the lead tapes the letter under the counter and lifts the shutter", "camera": {"motion": "tilt_up", "speed": "slow"}},
        ]},
    ],
}


@app.command("drama-dryrun")
def drama_dryrun(
    premise: str = typer.Argument("一個夜市老闆娘發現攤位下藏著一封信", help="The premise (recorded, not used offline)."),
    out: Path = typer.Option(Path("out"), help="Where the finished mp4 goes."),
    runs: Path = typer.Option(Path("runs/_dryrun"), help="Where the drama's state and stages go."),
    screenplay: Path | None = typer.Option(
        None, help="A JSON file with {outline, shots} to use instead of the built-in one."
    ),
) -> None:
    """Run the whole /短劇 stage machine offline: scripted screenwriter, stub
    Flux and H3 (ffmpeg testsrc2), real loudnorm + assembly. Proves the state
    file, the resume rule, the timeline and the splice with no pod and no
    money. Run it twice: the second run must render nothing paid for (the
    plan and offsets files are rewritten; they cost nothing)."""
    try:
        asyncio.run(_drama_dryrun(premise, out, runs, screenplay))
    except AIStudioError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None


async def _drama_dryrun(premise: str, out: Path, runs: Path, screenplay_file: Path | None) -> None:
    import json as _json
    from datetime import timedelta, timezone

    from ai_studio.llm.scripted import ScriptedLlmClient

    from fun_workflow.pipeline.drama import load_state, render_drama
    from fun_workflow.pipeline.queue import JobQueue
    from fun_workflow.prompts.drama import screenplay_payload, write_screenplay

    canned = DRYRUN_SCREENPLAY
    if screenplay_file is not None:
        canned = _json.loads(screenplay_file.read_text(encoding="utf-8"))
    shots = canned["shots"]
    client = ScriptedLlmClient(
        _json.dumps(canned["outline"], ensure_ascii=False),
        _json.dumps({"shots": shots[:3]}, ensure_ascii=False),
        _json.dumps({"shots": shots[3:]}, ensure_ascii=False),
    )
    screenplay, how = await write_screenplay(premise, client)
    console.print(f"[bold]{screenplay.title}[/bold] -- {screenplay.logline}  ({how})")
    console.print(f"  anchor: {screenplay.anchor.appearance}")

    runs.mkdir(parents=True, exist_ok=True)
    queue = JobQueue(runs / "dryrun.sqlite3")
    try:
        accepted, _ = queue.enqueue("dryrun", "Cdryrun", premise, media_kind=JobKind.DRAMA)
        queue.set_parsed(accepted.id, screenplay_payload(screenplay, how))
        job = queue.by_id(accepted.id)
        assert job is not None
        providers = {
            MediaKind.IMAGE: get_provider("stub-flux", work_dir=runs / "_stub"),
            MediaKind.VIDEO: get_provider("stub", work_dir=runs / "_stub"),
        }
        touches = 0

        def touched() -> None:
            nonlocal touches
            touches += 1

        started = time.monotonic()
        result = await render_drama(
            job, providers, files_dir=out, runs_dir=runs,
            deadline=datetime.now(timezone.utc) + timedelta(hours=2),
            poll_interval_s=0.0, on_activity=touched,
        )
        state = load_state(runs / "drama" / job.token)
    finally:
        queue.close()

    info = media.probe(result)
    console.print(
        f"\n[green]wrote[/green] {result}  {info.width}x{info.height} {info.duration_s:.1f}s "
        f"audio={'yes' if info.has_audio else 'no'} in {time.monotonic() - started:.0f}s"
    )
    console.print(
        f"  stills {len(state.character)}+{len(state.keyframes)}, clips {len(state.clips)}, "
        f"leveled {len(state.leveled)}, ffmpeg calls {len(state.ffmpeg_argv)}, "
        f"activity touches {touches}, face_repair={state.face_repair}"
    )
    manifest = _json.loads((runs / "drama" / job.token / "render_manifest.json").read_text(encoding="utf-8"))
    tl = manifest.get("timeline", {})
    console.print(
        f"  timeline {tl.get('total_s', 0):.1f}s, segments {tl.get('segments')}, "
        f"dissolves {tl.get('dissolves')}, plan_gate={state.plan_gate}"
    )
    console.print(f"  state: {runs / 'drama' / job.token / 'state.json'}  (run again: nothing re-renders)")
