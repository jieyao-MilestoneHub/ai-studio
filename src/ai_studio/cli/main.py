"""ai-studio command line."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from ai_studio import media
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import GenMode, MediaKind
from ai_studio.core.errors import AIStudioError
from ai_studio.core.ids import new_run_id, scene_id, shot_id
from ai_studio.core.provider_spec import ClipRequest
from ai_studio.editing.format_policy import plan_format, to_ffmpeg_filter
from ai_studio.providers.base import ClipProvider
from ai_studio.providers.registry import available, get_provider

app = typer.Typer(
    add_completion=False,
    help=(
        "Generate video (MiniMax H3) and images (Flux.1-dev) on RunPod, and run "
        "the LINE bot that triggers them."
    ),
)
console = Console()

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 14)


@app.command()
def doctor() -> None:
    """Check the local environment before anything expensive happens."""
    settings = get_settings()
    table = Table(title="ai-studio doctor", show_lines=False)
    table.add_column("check")
    table.add_column("result")
    table.add_column("detail", overflow="fold")

    ok = True

    # --- python: runpod-flash and several deps refuse 3.14+
    version = sys.version_info
    py_ok = MIN_PYTHON <= (version.major, version.minor) < MAX_PYTHON
    ok &= py_ok
    table.add_row(
        "python",
        _mark(py_ok),
        f"{version.major}.{version.minor}.{version.micro} (need >=3.10,<3.14)",
    )

    # --- ffmpeg
    ffmpeg_path = media.which(settings.ffmpeg_bin)
    ffprobe_path = media.which(settings.ffprobe_bin)
    ok &= bool(ffmpeg_path and ffprobe_path)
    table.add_row("ffmpeg", _mark(bool(ffmpeg_path)), ffmpeg_path or "not on PATH")
    table.add_row("ffprobe", _mark(bool(ffprobe_path)), ffprobe_path or "not on PATH")

    if ffmpeg_path:
        missing = media.missing_filters(settings.ffmpeg_bin)
        ok &= not missing
        table.add_row(
            "ffmpeg filters",
            _mark(not missing),
            "all present" if not missing else f"missing: {', '.join(missing)}",
        )

    # --- credentials
    has_key = settings.runpod_api_key is not None
    table.add_row(
        "RUNPOD_API_KEY",
        _mark(has_key, warn_only=True),
        "set" if has_key else "unset - only the stub provider will work",
    )
    table.add_row("comfy url", "-", settings.comfy_url)
    table.add_row("providers", "-", ", ".join(available()))
    table.add_row("cost ceiling", "-", f"${settings.max_cost_usd:.2f} per run")
    table.add_row("month ceiling", "-", f"${settings.max_month_usd:.2f} (VPS ${settings.vps_monthly_usd:.2f})")

    console.print(table)
    if not ok:
        console.print(
            "\n[yellow]Tip:[/yellow] ffmpeg not on PATH? On this machine it is at "
            r"[cyan]C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin[/cyan] - add that "
            "directory to PATH, or set AI_STUDIO_FFMPEG_BIN to the full exe path."
        )
        raise typer.Exit(1)
    console.print("\n[green]Environment looks good.[/green]")


@app.command("format")
def format_cmd(
    target: str = typer.Argument("yt_longform_1080p", help="Delivery target name."),
    width: int = typer.Option(864, help="Native output width."),
    height: int = typer.Option(480, help="Native output height."),
) -> None:
    """Show how a native canvas maps onto a delivery target."""
    try:
        plan = plan_format(width, height, target)
    except AIStudioError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    table = Table(title=f"{width}x{height} -> {target}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("strategy", plan.strategy.value)
    table.add_row("scale", f"{plan.scale_width}x{plan.scale_height}")
    table.add_row("crop", f"{plan.crop_width}x{plan.crop_height} @ +{plan.crop_x}+{plan.crop_y}")
    table.add_row("upscale", f"{plan.upscale_factor:.3f}x")
    table.add_row("area retained", f"{plan.area_retained:.1%}")
    if plan.waived:
        table.add_row("waived", "yes")
    console.print(table)
    console.print(f"\n[dim]{to_ffmpeg_filter(plan)}[/dim]")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="What to generate."),
    provider: str = typer.Option("stub", "--provider", "-p", help=f"One of: {', '.join(available())}"),
    workflow: Path | None = typer.Option(None, help="ComfyUI workflow JSON (comfyui provider)."),
    seconds: float = typer.Option(5.0, help="Clip length."),
    width: int = typer.Option(864),
    height: int = typer.Option(480),
    seed: int | None = typer.Option(None),
    out: Path = typer.Option(Path("out"), help="Where to write the clip."),
) -> None:
    """Generate a single clip. The end-to-end smoke test for a provider."""
    try:
        asyncio.run(_generate(prompt, provider, workflow, seconds, width, height, seed, out))
    except AIStudioError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None


async def _generate(
    prompt: str,
    provider_name: str,
    workflow: Path | None,
    seconds: float,
    width: int,
    height: int,
    seed: int | None,
    out: Path,
) -> None:
    settings = get_settings()
    kwargs = {"workflow": workflow} if workflow else {}
    # This command only ever builds a ClipRequest (see below), so it needs a
    # clip provider specifically — the registry serves both kinds now that
    # `flux` is registered alongside `stub`/`comfyui`.
    backend = cast(ClipProvider, get_provider(provider_name, **kwargs))
    caps = backend.capabilities()

    estimate = caps.estimated_cost_usd(seconds)
    console.print(
        f"[bold]{caps.model_id}[/bold]  {caps.native_width}x{caps.native_height}"
        f"@{caps.native_fps}  audio={'yes' if caps.has_native_audio else 'no'}"
    )
    console.print(f"estimated cost ${estimate:.4f}, ceiling ${settings.max_cost_usd:.2f}")
    if estimate > settings.max_cost_usd:
        raise AIStudioError(
            f"estimated ${estimate:.2f} exceeds the ${settings.max_cost_usd:.2f} ceiling; "
            "raise AI_STUDIO_MAX_COST_USD deliberately if that is what you want"
        )

    run = new_run_id()
    request = ClipRequest(
        shot_id=shot_id(scene_id(run, 0), 0),
        mode=GenMode.T2V,
        prompt=prompt,
        width=width,
        height=height,
        duration_s=seconds,
        fps=caps.native_fps,
        seed=seed,
    )

    try:
        job = await backend.submit(request)
        console.print(f"submitted [cyan]{job.job_id}[/cyan]")

        waited = 0.0
        while not job.is_terminal:
            await asyncio.sleep(settings.comfy_poll_interval_s)
            waited += settings.comfy_poll_interval_s
            if waited > settings.comfy_job_timeout_s:
                await backend.cancel(job)
                raise AIStudioError(f"job {job.job_id} exceeded {settings.comfy_job_timeout_s}s")
            job = await backend.poll(job)
            console.print(f"  {job.state.value} ({waited:.0f}s)", highlight=False)

        if not job.state.is_success:
            raise AIStudioError(f"generation failed: {job.error or job.state.value}")

        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"{run}.mp4"
        asset = await backend.fetch(job, dest)
    finally:
        await backend.aclose()

    console.print(
        f"\n[green]wrote[/green] {dest}\n"
        f"  {asset.width}x{asset.height} @ {asset.fps:.2f}fps, {asset.duration_s:.2f}s, "
        f"audio={'yes' if asset.has_audio else 'no'}, "
        f"{asset.size_bytes / 1_048_576:.1f} MB, cost ${asset.cost_usd:.4f}"
    )


line_app = typer.Typer(help="LINE bot: serve the webhook, or discover a group id.")
app.add_typer(line_app, name="line")


def _run_server(host: str, port: int, reload: bool = False) -> None:
    import logging

    import uvicorn

    from ai_studio.api.main import create_app

    # log_config=None keeps uvicorn from calling dictConfig, which clears every
    # existing handler and defines no root logger - so a basicConfig set up here
    # would be silently discarded and ai-studio's own lines would never appear.
    # Owning the config instead means our INFO lines and uvicorn's both show up,
    # here and under journalctl on the VPS.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    uvicorn.run(
        create_app(),
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
    settings = get_settings()
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
    settings = get_settings()
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
    # Suggest the ASCII alias: a CJK trigger word printed to a cp950 console
    # comes out as mojibake, which makes the instruction useless. /gen, /生成
    # and 生成 all trigger, so tell them the one that survives the terminal.
    console.print("  3. say [cyan]/gen test[/cyan] in the group "
                  "(the CJK trigger also works, but may not print here)")
    console.print("  the group id will be printed here, and replied into the chat")
    console.print("")
    _run_server(host, port)


session_app = typer.Typer(
    help="Service windows. `open` at the window start, `close` at the end — schedule both."
)
app.add_typer(session_app, name="session")

WINDOW_TZ = "Asia/Taipei"


def _window_end(clock: str, tz_name: str) -> datetime:
    """Parse ``HH:MM`` in `tz_name` into today's (or tomorrow's) instant."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_name)
    hour, _, minute = clock.partition(":")
    now = datetime.now(tz)
    end = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    if end <= now:
        end += timedelta(days=1)  # a window that has already passed means tomorrow's
    return end


@session_app.command("open")
def session_open(
    until: str = typer.Option(..., "--until", help="Window end as HH:MM, e.g. 14:48."),
    tz: str = typer.Option(WINDOW_TZ, "--tz", help="Timezone for --until."),
    name: str = typer.Option("ai-studio-window"),
) -> None:
    """Deploy the window's pod. Sets --terminate-after as a backstop.

    Two checks run before anything is created: a demand gate (skip entirely
    if the queue is empty — the timer fires unconditionally every day, this
    command decides whether that's worth spending on) and a monthly budget
    guard (refuse, or shrink the window, once this month's cap is close).
    """
    from datetime import timezone

    from ai_studio.pipeline.queue import JobQueue
    from ai_studio.runtime import session as sess
    from ai_studio.runtime.budget import MonthlyBudgetGuard, SpendLedger

    with JobQueue() as queue:
        if not queue.pending():
            console.print("no queued work; skipping window open (no spend today)")
            return

    settings = get_settings()
    guard = MonthlyBudgetGuard(
        SpendLedger(),
        cap_usd=settings.max_month_usd,
        vps_monthly_usd=settings.vps_monthly_usd,
    )
    try:
        guard.refuse_if_broke(sess.CANDIDATES)
    except AIStudioError as exc:
        console.print(f"[red]window did not open:[/red] {exc}")
        raise typer.Exit(1) from None

    end = _window_end(until, tz)
    worst_case_hourly = max(tier.usd_per_hr for tier in sess.CANDIDATES)
    end = guard.throttle(end, datetime.now(timezone.utc), worst_case_hourly)

    try:
        s = sess.open_session(end, name=name)
    except AIStudioError as exc:
        console.print(f"[red]window did not open:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(
        f"[green]window open[/green]  pod={s.pod_id}  {s.gpu.replace('NVIDIA ', '')}"
        f"  {s.datacenter}/{s.cloud}  ${s.cost_per_hr:.2f}/hr"
    )
    console.print(f"  closes {end.isoformat(timespec='minutes')} (self-terminates ~10min later)")
    console.print(f"  export [cyan]AI_STUDIO_COMFY_URL={s.comfy_url}[/cyan]")
    console.print("  ComfyUI answers 502 for ~4 min while it copies itself to /workspace.")


@session_app.command("close")
def session_close(name: str = typer.Option("ai-studio-window")) -> None:
    """Terminate the window's pod. Idempotent; safe to schedule unconditionally.

    `sess.close_session()` itself records the session's cost into the monthly
    ledger — not this command — so `session reap`'s early closes (the common
    case: see its own docstring) are recorded too, not just this scheduled one.
    """
    from ai_studio.runtime import session as sess

    terminated = sess.close_session(name=name)
    if not terminated:
        console.print("nothing to close. [dim]Nothing is billing.[/dim]")
        return
    for pod_id in terminated:
        console.print(f"[green]terminated[/green] {pod_id}")


@session_app.command("status")
def session_status() -> None:
    """Show the live window, what it has cost so far, and how long is left."""
    from ai_studio.runtime import session as sess

    s = sess.load_state()
    if s is None:
        console.print("no window open.")
        pods = sess.list_pods()
        if pods:
            console.print(f"[yellow]! {len(pods)} pod(s) running with no session state[/yellow]")
            for p in pods:
                console.print(f"    {p.get('id')} {p.get('name')} ${p.get('costPerHr')}/hr")
        return

    table = Table(title=f"window {s.pod_id}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("gpu", f"{s.gpu.replace('NVIDIA ', '')} ({s.datacenter}/{s.cloud})")
    table.add_row("rate", f"${s.cost_per_hr:.2f}/hr")
    table.add_row("elapsed", f"{s.elapsed_hours():.2f} h")
    table.add_row("spent", f"${s.spent_usd():.2f}")
    table.add_row("closes", s.window_end)
    table.add_row("past window", "yes" if s.past_window() else "no")
    table.add_row("comfy", s.comfy_url)
    console.print(table)


@session_app.command("reap")
def session_reap(
    idle_minutes: int = typer.Option(20, help="Close early after this much idle time."),
) -> None:
    """Close the window early if it has gone quiet. Schedule this every 5 minutes.

    A window sized for peak demand is mostly idle at low volume, and idle
    minutes cost the same as working ones.
    """
    from ai_studio.runtime import session as sess

    console.print(sess.close_if_idle(idle_minutes))


@session_app.command("drain")
def session_drain(
    max_clips: int | None = typer.Option(None, help="Stop after N clips."),
    poll_seconds: float = typer.Option(15.0, help="How often to poll ComfyUI."),
) -> None:
    """Render queued requests on the open window's pod.

    This is what turns an open window into videos. Schedule it alongside `reap`:
    it exits immediately and successfully when no window is open, so a timer
    firing outside the window is a no-op rather than a failure, and if a drain
    dies mid-window the next tick picks the queue back up.

    Which workflow runs is decided by the rung that answered, not by preference.
    A 48GB card takes the fp8 graph and applies the LoRA in bypass; a 24GB card
    takes int8 with the LoRA merged, which the node pack itself calls softer.
    """
    from ai_studio.pipeline.drain import drain_window
    from ai_studio.pipeline.queue import JobQueue
    from ai_studio.runtime import session as sess

    settings = get_settings()
    session = sess.load_state()
    if session is None:
        console.print("no window is open; nothing to drain")
        return
    if session.past_window():
        console.print("the window has already ended; not claiming new work")
        return

    workflow = Path("workflows") / (
        "h3_fl2va_turbo.json" if session.low_vram else "h3_fl2va_turbo_fp8.json"
    )
    flux_workflow = Path("workflows") / "flux_dev.json"
    if not workflow.is_file():
        console.print(f"[red]missing workflow {workflow}[/red] (run from the repo root)")
        raise typer.Exit(1)
    if not flux_workflow.is_file():
        console.print(f"[red]missing workflow {flux_workflow}[/red] (run from the repo root)")
        raise typer.Exit(1)

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
    )
    queue = JobQueue()
    try:
        report = asyncio.run(
            drain_window(
                queue,
                {MediaKind.VIDEO: h3_backend, MediaKind.IMAGE: flux_backend},
                window_end=datetime.fromisoformat(session.window_end),
                files_dir=settings.files_dir,
                gpu_tier=session.tier_label,
                poll_interval_s=poll_seconds,
                max_clips=max_clips,
                # Without this a long render looks like idleness and the reaper
                # closes the window out from under the clip being rendered.
                on_activity=sess.touch_activity,
            )
        )
    finally:
        queue.close()
    console.print(str(report))


class _RuntimeHost:
    """The composition root's half of `pipeline.worker.WindowHost`.

    `pipeline` sits below `runtime` in the layer contract, so the worker loop
    cannot import business hours, sessions or the provider registry. It takes
    them by protocol instead, and this is where the two halves are joined —
    which is exactly what the CLI is for, and why it is the one package
    exempted from the "phase-2 packages stay leaves" contract.
    """

    def __init__(self, *, name: str, poll_seconds: float) -> None:
        self.name = name
        self.poll_seconds = poll_seconds
        self._providers: dict[str, dict[MediaKind, object]] = {}

    def now(self) -> datetime:
        from datetime import timezone

        return datetime.now(timezone.utc)

    def is_open(self, now: datetime | None = None) -> bool:
        from ai_studio.runtime import hours

        return hours.is_open(now)

    def claim_deadline(self, now: datetime | None = None) -> datetime:
        """The end of business hours, not the end of a two-hour lease.

        Same reserve as `drain_window`'s, different bell: this is what stops a
        render being started at 12:58 that `--terminate-after` then throws away.
        """
        from ai_studio.runtime import hours

        return hours.window_end_for(now)

    def ensure_pod(self, queue: object) -> object:
        from ai_studio.pipeline.queue import JobQueue
        from ai_studio.runtime import session as sess

        session = sess.ensure_pod(cast(JobQueue, queue), name=self.name)
        console.print(
            f"[green]window[/green] pod={session.pod_id} {session.tier_label} "
            f"${session.cost_per_hr:.2f}/hr until {session.window_end}"
        )
        return session

    def wait_ready(self, session: object) -> float:
        from ai_studio.runtime import session as sess

        waited = sess.wait_ready(cast(sess.Session, session))
        console.print(f"  ComfyUI answered after {waited:.0f}s")
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

        workflow = Path("workflows") / (
            "h3_fl2va_turbo.json" if live.low_vram else "h3_fl2va_turbo_fp8.json"
        )
        flux_workflow = Path("workflows") / "flux_dev.json"
        for path in (workflow, flux_workflow):
            if not path.is_file():
                raise AIStudioError(f"missing workflow {path} (run from the repo root)")

        built: dict[MediaKind, object] = {
            MediaKind.VIDEO: get_provider(
                "comfyui", workflow=workflow, base_url=live.comfy_url,
                hourly_usd=live.cost_per_hr,
            ),
            MediaKind.IMAGE: get_provider(
                "flux", workflow=flux_workflow, base_url=live.comfy_url,
                hourly_usd=live.cost_per_hr,
            ),
        }
        self._providers = {live.pod_id: built}
        return built

    def touch_activity(self) -> None:
        from ai_studio.runtime import session as sess

        sess.touch_activity()


@app.command("worker")
def worker(
    name: str = typer.Option("ai-studio-window", help="Pod name to open and reuse."),
    poll_seconds: float = typer.Option(15.0, help="How often to poll ComfyUI."),
    idle_seconds: float = typer.Option(10.0, help="Queue check interval while open."),
    max_ticks: int | None = typer.Option(None, help="Stop after N passes. For testing."),
) -> None:
    """Serve the queue: open a pod when work arrives inside business hours.

    This is what replaces the `open` and `drain` timers. It runs as a systemd
    service with `Restart=always` and sleeps its way through everything outside
    11:00-13:00 Asia/Taipei, so there is nothing to schedule and nothing that
    fires when no one has asked for anything.

    Closing is still someone else's job, on purpose: `session reap` every five
    minutes, `session close` at the bell, and `--terminate-after` on the pod
    itself. Three independent ways for a machine to stop billing, none of which
    depends on this process still being alive.
    """
    from ai_studio.pipeline.queue import JobQueue
    from ai_studio.pipeline.worker import serve

    settings = get_settings()
    host = _RuntimeHost(name=name, poll_seconds=poll_seconds)
    queue = JobQueue()
    console.print(
        f"worker up. business hours 11:00-13:00 Asia/Taipei, "
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


pod_app = typer.Typer(help="RunPod pod lifecycle. `down` terminates — stopping still bills.")
app.add_typer(pod_app, name="pod")


@pod_app.command("capacity")
def pod_capacity() -> None:
    """Find a licence-safe datacenter that currently has stock."""
    from ai_studio.runtime.pod import PodManager

    try:
        with PodManager() as manager:
            gpu_id, dc_id = manager.find_capacity()
    except AIStudioError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]available:[/green] {gpu_id} in {dc_id}")


@pod_app.command("placement")
def pod_placement() -> None:
    """Check every ladder rung against the catalog, before a window needs it.

    A rung whose datacenter does not offer that card is refused on deploy just
    like one that is merely out of stock -- so without this the ladder can fall
    through to a softer GPU for months and look like bad luck.
    """
    from ai_studio.runtime.pod import LICENCE_SAFE_DATACENTERS, PodManager
    from ai_studio.runtime.session import CANDIDATES

    verdicts = {
        "stock": ("green", "has stock"),
        "empty": ("yellow", "no stock right now"),
        "not-offered": ("red", "NOT OFFERED HERE - this rung can never fill"),
        "unverifiable": ("cyan", "community cloud reports no per-dc breakdown"),
    }
    dead = 0
    with PodManager() as manager:
        for i, tier in enumerate(CANDIDATES, 1):
            if tier.datacenter not in LICENCE_SAFE_DATACENTERS:
                console.print(f"  {i}. [red]{tier.label}: outside H3's licence[/red]")
                dead += 1
                continue
            verdict = manager.verify_placement(
                tier.gpu, tier.datacenter, cloud=tier.cloud
            )
            colour, text = verdicts[verdict]
            console.print(
                f"  {i}. {tier.gpu} {tier.datacenter} {tier.cloud} "
                f"${tier.usd_per_hr:.3f}/hr  [{colour}]{text}[/{colour}]"
            )
            dead += verdict == "not-offered"

    if dead:
        console.print(
            f"\n[red]{dead} rung(s) can never be filled.[/red] The window will "
            "fall through to a lower rung every time, which is a quality "
            "downgrade, not just a price one."
        )
        raise typer.Exit(1)
    console.print("\n[green]every rung is licence-safe and actually offered[/green]")


@pod_app.command("up")
def pod_up(
    template_id: str = typer.Option(
        ..., help="Official RunPod ComfyUI template id — see runtime.session.TEMPLATE_COMFYUI_STANDARD."
    ),
    name: str = typer.Option("ai-studio-comfyui"),
) -> None:
    """Deploy the ComfyUI pod, verifying host RAM before accepting it."""
    from ai_studio.runtime.pod import PodManager

    try:
        with PodManager() as manager:
            status = manager.up(template_id=template_id, name=name)
    except AIStudioError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(f"[green]pod {status.id}[/green] {status.status} in {status.data_center_id}")
    console.print(
        "ComfyUI will 502 through the proxy for roughly the first four minutes "
        "while it copies itself into /workspace. Readiness log line: "
        "[dim][ComfyUI-Manager] All startup tasks have been completed.[/dim]"
    )
    console.print(f"\nSet [cyan]AI_STUDIO_COMFY_URL=https://{status.id}-8188.proxy.runpod.net[/cyan]")


@pod_app.command("status")
def pod_status(pod_id: str | None = typer.Argument(None)) -> None:
    """Show pods and any warnings worth acting on."""
    from ai_studio.runtime.pod import PodManager

    try:
        with PodManager() as manager:
            pods = [manager.status(pod_id)] if pod_id else manager.list_pods()
            warnings = {p.id: manager.health_warnings(p) for p in pods}
    except AIStudioError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    if not pods:
        console.print("no pods. [dim]Nothing is billing.[/dim]")
        return

    table = Table(title="pods")
    for column in ("id", "name", "status", "dc", "gpu", "ram", "uptime", "cost"):
        table.add_column(column)
    for pod in pods:
        spent = pod.cost_so_far_usd()
        table.add_row(
            pod.id,
            pod.name,
            pod.status,
            pod.data_center_id or "-",
            (pod.gpu_id or "-").replace("NVIDIA ", ""),
            f"{pod.host_ram_gb:.0f}G" if pod.host_ram_gb else "?",
            f"{pod.uptime_s / 60:.0f}m",
            f"${spent:.2f}" if spent is not None else "?",
        )
    console.print(table)

    for pod_key, messages in warnings.items():
        for message in messages:
            console.print(f"[yellow]! {pod_key}:[/yellow] {message}")


@pod_app.command("down")
def pod_down(pod_id: str = typer.Argument(..., help="Pod to terminate.")) -> None:
    """Terminate a pod. This is the only thing that stops billing."""
    from ai_studio.runtime.pod import PodManager

    try:
        with PodManager() as manager:
            manager.down(pod_id)
    except AIStudioError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]terminated[/green] {pod_id}")


def _mark(ok: bool, *, warn_only: bool = False) -> str:
    if ok:
        return "[green]ok[/green]"
    return "[yellow]warn[/yellow]" if warn_only else "[red]fail[/red]"


if __name__ == "__main__":
    app()
