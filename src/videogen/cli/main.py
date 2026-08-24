"""videogen command line."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from videogen import media
from videogen.config.settings import get_settings
from videogen.core.enums import GenMode
from videogen.core.errors import VideogenError
from videogen.core.ids import new_run_id, scene_id, shot_id
from videogen.core.provider_spec import ClipRequest
from videogen.editing.format_policy import plan_format, to_ffmpeg_filter
from videogen.providers.registry import available, get_provider

app = typer.Typer(
    add_completion=False,
    help="Generate video with MiniMax H3 on RunPod, assembled with an editing grammar.",
)
console = Console()

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 14)


@app.command()
def doctor() -> None:
    """Check the local environment before anything expensive happens."""
    settings = get_settings()
    table = Table(title="videogen doctor", show_lines=False)
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

    console.print(table)
    if not ok:
        console.print(
            "\n[yellow]Tip:[/yellow] ffmpeg not on PATH? On this machine it is at "
            r"[cyan]C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin[/cyan] - add that "
            "directory to PATH, or set VIDEOGEN_FFMPEG_BIN to the full exe path."
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
    except VideogenError as exc:
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
    except VideogenError as exc:
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
    backend = get_provider(provider_name, **kwargs)
    caps = backend.capabilities()

    estimate = caps.estimated_cost_usd(seconds)
    console.print(
        f"[bold]{caps.model_id}[/bold]  {caps.native_width}x{caps.native_height}"
        f"@{caps.native_fps}  audio={'yes' if caps.has_native_audio else 'no'}"
    )
    console.print(f"estimated cost ${estimate:.4f}, ceiling ${settings.max_cost_usd:.2f}")
    if estimate > settings.max_cost_usd:
        raise VideogenError(
            f"estimated ${estimate:.2f} exceeds the ${settings.max_cost_usd:.2f} ceiling; "
            "raise VIDEOGEN_MAX_COST_USD deliberately if that is what you want"
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
                raise VideogenError(f"job {job.job_id} exceeded {settings.comfy_job_timeout_s}s")
            job = await backend.poll(job)
            console.print(f"  {job.state.value} ({waited:.0f}s)", highlight=False)

        if not job.state.is_success:
            raise VideogenError(f"generation failed: {job.error or job.state.value}")

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

    from videogen.api.main import create_app

    # log_config=None keeps uvicorn from calling dictConfig, which clears every
    # existing handler and defines no root logger - so a basicConfig set up here
    # would be silently discarded and videogen's own lines would never appear.
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
    console.print(f"[bold]videogen[/bold] on {host}:{port}")
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
    name: str = typer.Option("videogen-window"),
) -> None:
    """Deploy the window's pod. Sets --terminate-after as a backstop."""
    from videogen.runtime import session as sess

    end = _window_end(until, tz)
    try:
        s = sess.open_session(end, name=name)
    except VideogenError as exc:
        console.print(f"[red]window did not open:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(
        f"[green]window open[/green]  pod={s.pod_id}  {s.gpu.replace('NVIDIA ', '')}"
        f"  {s.datacenter}/{s.cloud}  ${s.cost_per_hr:.2f}/hr"
    )
    console.print(f"  closes {end.isoformat(timespec='minutes')} (self-terminates ~10min later)")
    console.print(f"  export [cyan]VIDEOGEN_COMFY_URL={s.comfy_url}[/cyan]")
    console.print("  ComfyUI answers 502 for ~4 min while it copies itself to /workspace.")


@session_app.command("close")
def session_close(name: str = typer.Option("videogen-window")) -> None:
    """Terminate the window's pod. Idempotent; safe to schedule unconditionally."""
    from videogen.runtime import session as sess

    terminated = sess.close_session(name=name)
    if not terminated:
        console.print("nothing to close. [dim]Nothing is billing.[/dim]")
        return
    for pod_id in terminated:
        console.print(f"[green]terminated[/green] {pod_id}")


@session_app.command("status")
def session_status() -> None:
    """Show the live window, what it has cost so far, and how long is left."""
    from videogen.runtime import session as sess

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
    from videogen.runtime import session as sess

    console.print(sess.close_if_idle(idle_minutes))


pod_app = typer.Typer(help="RunPod pod lifecycle. `down` terminates — stopping still bills.")
app.add_typer(pod_app, name="pod")


@pod_app.command("capacity")
def pod_capacity() -> None:
    """Find a licence-safe datacenter that currently has stock."""
    from videogen.runtime.pod import PodManager

    try:
        with PodManager() as manager:
            gpu_id, dc_id = manager.find_capacity()
    except VideogenError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]available:[/green] {gpu_id} in {dc_id}")


@pod_app.command("up")
def pod_up(
    template_id: str = typer.Option(..., help="Official RunPod ComfyUI template id (CUDA 13)."),
    name: str = typer.Option("videogen-comfyui"),
) -> None:
    """Deploy the ComfyUI pod, verifying host RAM before accepting it."""
    from videogen.runtime.pod import PodManager

    try:
        with PodManager() as manager:
            status = manager.up(template_id=template_id, name=name)
    except VideogenError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(f"[green]pod {status.id}[/green] {status.status} in {status.data_center_id}")
    console.print(
        "ComfyUI will 502 through the proxy for roughly the first four minutes "
        "while it copies itself into /workspace. Readiness log line: "
        "[dim][ComfyUI-Manager] All startup tasks have been completed.[/dim]"
    )
    console.print(f"\nSet [cyan]VIDEOGEN_COMFY_URL=https://{status.id}-8188.proxy.runpod.net[/cyan]")


@pod_app.command("status")
def pod_status(pod_id: str | None = typer.Argument(None)) -> None:
    """Show pods and any warnings worth acting on."""
    from videogen.runtime.pod import PodManager

    try:
        with PodManager() as manager:
            pods = [manager.status(pod_id)] if pod_id else manager.list_pods()
            warnings = {p.id: manager.health_warnings(p) for p in pods}
    except VideogenError as exc:
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
    from videogen.runtime.pod import PodManager

    try:
        with PodManager() as manager:
            manager.down(pod_id)
    except VideogenError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]terminated[/green] {pod_id}")


def _mark(ok: bool, *, warn_only: bool = False) -> str:
    if ok:
        return "[green]ok[/green]"
    return "[yellow]warn[/yellow]" if warn_only else "[red]fail[/red]"


if __name__ == "__main__":
    app()
