"""ai-studio command line."""

from __future__ import annotations

import asyncio
import json
import sys
import time
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

    log_dir, archive_dir = settings.log_dir, settings.archive_dir
    log_bytes = sum(p.stat().st_size for p in log_dir.rglob("*") if p.is_file()) if log_dir.is_dir() else 0
    newest_log = max((p for p in log_dir.rglob("*.jsonl")), key=lambda p: p.stat().st_mtime, default=None) if log_dir.is_dir() else None
    table.add_row(
        "logs", _mark(log_dir.is_dir()),
        f"{log_dir} {log_bytes / 1_048_576:.1f} MB" + (f", newest {newest_log.name}" if newest_log else "")
        + f" (hot {settings.log_hot_days:.0f}d, level {settings.log_level})",
    )
    days = sorted(p.name for p in archive_dir.iterdir() if p.is_dir()) if archive_dir.is_dir() else []
    arch_bytes = sum(p.stat().st_size for p in archive_dir.rglob("*") if p.is_file()) if archive_dir.is_dir() else 0
    table.add_row(
        "archive", _mark(bool(days)),
        f"{archive_dir} {len(days)} day(s), {arch_bytes / 1_048_576:.1f} MB"
        + (f", last {days[-1]}" if days else ", none yet") + f" (keep {settings.archive_keep_days:.0f}d, `ai-studio archive`)",
    )

    console.print(table)
    if not ok:
        console.print(
            "\n[yellow]Tip:[/yellow] ffmpeg not on PATH? On this machine it is at "
            r"[cyan]C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin[/cyan] - add that "
            "directory to PATH, or set AI_STUDIO_FFMPEG_BIN to the full exe path."
        )
        raise typer.Exit(1)
    console.print("\n[green]Environment looks good.[/green]")


@app.command("archive")
def archive(
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan and report; write and delete nothing."),
) -> None:
    """Snapshot, compress, verify, then prune. Schedule this daily (03:00 Asia/Taipei).

    Tars the JSONL traces (days before today), session and pod records and
    the spend ledger into archive/<day>/*.tar.zst with a manifest of
    sha256s; then deletes hot logs older than AI_STUDIO_LOG_HOT_DAYS (only
    what a manifest names), archives older than AI_STUDIO_ARCHIVE_KEEP_DAYS
    and stale dry-run/stub/out files; then folds any real render since the
    last run into runs/benchmark/<month>.json, a durable per-GPU-tier
    performance aggregate (see docs/observability.md). Idempotent: a second
    run the same day only prunes and has nothing new to fold.

    The request-taking side has more to keep (its queue, drama state,
    delivery index): `funapp archive` runs this with those added, and is
    what the daily timer calls on a host that serves the group.
    """
    _setup_logging("archive")
    from ai_studio.storage.archive import run_archive

    settings = get_settings()
    result = run_archive(
        root=Path.cwd(),
        log_dir=settings.log_dir,
        runs_dir=settings.runs_dir,
        out_dir=Path("out"),
        archive_dir=settings.archive_dir,
        hot_days=settings.log_hot_days,
        keep_days=settings.archive_keep_days,
        dry_run=dry_run,
    )
    if dry_run:
        console.print(result.plan.summary())
        for member in result.plan.members[:20]:
            console.print(f"  + {member}")
        if len(result.plan.members) > 20:
            console.print(f"  ... {len(result.plan.members) - 20} more")
    console.print(f"[green]{result.summary()}[/green]")


@app.command("bench")
def bench(
    month: str | None = typer.Option(None, "--month", help="YYYY-MM; default this month."),
    as_json: bool = typer.Option(False, "--json", help="Print the raw report instead of a table."),
) -> None:
    """What each GPU tier has measured this month, and what the open pod rents for.

    Reads runs/benchmark/<month>.json (folded daily by `archive` from real
    renders only) and runs/.session.json. Every number here is 📏 measured
    on our own hardware; nothing is promoted to docs/ without a person
    reading this first (CLAUDE.md, "Number honesty").
    """
    from ai_studio.benchmark import live_rate, month_report
    from ai_studio.runtime.session import load_state

    settings = get_settings()
    rate = live_rate(load_state())
    report = month_report(Path(settings.runs_dir), month)
    if as_json:
        console.print_json(json.dumps({"live": rate.__dict__ if rate else None, "report": report}))
        return

    if rate:
        console.print(
            f"[green]open[/green] {rate.tier}  ${rate.usd_per_hr:.3f}/hr  {rate.vram_gb}GB  "
            f"{rate.datacenter}  since {rate.since}"
        )
    else:
        console.print("no pod open")
    if not report:
        console.print("no benchmark report yet (nothing real has rendered and been archived)")
        return

    table = Table(title=f"benchmark {report['month']}  days={len(report.get('days_included', []))}")
    for col in ("kind/gpu_tier", "n", "s mean", "$ mean", "VRAM GB", "frames/s"):
        table.add_column(col, justify="right" if col != "kind/gpu_tier" else "left")
    for key, g in sorted(report.get("groups", {}).items()):
        table.add_row(
            key, str(g.get("count", 0)),
            _fmt(g.get("seconds_mean"), 1), _fmt(g.get("cost_usd_mean"), 3),
            _fmt(g.get("vram_gb_mean"), 1), _fmt(g.get("frames_per_s_mean"), 2),
        )
    console.print(table)


def _fmt(value: object, places: int) -> str:
    return "-" if not isinstance(value, int | float) else f"{value:.{places}f}"


@app.command("preflight")
def preflight_cmd(
    skip_suite: bool = typer.Option(
        False, "--skip-suite", help="Skip check 1 (pytest/ruff/lint-imports/mypy)."
    ),
) -> None:
    """Run the GPU-side pre-launch checks. Nothing here opens a pod.

    Everything provable without spending a GPU-second, proved, so a live run
    is spent only on the part that genuinely needs a GPU: the offline suite,
    the poster path, every ComfyUI graph, and the placement ladder against
    the live catalog. The request-taking side has its own list: `funapp
    preflight`.

    Exits 0 only when every check PASSes. A check that cannot run is SKIP,
    not PASS -- "could not verify" must never read as "verified".
    """
    from datetime import timezone

    from ai_studio.cli.preflight import Status, run_all, stamp, summarise

    results = run_all(run_suite=not skip_suite)

    table = Table(title="preflight (GPU side)")
    table.add_column("#", justify="right")
    table.add_column("check")
    table.add_column("", justify="center")
    table.add_column("detail", overflow="fold")
    colour = {Status.PASS: "green", Status.FAIL: "red", Status.SKIP: "yellow"}
    for result in results:
        table.add_row(
            str(result.number),
            result.name,
            f"[{colour[result.status]}]{result.status.value}[/{colour[result.status]}]",
            result.detail,
        )
    console.print(table)

    green, summary = summarise(results)
    console.print(f"\n{summary}")
    console.print(f"[dim]{stamp(results, when=datetime.now(timezone.utc)).splitlines()[0]}[/dim]")
    if green:
        console.print("[green]all green: the only thing left unproven is generation.[/green]")
        return
    console.print("[yellow]not green.[/yellow] A skip here is an unknown a live run would spend money discovering.")
    raise typer.Exit(1)


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


_UNDERSTAND_KINDS = {
    "image": MediaKind.IMAGE_UNDERSTAND,
    "audio": MediaKind.AUDIO_UNDERSTAND,
    "video": MediaKind.VIDEO_UNDERSTAND,
}


@app.command()
def understand(
    path: Path = typer.Argument(..., help="Photo/audio/video to describe."),
    kind: str = typer.Option(..., "--kind", "-k", help=f"One of: {', '.join(_UNDERSTAND_KINDS)}."),
    provider: str = typer.Option("stub-understanding", "--provider", "-p"),
    prompt: str | None = typer.Option(
        None, "--prompt", "-q",
        help="A question for the model, sent as typed. Omit for the engineered default.",
    ),
) -> None:
    """Describe one photo/audio/video clip. The offline smoke test for an
    understanding provider -- the `generate`/`--provider stub` of the
    understanding path: no GPU, no RunPod account, no money."""
    try:
        asyncio.run(_understand(path, kind, provider, prompt=prompt))
    except AIStudioError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None


async def _understand(
    path: Path, kind: str, provider_name: str, *, prompt: str | None = None
) -> None:
    from ai_studio.core.understanding_spec import UnderstandingRequest

    modality = _UNDERSTAND_KINDS.get(kind)
    if modality is None:
        raise AIStudioError(f"--kind must be one of {', '.join(_UNDERSTAND_KINDS)}, got {kind!r}")
    if not path.is_file():
        raise AIStudioError(f"no such file: {path}")

    settings = get_settings()
    backend: Any = get_provider(provider_name, modality=modality)
    caps = backend.capabilities()
    console.print(
        f"[bold]{caps.model_id}[/bold]  modality={caps.modality.value}  "
        f"accepts_prompt={caps.accepts_prompt}  cost/call ${caps.cost_per_call_usd:.4f}"
    )

    request = UnderstandingRequest(
        shot_id=new_run_id(), modality=modality, input_media_path=str(path),
        prompt=prompt or None,
    )
    try:
        job = await backend.submit(request)
        console.print(f"submitted [cyan]{job.job_id}[/cyan]")

        waited = 0.0
        poll_s = max(1.0, settings.inference_timeout_s / 6)
        while not job.is_terminal:
            await asyncio.sleep(poll_s)
            waited += poll_s
            if waited > settings.inference_job_timeout_s:
                await backend.cancel(job)
                raise AIStudioError(
                    f"job {job.job_id} exceeded {settings.inference_job_timeout_s}s"
                )
            job = await backend.poll(job)
            console.print(f"  {job.state.value} ({waited:.0f}s)", highlight=False)

        if not job.state.is_success:
            raise AIStudioError(f"understanding failed: {job.error or job.state.value}")

        asset = await backend.fetch(job)
    finally:
        await backend.aclose()

    console.print(
        f"\n[green]{asset.modality.value}[/green]  cost ${asset.cost_usd:.4f}\n{asset.result_text}"
    )


def _setup_logging(service: str) -> None:
    """The one place a process turns logging on: stderr for journald plus the
    JSONL trace under settings.log_dir. Every command that does work calls it
    first (worker, serve, the timers, archive); read-only commands do not,
    so `ai-studio doctor` never creates a logs/ directory."""
    from ai_studio.core.observability import configure_logging

    settings = get_settings()
    configure_logging(service=service, log_dir=settings.log_dir, level=settings.log_level)


_REWRITE_KINDS = ("video", "image", "image-q", "audio-q", "video-q")


@app.command("rewrite")
def rewrite(
    text: str = typer.Argument(..., help="The group member's words, as they would type them."),
    kind: str = typer.Option(..., "--kind", "-k", help=f"One of: {', '.join(_REWRITE_KINDS)}."),
    seconds: float = typer.Option(10.12, "--seconds", help="Clip length, video only."),
) -> None:
    """Run the prompt rewriter on an open pod and print what the model would get.

    The live smoke test of prompts/convert.py, prompts/flux.py and
    prompts/understanding.py against gpt-oss-20b -- no LINE, no render. Needs
    a pod with the inference server up (AI_STUDIO_INFERENCE_URL); evicting
    ComfyUI's checkpoint is left to you (it is the worker's job in service).
    """
    try:
        asyncio.run(_rewrite(text, kind, seconds))
    except AIStudioError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None


async def _rewrite(text: str, kind: str, seconds: float) -> None:
    from ai_studio.inference.client import InferenceClient
    from ai_studio.pipeline.pod_llm import PodLlmClient
    from ai_studio.prompts import flux as flux_prompts
    from ai_studio.prompts import understanding as und
    from ai_studio.prompts.convert import convert
    from ai_studio.prompts.h3 import H3Mode

    if kind not in _REWRITE_KINDS:
        raise AIStudioError(f"--kind must be one of {', '.join(_REWRITE_KINDS)}, got {kind!r}")
    settings = get_settings()
    llm = PodLlmClient(
        InferenceClient(settings.inference_url, timeout_s=settings.inference_timeout_s),
        job_timeout_s=settings.inference_job_timeout_s,
    )
    started = time.monotonic()
    try:
        if kind == "video":
            h3, how = await convert(text, llm, duration_s=seconds, mode=H3Mode.T2VA)
            out = h3.render()
        elif kind == "image":
            fx, how = await flux_prompts.convert(text, llm)
            out = fx.render()
        else:
            modality = {
                "image-q": MediaKind.IMAGE_UNDERSTAND,
                "audio-q": MediaKind.AUDIO_UNDERSTAND,
                "video-q": MediaKind.VIDEO_UNDERSTAND,
            }[kind]
            question, how = await und.convert_question(text, llm, modality=modality)
            out = question or "(no question: caption path)"
    finally:
        await llm.aclose()
    console.print(f"[bold]built_by[/bold] {how}   {time.monotonic() - started:.1f}s")
    console.print(out, highlight=False, markup=False)


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
    until: str | None = typer.Option(None, "--until", help="Lease end as HH:MM; default now + LEASE_HOURS."),
    tz: str = typer.Option(WINDOW_TZ, "--tz", help="Timezone for --until."),
    name: str = typer.Option("ai-studio-window"),
) -> None:
    """Deploy the window's pod by hand. Sets --terminate-after as a backstop.

    Nothing runs this on a timer any more: the worker opens pods on demand
    (`runtime.session.ensure_pod`). This is the operator's manual path — a
    GPU test session, or a pod for `session drain`. The monthly budget guard
    still applies (refuse, or shrink the window, once this month's cap is
    close), and the open is counted against the daily cap like any other.
    """
    _setup_logging("session")
    from datetime import timezone

    from ai_studio.runtime import session as sess
    from ai_studio.runtime.budget import MonthlyBudgetGuard, SpendLedger

    settings = get_settings()
    guard = MonthlyBudgetGuard(
        SpendLedger(),
        cap_usd=settings.max_month_usd,
        vps_monthly_usd=settings.vps_monthly_usd,
    )
    try:
        candidates, network_volume_id = sess.placement()
        guard.refuse_if_broke(candidates)
    except AIStudioError as exc:
        console.print(f"[red]window did not open:[/red] {exc}")
        raise typer.Exit(1) from None

    from ai_studio.runtime import hours

    end = _window_end(until, tz) if until else hours.window_end_for()
    worst_case_hourly = max(tier.usd_per_hr for tier in candidates)
    end = guard.throttle(end, datetime.now(timezone.utc), worst_case_hourly)

    try:
        s = sess.open_session(
            end, name=name, candidates=candidates, network_volume_id=network_volume_id
        )
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
    _setup_logging("close")
    from ai_studio.runtime import session as sess

    terminated = sess.close_session(name=name, reason="scheduled close")
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
    table.add_row("inference", s.inference_url)
    console.print(table)


@session_app.command("reap")
def session_reap(
    image_idle_minutes: int = typer.Option(
        None, help="Grace after an image render (default: runtime.session.IMAGE_IDLE_MINUTES)."
    ),
    video_idle_minutes: int = typer.Option(
        None, help="Grace after a video render (default: runtime.session.VIDEO_IDLE_MINUTES)."
    ),
    understanding_idle_minutes: int = typer.Option(
        None,
        help="Grace after an understanding job (default: "
        "runtime.session.UNDERSTANDING_IDLE_MINUTES).",
    ),
    chat_idle_minutes: int = typer.Option(
        None, help="Grace after a /himonkey reply (default: runtime.session.CHAT_IDLE_MINUTES)."
    ),
    drama_idle_minutes: int = typer.Option(
        None, help="Grace after a /短劇 artifact (default: runtime.session.DRAMA_IDLE_MINUTES)."
    ),
    hold: bool = typer.Option(
        False, "--hold/--no-hold",
        help="Never close: work is about to land on the pod. The caller that owns "
        "the request queue decides this; a bare `session reap` has no queue to ask.",
    ),
) -> None:
    """Close the pod once it has gone quiet. Schedule this every minute.

    The grace depends on what the pod last rendered. Pass --hold when work
    is waiting for the pod — it is then never closed, whatever the clock says.
    """
    _setup_logging("reap")
    from ai_studio.runtime import session as sess

    decision = sess.close_if_idle(
        image_idle_minutes=image_idle_minutes or sess.IMAGE_IDLE_MINUTES,
        video_idle_minutes=video_idle_minutes or sess.VIDEO_IDLE_MINUTES,
        understanding_idle_minutes=(
            understanding_idle_minutes or sess.UNDERSTANDING_IDLE_MINUTES
        ),
        chat_idle_minutes=chat_idle_minutes or sess.CHAT_IDLE_MINUTES,
        drama_idle_minutes=drama_idle_minutes or sess.DRAMA_IDLE_MINUTES,
        hold=hold,
    )
    console.print(str(decision))
    sess.log_reap(decision)


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
