"""ai-studio command line."""

from __future__ import annotations

import asyncio
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

    import shutil as _shutil

    usage = _shutil.disk_usage(settings.files_dir if settings.files_dir.exists() else ".")
    free_gb = usage.free / 1_073_741_824
    table.add_row(
        "disk free",
        _mark(free_gb >= 5.0, warn_only=True),
        f"{free_gb:.0f} GB free of {usage.total / 1_073_741_824:.0f} GB "
        f"(retention {settings.files_retention_days:.0f}d, `ai-studio gc`)",
    )
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
    from ai_studio.pipeline.queue import JobQueue
    from ai_studio.storage.retention import sweep_old_files

    settings = get_settings()
    max_age = settings.files_retention_days if days is None else days
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
    from ai_studio.storage.index import index_path

    protected = protected | {str(index_path(settings.files_dir).resolve())}
    sweep_targets = [("files", settings.files_dir), ("incoming", settings.incoming_dir)]
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
    console.print(
        f"[green]gc[/green] {'(dry run) ' if dry_run else ''}"
        f"{total_removed} file(s), {total_freed / 1_048_576:.1f} MB, "
        f"older than {max_age:.0f}d"
    )


@app.command("archive")
def archive(
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan and report; write and delete nothing."),
) -> None:
    """Snapshot, compress, verify, then prune. Schedule this daily (03:00 Asia/Taipei).

    Tars the JSONL traces (days before today), session and pod records, drama
    state/manifests, the spend ledger and files/index.jsonl plus a consistent
    sqlite backup of the queue into archive/<day>/*.tar.zst with a manifest
    of sha256s; then deletes hot logs older than AI_STUDIO_LOG_HOT_DAYS
    (only what a manifest names), archives older than
    AI_STUDIO_ARCHIVE_KEEP_DAYS, stale dry-run/stub/out files, empty drama
    dirs and old chat_turns; then folds any real render since the last run
    into runs/benchmark/<month>.json, a durable per-GPU-tier performance
    aggregate (see docs/observability.md). Idempotent: a second run the
    same day only prunes and has nothing new to fold.
    """
    _setup_logging("archive")
    from ai_studio.storage.archive import run_archive

    settings = get_settings()
    result = run_archive(
        root=Path.cwd(),
        log_dir=settings.log_dir,
        runs_dir=settings.runs_dir,
        files_dir=settings.files_dir,
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


@app.command("preflight")
def preflight_cmd(
    skip_suite: bool = typer.Option(
        False, "--skip-suite", help="Skip check 1 (pytest/ruff/lint-imports/mypy)."
    ),
    push: bool = typer.Option(
        False, "--push", help="Check 5 only: actually send a message to the real group."
    ),
) -> None:
    """Run the nine pre-launch checks. Nothing here opens a pod.

    This is what replaces the stub: everything provable without spending a
    GPU-second, proved, so the one affordable live run is spent only on the
    part that genuinely needs a GPU. See PLAN.md Phase 4.

    Exits 0 only when all nine PASS. A check that cannot run is SKIP, not PASS
    -- "could not verify" must never read as "verified" -- so offline you
    should expect several skips and a non-zero exit, which is the honest
    answer: Phase 4 is not complete off the VM.

    `--push` sends a real message to the real group and spends real quota. It
    is opt-in rather than merely credential-gated for exactly that reason.
    """
    from datetime import timezone

    from ai_studio.cli.preflight import Status, run_all, stamp, summarise

    results = run_all(run_suite=not skip_suite, send_push=push)

    table = Table(title="preflight (PLAN.md Phase 4)")
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
        console.print("[green]all nine green: the only thing left unproven is generation.[/green]")
        return
    console.print(
        "[yellow]not green.[/yellow] Phase 7 spends the one affordable run; "
        "a skip here is an unknown it would spend that run discovering."
    )
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


DRYRUN_SCREENPLAY: dict[str, Any] = {
    # The canned screenwriter replies `drama-dryrun` feeds through the real
    # `prompts.drama` parser, so the offline run exercises validation too.
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
        "beats": [
            "She lifts the stall's shutter before dawn and finds an envelope taped underneath.",
            "She reads it between customers: the market closes tomorrow.",
            "A regular asks what is wrong; she says nothing is.",
            "Evening: she looks down the row of stalls packing up early.",
            "She writes a reply on the back of the letter.",
            "Dawn again: she tapes her reply where the first one was, and opens as usual.",
        ],
        "overall_soundscape": "Sizzling oil, a crowd murmuring, scooters passing on the road behind.",
        "non_diegetic_music": "N/A",
    },
    "shots": [
        {"index": 1, "scene": "a night-market stall before dawn, shutter half up, string lights off", "framing": "medium", "action": "the lead crouches and peels an envelope from under the counter", "camera": {"motion": "push_in", "amplitude": "small", "speed": "slow"}},
        {"index": 2, "scene": "the same stall, mid-evening, steam from the wok, a paper letter in hand", "framing": "close-up", "action": "the lead reads the letter and her hands go still", "camera": {"motion": "static_shot"}},
        {"index": 3, "scene": "the stall counter, a regular customer's shoulder in the foreground", "framing": "over-the-shoulder", "action": "the lead answers with a small shake of the head", "camera": {"motion": "static_shot"}, "dialogue": [{"speaker_id": "S1", "identity": "the lead", "language": "Mandarin Chinese", "text": "沒事,明天照常開。"}]},
        {"index": 4, "scene": "the market row at night, neighbouring stalls stacking crates", "framing": "wide", "action": "the lead stands at her counter looking down the row", "camera": {"motion": "pan_right", "speed": "slow"}, "cut_reason": "time_passing"},
        {"index": 5, "scene": "the counter under one work lamp, the letter turned face down, a pen", "framing": "medium close-up", "action": "the lead writes on the back of the letter", "camera": {"motion": "push_in", "amplitude": "small", "speed": "slow"}},
        {"index": 6, "scene": "the stall before dawn again, shutter going up, first light", "framing": "close-up", "action": "the lead tapes the letter under the counter and stands", "camera": {"motion": "tilt_up", "speed": "slow"}, "cut_reason": "time_passing"},
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
    Flux and H3 (ffmpeg testsrc2), real loudnorm + concat. Proves the state
    file, the resume rule and the assembly with no pod and no money. Run it
    twice: the second run must render nothing."""
    try:
        asyncio.run(_drama_dryrun(premise, out, runs, screenplay))
    except AIStudioError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None


async def _drama_dryrun(premise: str, out: Path, runs: Path, screenplay_file: Path | None) -> None:
    import json as _json
    from datetime import timedelta, timezone

    from ai_studio.llm.scripted import ScriptedLlmClient
    from ai_studio.pipeline.drama import load_state, render_drama
    from ai_studio.pipeline.queue import JobQueue
    from ai_studio.prompts.drama import screenplay_payload, write_screenplay

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
        accepted, _ = queue.enqueue("dryrun", "Cdryrun", premise, media_kind=MediaKind.DRAMA)
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
    console.print(f"  state: {runs / 'drama' / job.token / 'state.json'}  (run again: nothing re-renders)")


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


line_app = typer.Typer(help="LINE bot: serve the webhook, or discover a group id.")
app.add_typer(line_app, name="line")


def _setup_logging(service: str) -> None:
    """The one place a process turns logging on: stderr for journald plus the
    JSONL trace under settings.log_dir. Every command that does work calls it
    first (worker, serve, the timers, archive); read-only commands do not,
    so `ai-studio doctor` never creates a logs/ directory."""
    from ai_studio.core.observability import configure_logging

    settings = get_settings()
    configure_logging(service=service, log_dir=settings.log_dir, level=settings.log_level)


def _run_server(host: str, port: int, reload: bool = False) -> None:
    import uvicorn

    from ai_studio.api.main import create_app

    # log_config=None keeps uvicorn from calling dictConfig, which clears every
    # existing handler and defines no root logger - so a config set up here
    # would be silently discarded and ai-studio's own lines would never appear.
    # Owning the config instead means our INFO lines and uvicorn's both show up,
    # here and under journalctl on the host.
    _setup_logging("webhook")
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
    # The trigger is CJK and has no ASCII alias any more (one spelling per
    # trigger, see bots.line.webhook). On a cp950 console it may print as
    # mojibake; the docs carry the same string for copy-pasting.
    console.print("  3. say [cyan]/影片 test[/cyan] in the group")
    console.print("  the group id will be printed here, and replied into the chat")
    console.print("")
    _run_server(host, port)


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
    _log_reap(decision)


_REAP_LAST = Path("runs/.reap_last.json")


def _log_reap(decision: Any) -> None:
    """DEBUG every minute (JSONL only), INFO only when the action changes.

    The reaper fires every minute and used to be ~65 % of ai-studio's journald
    volume saying `held: work pending` (📏 2026-08-28); the transitions are
    the information, the repeats are not."""
    import json
    import logging

    log = logging.getLogger("ai_studio.reap")
    fields = {
        "action": getattr(decision, "action", None), "idle_min": round(getattr(decision, "idle_min", 0.0), 1),
        "grace": getattr(decision, "grace", None), "spent": round(getattr(decision, "spent_usd", 0.0), 2),
        "pod_id": getattr(decision, "pod_id", None),
    }
    previous = None
    try:
        previous = json.loads(_REAP_LAST.read_text(encoding="utf-8")).get("action")
    except Exception:
        previous = None
    if fields["action"] != previous:
        log.info("reap: %s", decision, extra=fields)
        try:
            _REAP_LAST.parent.mkdir(parents=True, exist_ok=True)
            _REAP_LAST.write_text(json.dumps({"action": fields["action"]}), encoding="utf-8")
        except OSError:
            pass
    else:
        log.debug("reap: %s", decision, extra=fields)


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
    _setup_logging("session")
    from ai_studio.inference.client import InferenceClient
    from ai_studio.pipeline.drain import drain_window
    from ai_studio.pipeline.pod_llm import PodLlmClient
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
    # Understanding and chat jobs share this pod's one FIFO queue -- a manual
    # drain that omitted them would KeyError the moment one was claimed.
    understand_backends = {
        kind: get_provider(name, base_url=session.inference_url, hourly_usd=session.cost_per_hr)
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
                files_dir=settings.files_dir,
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

    `pipeline` sits below `runtime` in the layer contract, so the worker loop
    cannot import business hours, sessions or the provider registry. It takes
    them by protocol instead, and this is where the two halves are joined —
    which is exactly what the CLI is for, and why it is the one package
    exempted from the "phase-2 packages stay leaves" contract.
    """

    def __init__(self, *, name: str, poll_seconds: float, push: object | None = None) -> None:
        from ai_studio.bots.line.push import LinePushClient, NullPushClient

        self.name = name
        self.poll_seconds = poll_seconds
        self._providers: dict[str, dict[MediaKind, object]] = {}
        self._llms: dict[str, object] = {}

        settings = get_settings()
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
            console.print(f"  provisioning {live.pod_id} (pod_setup.sh over ssh)")
            sess.provision(live)
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
            MediaKind.IMAGE_UNDERSTAND: get_provider(
                "understand-image", base_url=live.inference_url, hourly_usd=live.cost_per_hr,
            ),
            MediaKind.AUDIO_UNDERSTAND: get_provider(
                "understand-audio", base_url=live.inference_url, hourly_usd=live.cost_per_hr,
            ),
            MediaKind.VIDEO_UNDERSTAND: get_provider(
                "understand-video", base_url=live.inference_url, hourly_usd=live.cost_per_hr,
            ),
            MediaKind.CHAT: get_provider(
                "chat", base_url=live.inference_url, hourly_usd=live.cost_per_hr,
            ),
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

    def touch_activity(self, media_kind: str) -> None:
        from ai_studio.runtime import session as sess

        sess.touch_activity(media_kind)

    async def deliver(self, job: Any, asset: Path | None) -> str:
        """Push the finished media into the group that asked for it, @-ing them.

        The poster is built here rather than in `push.py` because it is an
        ffmpeg call and `bots` has no business shelling out. If it fails, the
        delivery degrades to text and a link rather than being abandoned — a
        thumbnail is not worth losing a clip that cost GPU-minutes over.
        """
        from ai_studio.bots.line import push as line_push

        status_url = f"{self.base_url}/q/{job.token}"

        if job.result_text:
            # An understanding job (/說圖 /說音 /說影): text, no media object
            # and no poster -- `asset` is always None for these.
            text = job.result_text
            if job.media_kind is MediaKind.IMAGE_UNDERSTAND:
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
                preview = media.poster(asset, self.files_dir / f"{asset.stem}_poster.jpg")
            except AIStudioError as exc:
                console.print(f"[yellow]no poster for {asset.name}:[/yellow] {exc}")
            else:
                messages = line_push.delivered_messages(
                    media_url=f"{self.base_url}/files/{asset.name}",
                    preview_url=f"{self.base_url}/files/{preview.name}",
                    status_url=status_url,
                    is_video=job.media_kind in (MediaKind.VIDEO, MediaKind.DRAMA),
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
    if job.media_kind is not MediaKind.DRAMA:
        return None
    screenplay = (job.prompt or {}).get("screenplay") or {}
    title = str(screenplay.get("title") or job.text[:20])
    logline = str(screenplay.get("logline") or "")
    return f"🎭《{title}》完成了\n{logline[:80]}".rstrip()


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

    Closing is still someone else's job, on purpose: `session reap` every five
    minutes, `session close` at the bell, and `--terminate-after` on the pod
    itself. Three independent ways for a machine to stop billing, none of which
    depends on this process still being alive.
    """
    _setup_logging("worker")
    from ai_studio.pipeline.queue import JobQueue
    from ai_studio.pipeline.worker import serve

    settings = get_settings()
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
