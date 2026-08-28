"""Render a `/短劇`: one screenplay -> character sheet -> six keyframes -> six
clips -> one levelled, concatenated minute.

Why this is its own module rather than a loop around `render_clip`: a drama is
**fourteen GPU submissions and two checkpoint swaps under one queue row**, and
the expensive mistakes are all about ordering and re-payment:

- **Stage order is checkpoint order.** Every Flux job runs before any H3 job, so
  ComfyUI swaps its resident model twice (Flux, then H3), not twelve times.
  `make_room_for` is called once per side.
- **Every artifact is on disk with its sha256 before the stage advances.** The
  state file (`runs/drama/<token>/state.json`) is the resume point: a lease
  end, a requeue or a worker restart re-renders only what is missing or
  corrupt. A drama that dies after clip 4 costs clips 5 and 6 on the next
  attempt, not all six again.
- **Two gates before any spend.** A cost gate against `AI_STUDIO_MAX_COST_USD`
  (the per-run ceiling, finally used on the LINE path) and, before each GPU
  submit, a time gate against the pod's lease -- a render that the lease would
  cut short is not started; the job goes back to the queue with its state
  intact and finishes in the next window.
- **The face is re-anchored every shot.** Keyframes are image-to-image from the
  *character sheet* -- never from the previous shot's last frame -- with the
  anchor string verbatim in the prompt, so drift cannot accumulate across
  shots. FaceDetailer runs on the stills only, when the pod has it, and what
  actually happened is written into the state (`face_repair`).

Sits in `pipeline` (L4): may import `media`, `prompts`, `core`; never `runtime`
or `bots`. The pod's lease deadline and the activity callback arrive as
arguments for the same reason they do in `worker.py`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_studio import media
from ai_studio.config.fun_settings import get_fun_settings
from ai_studio.config.settings import get_settings
from ai_studio.core.drama_spec import ArtifactRecord, DramaState, Screenplay
from ai_studio.core.enums import GenMode, MediaKind
from ai_studio.core.errors import AIStudioError, CostCeilingExceeded, DramaResume, ProviderError
from ai_studio.core.image_provider_spec import ImageRequest
from ai_studio.core.observability import utc_now_iso
from ai_studio.core.provider_spec import ClipRequest
from ai_studio.pipeline.convert_worker import DEFAULT_DURATION_S, snap_frames
from ai_studio.pipeline.queue import Job
from ai_studio.pipeline.residency import make_room_for
from ai_studio.prompts.drama import character_sheet_prompts, h3_prompt
from ai_studio.storage.base import sha256_file

_log = logging.getLogger("ai_studio.drama")

STAGE_RESERVE_S: dict[str, float] = {"image": 90.0, "video": 6 * 60.0}
"""Seconds that must remain on the lease before a submit of that kind.

`[speculative]`, from the measured single-job figures: a Flux still is ~30 s
warm and an H3 clip 79-215 s plus fetch, so a submit with less than this left
would be cut off by `--terminate-after` and paid for twice. Sized generously
on purpose -- stopping early costs a resume, running late costs a clip."""

CHARACTER_VIEWS = ("front", "three_quarter")
"""The sheet is two stills; `front` is what every keyframe repaints from."""

OnActivity = Callable[[], None] | None


async def render_drama(
    job: Job,
    providers: dict[MediaKind, Any],
    *,
    files_dir: Path,
    runs_dir: Path,
    deadline: datetime,
    poll_interval_s: float,
    on_activity: OnActivity = None,
) -> Path:
    """Render the whole drama for `job`; return the finished mp4 under `files_dir`.

    Raises `ProviderError` (requeue: the lease ran out, the backend failed) or
    `AIStudioError` (terminal: no screenplay, over the cost ceiling, a
    provider this pod does not serve). Resumable: call again with the same
    `job` and it continues from the state file.
    """
    screenplay = _screenplay_of(job)
    image = providers.get(MediaKind.IMAGE)
    clip = providers.get(MediaKind.VIDEO)
    if image is None or clip is None:
        raise AIStudioError("a drama needs both the image and the video provider on this pod")
    image_caps = image.capabilities()
    clip_caps = clip.capabilities()

    run_dir = Path(runs_dir) / "drama" / job.token
    run_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(run_dir)
    settings = get_settings()
    fun = get_fun_settings()

    width, height = clip_caps.native_width, clip_caps.native_height
    fps = clip_caps.native_fps
    frames = snap_frames(round(DEFAULT_DURATION_S * fps))
    clip_s = frames / fps

    # -- cost gate: what is still to be spent, against the per-run ceiling
    still_to_render = _count_missing(state, len(screenplay.shots))
    if any(still_to_render.values()) and (state.character or state.keyframes or state.clips):
        _log.info("resuming drama", extra={"stage": "drama", "reason": str(still_to_render),
                                           "cost_usd": state.spent_usd})
    estimate = (
        still_to_render["images"] * float(image_caps.cost_per_image_usd)
        + still_to_render["clips"] * float(clip_caps.estimated_cost_usd(clip_s))
    )
    if estimate + state.spent_usd > settings.max_cost_usd:
        raise CostCeilingExceeded(
            f"drama {job.token}: ~${estimate:.2f} still to render + ${state.spent_usd:.2f} "
            f"spent exceeds AI_STUDIO_MAX_COST_USD=${settings.max_cost_usd:.2f}"
        )

    def touch() -> None:
        if on_activity is not None:
            on_activity()

    # -- stage 1 + 2: everything Flux, one checkpoint residency
    if still_to_render["images"]:
        await make_room_for(MediaKind.IMAGE, providers)
        state.stage_start("character", utc_now_iso())

    for view, prompt in character_sheet_prompts(screenplay.anchor).items():
        if _fresh(state.character.get(view)):
            continue
        _require_time(deadline, "image")
        request = ImageRequest(
            shot_id=f"job{job.id}_char_{view}",
            mode=GenMode.T2I,
            prompt=prompt,
            width=width,
            height=height,
            seed=_seed(job.id, "character", view),
        )
        _t0 = time.monotonic()
        record, _ = await _render_image(
            image, request, run_dir / "character" / f"{view}.png", deadline, poll_interval_s
        )
        state.character[view] = record
        state.add_cost(record.cost_usd)
        _log.info("character %s", view, extra={"stage": "character", "sha256": record.sha256[:12],
                                               "cost_usd": record.cost_usd, "seconds": round(time.monotonic() - _t0, 1)})
        save_state(run_dir, state)
        touch()

    reference = Path(state.character["front"].path)
    state.stage_finish("character", utc_now_iso())
    state.stage_start("keyframes", utc_now_iso())
    for shot in screenplay.shots:
        key = str(shot.index)
        if _fresh(state.keyframes.get(key)):
            continue
        _require_time(deadline, "image")
        _t0 = time.monotonic()
        extra: dict[str, Any] = {"denoise": fun.drama_keyframe_denoise}
        # Once this pod's face graph has failed, stop asking: a requeued drama
        # would otherwise pay the failed attempt again on every keyframe.
        if fun.drama_face_repair and not state.face_repair.startswith("failed"):
            extra["face_repair"] = True
        request = ImageRequest(
            shot_id=f"job{job.id}_kf{key}",
            mode=GenMode.I2I,
            prompt=shot.keyframe_prompt,
            width=width,
            height=height,
            seed=_seed(job.id, "keyframe", key),
            source_image_path=str(reference),
            extra=extra,
        )
        dest = run_dir / "keyframes" / f"shot_{key}.png"
        try:
            record, raw = await _render_image(image, request, dest, deadline, poll_interval_s)
        except ProviderError as exc:
            if not extra.get("face_repair"):
                raise
            # The face graph itself is [speculative] on this pod. One clean
            # retry without it beats failing a whole drama over a detailer.
            _log.warning("keyframe %s face-repair graph failed (%s); retrying plain i2i", key, exc)
            plain = request.model_copy(update={"extra": {"denoise": extra["denoise"]}})
            record, raw = await _render_image(image, plain, dest, deadline, poll_interval_s)
            state.face_repair = f"failed: {str(exc)[:120]}"
        else:
            if not fun.drama_face_repair:
                state.face_repair = "off: AI_STUDIO_DRAMA_FACE_REPAIR=false"
            elif raw.get("face_repair"):
                state.face_repair = "applied"
            elif not state.face_repair.startswith("failed"):
                state.face_repair = "skipped: pod has no FaceDetailer nodes"
        state.keyframes[key] = record
        state.add_cost(record.cost_usd)
        _log.info("keyframe %s/%d", key, len(screenplay.shots),
                  extra={"stage": "keyframe", "sha256": record.sha256[:12], "cost_usd": record.cost_usd,
                         "seconds": round(time.monotonic() - _t0, 1), "reason": state.face_repair})
        save_state(run_dir, state)
        touch()

    state.stage_finish("keyframes", utc_now_iso())
    # -- stage 3: everything H3, one checkpoint residency
    if still_to_render["clips"] or not all(_fresh(state.clips.get(str(s.index))) for s in screenplay.shots):
        await make_room_for(MediaKind.VIDEO, providers)

    state.stage_start("clips", utc_now_iso())
    for shot in screenplay.shots:
        key = str(shot.index)
        if _fresh(state.clips.get(key)):
            continue
        _require_time(deadline, "video")
        _t0 = time.monotonic()
        prompt = h3_prompt(shot, screenplay, duration_s=clip_s).render()
        clip_request = ClipRequest(
            shot_id=f"job{job.id}_shot{key}",
            mode=GenMode.I2V,
            prompt=prompt,
            width=width,
            height=height,
            duration_s=clip_s,
            fps=fps,
            seed=_seed(job.id, "clip", key),
            first_frame_path=state.keyframes[key].path,
        )
        record = await _render_clip(
            clip, clip_request, run_dir / "clips" / f"shot_{key}.mp4", deadline, poll_interval_s
        )
        state.clips[key] = record
        state.add_cost(record.cost_usd)
        _log.info("clip %s/%d", key, len(screenplay.shots),
                  extra={"stage": "clip", "sha256": record.sha256[:12], "cost_usd": record.cost_usd,
                         "seconds": round(time.monotonic() - _t0, 1)})
        save_state(run_dir, state)
        touch()

    # -- stage 4 + 5: CPU only. No GPU-time gate; the pod may already be gone.
    state.stage_finish("clips", utc_now_iso())
    state.stage_start("level", utc_now_iso())
    for shot in screenplay.shots:
        key = str(shot.index)
        if _fresh(state.leveled.get(key)):
            continue
        dest = run_dir / "leveled" / f"shot_{key}.mp4"
        argvs = media.normalize_loudness(Path(state.clips[key].path), dest)
        state.ffmpeg_argv.extend(argvs)
        state.leveled[key] = ArtifactRecord(path=str(dest), sha256=sha256_file(dest), created_at=utc_now_iso())
        save_state(run_dir, state)
        _log.info("leveled %s", key, extra={"stage": "level", "sha256": state.leveled[key].sha256[:12]})

    output = Path(files_dir) / f"{job.token}.mp4"
    state.stage_finish("level", utc_now_iso())
    state.stage_start("concat", utc_now_iso())
    if not _fresh(state.output):
        ordered = [Path(state.leveled[str(s.index)].path) for s in screenplay.shots]
        argv = media.concat(ordered, output)
        state.ffmpeg_argv.append(argv)
        state.output = ArtifactRecord(path=str(output), sha256=sha256_file(output), created_at=utc_now_iso())
        save_state(run_dir, state)
        _log.info("drama assembled", extra={"stage": "concat", "sha256": state.output.sha256[:12],
                                            "cost_usd": state.spent_usd})
        state.stage_finish("concat", utc_now_iso())
        save_state(run_dir, state)
        (run_dir / "render_manifest.json").write_text(
            json.dumps(
                {
                    "generated_at": utc_now_iso(),
                    "token": job.token,
                    "job_id": job.id,
                    "spent_usd": state.spent_usd,
                    "face_repair": state.face_repair,
                    "stages": {k: v.model_dump() for k, v in state.stages.items()},
                    "ffmpeg": state.ffmpeg_argv,
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        touch()
    return output


# --------------------------------------------------------------------- stages


async def _render_image(
    provider: Any, request: ImageRequest, dest: Path, deadline: datetime, poll_s: float
) -> tuple[ArtifactRecord, dict[str, Any]]:
    image_job = await provider.submit(request)
    image_job = await _await_terminal(provider, image_job, deadline, poll_s, what="still")
    asset = await provider.fetch(image_job, dest)
    # Hashed from the file on disk, not copied from the asset: `_fresh` will
    # compare against exactly this, and the file is the thing that must match.
    record = ArtifactRecord(
        path=str(dest), sha256=sha256_file(dest), created_at=utc_now_iso(), cost_usd=float(asset.cost_usd), job_id=image_job.job_id
    )
    return record, dict(getattr(image_job, "raw", {}) or {})


async def _render_clip(
    provider: Any, request: ClipRequest, dest: Path, deadline: datetime, poll_s: float
) -> ArtifactRecord:
    clip_job = await provider.submit(request)
    clip_job = await _await_terminal(provider, clip_job, deadline, poll_s, what="clip")
    asset = await provider.fetch(clip_job, dest)
    return ArtifactRecord(
        path=str(dest), sha256=sha256_file(dest), created_at=utc_now_iso(), cost_usd=float(asset.cost_usd), job_id=clip_job.job_id
    )


async def _await_terminal(provider: Any, job: Any, deadline: datetime, poll_s: float, *, what: str) -> Any:
    """The one submit-side wait loop: poll until terminal, cancel at the bell."""
    while not job.is_terminal:
        if datetime.now(timezone.utc) >= deadline:
            await provider.cancel(job)
            raise DramaResume(f"window closed while a drama {what} was rendering; resuming next window")
        await asyncio.sleep(poll_s)
        job = await provider.poll(job)
    if not job.state.is_success:
        raise ProviderError(f"drama {what} failed: {job.error or job.state.value}")
    return job


# ---------------------------------------------------------------------- gates


def _require_time(deadline: datetime, kind: str) -> None:
    """Refuse to *start* a GPU job the lease would cut short. `DramaResume`
    so the job is requeued -- with its state file intact and its attempt
    handed back -- not failed."""
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining < STAGE_RESERVE_S[kind]:
        _log.info("paused at %s: %.0fs left on the lease", kind, remaining,
                  extra={"stage": kind, "outcome": "paused", "seconds": round(remaining)})
        raise DramaResume(
            f"lease ends in {remaining:.0f}s, under the {STAGE_RESERVE_S[kind]:.0f}s a "
            f"drama {kind} needs; resuming next window"
        )


def _count_missing(state: DramaState, shots: int) -> dict[str, int]:
    images = sum(1 for v in CHARACTER_VIEWS if not _fresh(state.character.get(v)))
    images += sum(1 for i in range(1, shots + 1) if not _fresh(state.keyframes.get(str(i))))
    clips = sum(1 for i in range(1, shots + 1) if not _fresh(state.clips.get(str(i))))
    return {"images": images, "clips": clips}


# ---------------------------------------------------------------------- state


def load_state(run_dir: Path) -> DramaState:
    path = Path(run_dir) / "state.json"
    if not path.is_file():
        return DramaState()
    try:
        return DramaState.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        # A corrupt state file must not silently reset the spend record; the
        # artifacts on disk are still found by sha, so nothing is re-paid --
        # but the money already spent would be forgotten. Say so.
        raise AIStudioError(f"{path} is not a readable drama state: {exc}") from exc


def save_state(run_dir: Path, state: DramaState) -> None:
    now = utc_now_iso()
    state.updated_at = now
    if not state.created_at:
        state.created_at = now
    path = Path(run_dir) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def _fresh(record: ArtifactRecord | None) -> bool:
    """An artifact counts only if the file is there and still hashes right."""
    if record is None:
        return False
    path = Path(record.path)
    return path.is_file() and sha256_file(path) == record.sha256


def _screenplay_of(job: Job) -> Screenplay:
    plan = job.prompt or {}
    raw = plan.get("screenplay")
    if not raw:
        raise AIStudioError("drama job has no screenplay; conversion did not run")
    try:
        return Screenplay.model_validate(raw)
    except ValueError as exc:
        raise AIStudioError(f"stored screenplay is invalid: {exc}") from exc


def _seed(*parts: object) -> int:
    """Deterministic per-shot seed so a resume re-renders the same shot."""
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")
