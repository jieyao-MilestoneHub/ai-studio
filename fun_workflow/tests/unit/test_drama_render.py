"""`pipeline.drama.render_drama`: the stage machine, its two gates, and the
resume contract. Every assertion is about not paying twice or not paying for
something the lease would throw away.

Providers are fakes that write real bytes (the resume rule hashes files);
ffmpeg is monkeypatched out -- `test_media_concat.py` covers the real argv.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from ai_studio import media
from ai_studio.core.enums import GenMode, MediaKind
from ai_studio.core.enums import JobState as ClipState
from ai_studio.core.errors import AIStudioError, CostCeilingExceeded, ProviderError
from ai_studio.core.image_provider_spec import ImageAsset, ImageJob, ImageProviderCapabilities
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ProviderCapabilities
from ai_studio.llm.scripted import ScriptedLlmClient

from fun_workflow.core.kinds import JobKind
from fun_workflow.pipeline import drama
from fun_workflow.pipeline.queue import JobQueue
from fun_workflow.prompts.drama import screenplay_payload

# The same canned screenwriter replies as test_drama_prompt.py, inlined: the
# tests directory is not a package, so nothing here can import from a sibling.
APPEARANCE = "25-year-old Asian woman, oval face, small mole under right eye, dark chin-length straight hair"
OUTLINE = {
    "title": "夜市的信",
    "logline": "A night-market stall owner finds a letter that says the market closes tomorrow.",
    "style": "Live-action, cinematic",
    "anchor": {"name": "阿玲", "appearance": APPEARANCE,
               "wardrobe": "a faded red apron over a white t-shirt", "voice": "soft, low"},
    "world": {"location": "a narrow night-market food stall facing one row of stalls",
              "light": "warm tungsten string lights from above left", "signature_prop": "a folded paper letter"},
    "beats": {b: f"{b} beat" for b in ("hook", "setup", "conflict", "turn", "payoff", "cliffhanger")},
    "overall_soundscape": "Sizzling oil, a crowd murmuring.",
    "non_diegetic_music": "N/A",
}
FRAMINGS = {
    1: ["close-up", "medium"], 2: ["wide", "close-up"], 3: ["over-the-shoulder"],
    4: ["wide", "medium close-up"], 5: ["close-up", "medium"], 6: ["wide"],
}


def _shots(indices: list[int]) -> dict:
    return {"shots": [
        {"index": i, "scene": f"the night market stall, beat {i}",
         "cut_reason": "time_passing" if i == 4 else "default",
         "sub_shots": [{"framing": f, "action": f"the lead does thing {i}.{k}", "camera": {"motion": "static_shot"},
                        **({"line": "沒事。"} if i == 3 else {})}
                       for k, f in enumerate(FRAMINGS[i], start=1)]}
        for i in indices
    ]}


def _good_client() -> ScriptedLlmClient:
    return ScriptedLlmClient(*(json.dumps(r, ensure_ascii=False) for r in (OUTLINE, _shots([1, 2, 3]), _shots([4, 5, 6]))))

CLIP_CAPS = ProviderCapabilities(
    provider="fake", model_id="fake-h3", native_width=864, native_height=480, native_fps=24,
    modes=frozenset({GenMode.T2V, GenMode.I2V}), min_clip_s=5.0, max_clip_s=15.0,
    has_native_audio=True, cost_per_second_usd=0.005,
)
IMAGE_CAPS = ImageProviderCapabilities(
    provider="fake-flux", model_id="fake-flux-dev", native_width=1024, native_height=1024,
    modes=frozenset({GenMode.T2I, GenMode.I2I}), output_format="png", cost_per_image_usd=0.006,
)


class Ledger:
    """Shared across both fakes so the *order* of submits is observable."""

    def __init__(self) -> None:
        self.events: list[str] = []


class FakeClipProvider:
    residency_group = "comfyui"
    def __init__(self, ledger: Ledger, *, fail_on: set[str] | None = None) -> None:
        self.ledger = ledger
        self.fail_on = fail_on or set()
        self.requests: list[Any] = []

    def capabilities(self) -> ProviderCapabilities:
        return CLIP_CAPS

    async def evict(self) -> None:
        self.ledger.events.append("evict:video")

    async def submit(self, request: Any) -> ClipJob:
        self.requests.append(request)
        self.ledger.events.append(f"clip:{request.shot_id}")
        if request.shot_id in self.fail_on:
            raise ProviderError(f"boom {request.shot_id}")
        return ClipJob(
            provider="fake", job_id=f"j-{request.shot_id}", shot_id=request.shot_id,
            state=ClipState.COMPLETED, submitted_at=0.0, updated_at=0.0,
        )

    async def poll(self, job: ClipJob) -> ClipJob:
        return job

    async def fetch(self, job: ClipJob, dest: Path) -> ClipAsset:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"mp4:" + job.shot_id.encode())
        return ClipAsset(
            shot_id=job.shot_id, key=Path(dest).name, sha256="0" * 64, size_bytes=8,
            width=864, height=480, fps=24.0, duration_s=10.125, has_audio=True,
            provider="fake", job_id=job.job_id, cost_usd=0.03,
        )

    async def cancel(self, job: ClipJob) -> None:
        self.ledger.events.append("cancel")


class FakeImageProvider:
    residency_group = "comfyui"
    def __init__(self, ledger: Ledger, *, face_nodes: bool = True, face_fails: bool = False) -> None:
        self.ledger = ledger
        self.face_nodes = face_nodes
        self.face_fails = face_fails
        self.requests: list[Any] = []

    def capabilities(self) -> ImageProviderCapabilities:
        return IMAGE_CAPS

    async def evict(self) -> None:
        self.ledger.events.append("evict:image")

    async def submit(self, request: Any) -> ImageJob:
        self.requests.append(request)
        self.ledger.events.append(f"image:{request.shot_id}")
        wants_face = bool(request.extra.get("face_repair"))
        if wants_face and self.face_fails:
            raise ProviderError("FaceDetailer exploded")
        return ImageJob(
            provider="fake-flux", job_id=f"i-{request.shot_id}", shot_id=request.shot_id,
            state=ClipState.COMPLETED, submitted_at=0.0, updated_at=0.0,
            raw={"face_repair": wants_face and self.face_nodes},
        )

    async def poll(self, job: ImageJob) -> ImageJob:
        return job

    async def fetch(self, job: ImageJob, dest: Path) -> ImageAsset:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"png:" + job.shot_id.encode())
        return ImageAsset(
            shot_id=job.shot_id, key=Path(dest).name, sha256="0" * 64, size_bytes=8,
            width=864, height=480, format="png", provider="fake-flux", job_id=job.job_id,
            cost_usd=0.006,
        )

    async def cancel(self, job: ImageJob) -> None:
        self.ledger.events.append("cancel")


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake_normalize(src: Path, dest: Path, **_: Any) -> list[list[str]]:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(src).read_bytes() + b":lvl")
        calls.append(f"level:{Path(src).name}")
        return [["ffmpeg", "measure"], ["ffmpeg", "apply", "linear=true"]]

    def fake_assemble(clips: list[Path], dest: Path, **kw: Any) -> list[str]:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"".join(Path(c).read_bytes() for c in clips))
        calls.append("assemble:" + ",".join(Path(c).name for c in clips))
        calls.append(f"boundaries:{','.join(b.kind for b in kw['boundaries'])}")
        calls.append(f"total_s:{kw['total_s']:.3f}")
        calls.append(f"subtitles:{Path(kw['subtitles']).name}")
        return ["ffmpeg", "assemble"]

    monkeypatch.setattr(media, "normalize_loudness", fake_normalize)
    monkeypatch.setattr(media, "assemble", fake_assemble)
    return calls


@pytest.fixture
def parsed_job(tmp_path: Path):
    q = JobQueue(tmp_path / "q.sqlite3")
    job, _ = q.enqueue("evt-d", "Cgroup", "夜市老闆娘發現一封信", user_id="U1", media_kind=JobKind.DRAMA)
    yield q, job
    q.close()


async def _with_screenplay(q: JobQueue, job: Any) -> Any:
    from fun_workflow.prompts.drama import write_screenplay

    screenplay, how = await write_screenplay(job.text, _good_client())
    q.set_parsed(job.id, screenplay_payload(screenplay, how))
    return q.by_token(job.token)


def _deadline(minutes: float = 120.0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


async def _render(job: Any, providers: dict[JobKind, Any], tmp_path: Path, **kw: Any) -> Path:
    return await drama.render_drama(
        job, providers, files_dir=tmp_path / "files", runs_dir=tmp_path / "runs",
        deadline=kw.pop("deadline", _deadline()), poll_interval_s=0.0, **kw,
    )


# ------------------------------------------------------------------ the shape


async def test_all_flux_then_all_h3_then_one_file(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    image, clip = FakeImageProvider(ledger), FakeClipProvider(ledger)
    touches: list[int] = []

    out = await _render(
        job, {MediaKind.IMAGE: image, MediaKind.VIDEO: clip}, tmp_path,
        on_activity=lambda: touches.append(1),
    )

    assert out == tmp_path / "files" / f"{job.token}.mp4" and out.is_file()
    image_events = [e for e in ledger.events if e.startswith("image:")]
    clip_events = [e for e in ledger.events if e.startswith("clip:")]
    assert len(image_events) == 2 + 6 and len(clip_events) == 6
    # Strict ordering: every still before the first clip.
    last_image = max(i for i, e in enumerate(ledger.events) if e.startswith("image:"))
    first_clip = min(i for i, e in enumerate(ledger.events) if e.startswith("clip:"))
    assert last_image < first_clip
    assert OUTLINE["anchor"]["appearance"] in image.requests[2].prompt  # keyframe 1
    assert image.requests[2].source_image_path.endswith("character/front.png")
    assert image.requests[2].width == 864 and image.requests[2].height == 480, "H3's canvas, not Flux's"
    assert all(r.first_frame_path.endswith(f"keyframes/shot_{i}.png") for i, r in enumerate(clip.requests, 1))
    assert clip.requests[0].mode is GenMode.I2V
    assert [r.duration_s for r in clip.requests] == pytest.approx([f / 24 for f in (158, 243, 192, 243, 243, 209)])
    assert "At 00:02.500, the camera cuts to" in clip.requests[0].prompt
    assert fake_ffmpeg[:6] == [f"level:shot_{i}.mp4" for i in range(1, 7)]
    assert fake_ffmpeg[6].startswith("assemble:shot_1.mp4,shot_2.mp4")
    # Five clip boundaries: the cut into shot 4 is time_passing -> dissolve.
    assert fake_ffmpeg[7] == "boundaries:hard_cut,hard_cut,dissolve,hard_cut,hard_cut"
    assert fake_ffmpeg[8] == f"total_s:{(1288 - 12) / 24:.3f}"
    assert fake_ffmpeg[9] == "subtitles:captions.ass"
    assert len(touches) == 2 + 6 + 6 + 1, "every artifact resets the reaper, plus the final file"

    run_dir = tmp_path / "runs" / "drama" / job.token
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    offsets = json.loads((run_dir / "offsets.json").read_text(encoding="utf-8"))
    assert len(plan["segments"]) == 10 == len(offsets["segments"])
    assert plan["transitions"][2] == {"after_clip": "3", "reason": "time_passing", "kind": "dissolve", "downgraded_from": None}
    assert [c["text"] for c in plan["cues"]] == ["沒事。"]
    assert offsets["clip_offsets"][1] == pytest.approx(158 / 24)
    gate = json.loads((run_dir / "gates" / "plan_gate.json").read_text(encoding="utf-8"))
    assert gate["gate"] == "plan_gate" and [f["rule_id"] for f in gate["findings"]] == ["R-BAND-WARN"]
    assert drama.load_state(run_dir).plan_gate == "passed with 1 warning(s): R-BAND-WARN"
    captions = (run_dir / "captions.ass").read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:00.00,0:00:01.50,Title,,0,0,0,,{\\fad(0,300)}夜市的信" in captions
    # Shot 3 is one 8.0 s segment starting at 158+243 frames, minus the margin.
    assert f"Dialogue: 0,0:00:{(158 + 243) / 24 + 0.1:05.2f},0:00:{(158 + 243 + 192) / 24 - 0.1:05.2f},Main,,0,0,0,,沒事。" in captions
    assert drama.load_state(run_dir).captions is not None

    state = drama.load_state(tmp_path / "runs" / "drama" / job.token)
    assert state.face_repair == "applied"
    assert state.spent_usd == pytest.approx(8 * 0.006 + 6 * 0.03)
    manifest = json.loads((tmp_path / "runs" / "drama" / job.token / "render_manifest.json").read_text())
    assert len(manifest["ffmpeg"]) == 6 * 2 + 1


async def test_make_room_for_evicts_the_other_side_not_comfyui(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    """Flux and H3 both live in ComfyUI; the drama's `make_room_for` calls must
    evict the inference-server side and never ComfyUI's own checkpoint."""
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()

    class Chat:
        residency_group = "inference"

        async def evict(self) -> None:
            ledger.events.append("evict:chat")

    providers = {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger), MediaKind.CHAT: Chat()}
    await _render(job, providers, tmp_path)

    assert ledger.events.count("evict:chat") == 2  # once before Flux, once before H3
    assert "evict:video" not in ledger.events
    # Flux *is* released once, after its last still and before the first clip:
    # it shares ComfyUI with H3, so make_room_for never touches it, and its
    # staged weights cost the fourth clip its VRAM on 2026-08-29.
    assert ledger.events.count("evict:image") == 1
    last_image = max(i for i, e in enumerate(ledger.events) if e.startswith("image:"))
    first_clip = min(i for i, e in enumerate(ledger.events) if e.startswith("clip:"))
    assert last_image < ledger.events.index("evict:image") < first_clip


# ---------------------------------------------------------------------- resume


async def test_a_failure_after_three_clips_resumes_with_the_other_three(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    providers = {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger, fail_on={f"job{job.id}_shot4"})}

    with pytest.raises(ProviderError, match="boom"):
        await _render(job, providers, tmp_path)
    first_run = list(ledger.events)
    assert sum(e.startswith("clip:") for e in first_run) == 4  # 1, 2, 3 done; 4 raised

    ledger.events.clear()
    providers[JobKind.VIDEO] = FakeClipProvider(ledger)
    out = await _render(job, providers, tmp_path)

    assert out.is_file()
    assert [e for e in ledger.events if e.startswith("image:")] == [], "no still is re-rendered"
    assert [e for e in ledger.events if e.startswith("clip:")] == [f"clip:job{job.id}_shot{i}" for i in (4, 5, 6)]


async def test_a_corrupted_artifact_is_re_rendered_alone(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    providers = {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger)}
    await _render(job, providers, tmp_path)

    run_dir = tmp_path / "runs" / "drama" / job.token
    (run_dir / "keyframes" / "shot_2.png").write_bytes(b"truncated")
    (tmp_path / "files" / f"{job.token}.mp4").unlink()
    ledger.events.clear()

    await _render(job, providers, tmp_path)

    assert [e for e in ledger.events if e.startswith("image:")] == [f"image:job{job.id}_kf2"]
    assert [e for e in ledger.events if e.startswith("clip:")] == []
    assert (tmp_path / "files" / f"{job.token}.mp4").is_file()


async def test_a_second_call_after_success_renders_nothing(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    providers = {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger)}
    await _render(job, providers, tmp_path)
    ledger.events.clear()
    fake_ffmpeg.clear()

    await _render(job, providers, tmp_path)

    assert ledger.events == [] and fake_ffmpeg == []


# ----------------------------------------------------------------------- gates


async def test_the_cost_gate_refuses_before_any_submit(parsed_job, tmp_path: Path, fake_ffmpeg, monkeypatch) -> None:
    from ai_studio.config import settings as settings_mod

    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    monkeypatch.setattr(settings_mod.get_settings(), "max_cost_usd", 0.01)

    with pytest.raises(CostCeilingExceeded, match="AI_STUDIO_MAX_COST_USD"):
        await _render(job, {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger)}, tmp_path)
    assert ledger.events == []


async def test_the_time_gate_requeues_rather_than_starting_a_clip_the_lease_would_cut(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    providers = {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger)}

    # Enough for stills (90 s reserve) but not for a clip (360 s reserve).
    with pytest.raises(ProviderError, match="lease ends"):
        await _render(job, providers, tmp_path, deadline=_deadline(minutes=3))

    assert sum(e.startswith("image:") for e in ledger.events) == 8
    assert not any(e.startswith("clip:") for e in ledger.events)
    state = drama.load_state(tmp_path / "runs" / "drama" / job.token)
    assert len(state.keyframes) == 6, "the stills are kept for the next window"


async def test_the_bell_mid_clip_cancels_and_requeues(parsed_job, tmp_path: Path, fake_ffmpeg, monkeypatch) -> None:
    monkeypatch.setitem(drama.STAGE_RESERVE_S, "video", 0.3)
    monkeypatch.setitem(drama.STAGE_RESERVE_S, "image", 0.0)
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()

    class NeverFinishes(FakeClipProvider):
        async def submit(self, request: Any) -> ClipJob:
            job_ = await super().submit(request)
            return job_.model_copy(update={"state": ClipState.RUNNING})

    # Stills go through; the clip loop then sees the (already past) deadline
    # on its first poll. The time gate is bypassed by a deadline far enough
    # away at the gate but crossed by the time the loop checks: simulate with
    # a provider that never finishes and a deadline the poll loop crosses.
    providers = {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: NeverFinishes(ledger)}
    deadline = datetime.now(timezone.utc) + timedelta(seconds=1.0)
    with pytest.raises(ProviderError, match="window closed"):
        await drama.render_drama(
            job, providers, files_dir=tmp_path / "files", runs_dir=tmp_path / "runs",
            deadline=deadline, poll_interval_s=0.2,
        )
    assert "cancel" in ledger.events


# --------------------------------------------------------------- face repair


async def test_face_repair_is_recorded_as_skipped_when_the_pod_lacks_the_nodes(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    await _render(job, {MediaKind.IMAGE: FakeImageProvider(ledger, face_nodes=False), MediaKind.VIDEO: FakeClipProvider(ledger)}, tmp_path)
    assert drama.load_state(tmp_path / "runs" / "drama" / job.token).face_repair.startswith("skipped")


async def test_a_failing_face_graph_falls_back_to_plain_i2i_once(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    image = FakeImageProvider(ledger, face_fails=True)
    await _render(job, {MediaKind.IMAGE: image, MediaKind.VIDEO: FakeClipProvider(ledger)}, tmp_path)

    keyframe_requests = [r for r in image.requests if "_kf" in r.shot_id]
    # Keyframe 1: face attempt + plain retry. The failure is then recorded,
    # so keyframes 2-6 go straight to plain i2i -- a requeued drama must not
    # pay the broken face graph again on every shot.
    assert len(keyframe_requests) == 2 + 5
    assert keyframe_requests[0].extra.get("face_repair") is True
    assert all(not r.extra.get("face_repair") for r in keyframe_requests[1:])
    assert drama.load_state(tmp_path / "runs" / "drama" / job.token).face_repair.startswith("failed")


async def test_face_repair_off_by_setting_never_asks_for_it(parsed_job, tmp_path: Path, fake_ffmpeg, monkeypatch) -> None:
    from fun_workflow.config import settings as fun_settings

    monkeypatch.setattr(fun_settings.get_fun_settings(), "drama_face_repair", False)
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    image = FakeImageProvider(ledger)
    await _render(job, {MediaKind.IMAGE: image, MediaKind.VIDEO: FakeClipProvider(ledger)}, tmp_path)
    assert all("face_repair" not in r.extra for r in image.requests)
    assert drama.load_state(tmp_path / "runs" / "drama" / job.token).face_repair.startswith("off")


# ------------------------------------------------------------------- refusals


async def test_a_job_without_a_screenplay_is_terminal(parsed_job, tmp_path: Path) -> None:
    q, job = parsed_job
    q.set_parsed(job.id, {"_rendered": "x", "_built_by": "raw"})
    job = q.by_token(job.token)
    with pytest.raises(AIStudioError, match="no screenplay"):
        await _render(job, {MediaKind.IMAGE: FakeImageProvider(Ledger()), MediaKind.VIDEO: FakeClipProvider(Ledger())}, tmp_path)


async def test_a_pod_without_both_providers_is_terminal(parsed_job, tmp_path: Path) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    with pytest.raises(AIStudioError, match="both the image and the video"):
        await _render(job, {MediaKind.VIDEO: FakeClipProvider(Ledger())}, tmp_path)


def test_a_corrupt_state_file_is_loud_not_a_reset(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(AIStudioError, match="not a readable drama state"):
        drama.load_state(tmp_path)


@pytest.mark.asyncio
async def test_every_artifact_and_stage_is_timestamped(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    """A 15-30 minute multi-window render left no record of *when* anything
    happened before 2026-08-28 -- only sha256 and cost. Now each artifact
    carries created_at, each stage its started/finished, and the manifest
    says when it was generated and for which request."""
    import re

    from fun_workflow.pipeline import drama as mod

    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    await _render(job, {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger)}, tmp_path)

    run_dir = tmp_path / "runs" / "drama" / job.token
    state = mod.load_state(run_dir)
    iso = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00")
    for bucket in (state.character, state.keyframes, state.clips, state.leveled):
        assert bucket and all(iso.fullmatch(r.created_at) for r in bucket.values())
    assert state.output is not None and iso.fullmatch(state.output.created_at)
    assert iso.fullmatch(state.created_at) and iso.fullmatch(state.updated_at)
    assert set(state.stages) == {"character", "keyframes", "clips", "level", "assemble"}
    for name, timing in state.stages.items():
        assert iso.fullmatch(timing.started_at) and iso.fullmatch(timing.finished_at), name
        assert timing.started_at <= timing.finished_at, name

    manifest = json.loads((run_dir / "render_manifest.json").read_text(encoding="utf-8"))
    assert iso.fullmatch(manifest["generated_at"])
    assert manifest["token"] == job.token and manifest["job_id"] == job.id
    assert manifest["stages"]["assemble"]["finished_at"]
    assert manifest["timeline"] == {"total_s": pytest.approx((1288 - 12) / 24), "segments": 10, "dissolves": 1}
    assert len(manifest["ffmpeg"]) == 6 * 2 + 1


def test_a_state_file_from_before_timestamps_still_loads(tmp_path: Path) -> None:
    from fun_workflow.pipeline import drama as mod

    (tmp_path / "state.json").write_text(
        '{"character": {"front": {"path": "x.png", "sha256": "ab"}}, "spent_usd": 0.1}',
        encoding="utf-8",
    )
    state = mod.load_state(tmp_path)
    assert state.character["front"].created_at == "" and state.stages == {}


async def test_a_wide_opening_gets_the_wide_denoise(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    from fun_workflow.config import settings as fun_settings

    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    image = FakeImageProvider(ledger)
    await _render(job, {MediaKind.IMAGE: image, MediaKind.VIDEO: FakeClipProvider(ledger)}, tmp_path)
    fun = fun_settings.get_fun_settings()
    by_shot = {r.shot_id: r.extra["denoise"] for r in image.requests if "_kf" in r.shot_id}
    assert by_shot[f"job{job.id}_kf1"] == fun.drama_keyframe_denoise  # opens close-up
    assert by_shot[f"job{job.id}_kf2"] == fun.drama_keyframe_denoise_wide  # opens wide


async def test_subshots_off_makes_six_segments_and_one_prompt_shot(parsed_job, tmp_path: Path, fake_ffmpeg, monkeypatch) -> None:
    from fun_workflow.config import settings as fun_settings

    monkeypatch.setattr(fun_settings.get_fun_settings(), "drama_subshots", False)
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    clip = FakeClipProvider(ledger)
    await _render(job, {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: clip}, tmp_path)
    assert "[Shot 2]" not in clip.requests[0].prompt
    offsets = json.loads((tmp_path / "runs" / "drama" / job.token / "offsets.json").read_text(encoding="utf-8"))
    assert len(offsets["segments"]) == 6


async def test_a_provider_off_the_template_frame_rate_is_terminal(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    q, job = parsed_job
    job = await _with_screenplay(q, job)

    class Thirty(FakeClipProvider):
        def capabilities(self) -> ProviderCapabilities:
            return CLIP_CAPS.model_copy(update={"native_fps": 30})

    with pytest.raises(AIStudioError, match="24 fps grid"):
        await _render(job, {MediaKind.IMAGE: FakeImageProvider(Ledger()), MediaKind.VIDEO: Thirty(Ledger())}, tmp_path)


def test_a_state_file_with_the_old_concat_stage_still_loads(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(
        '{"stages": {"concat": {"started_at": "2026-08-28T00:00:00.000+00:00"}}, "spent_usd": 0.3}',
        encoding="utf-8",
    )
    state = drama.load_state(tmp_path)
    assert state.stages["concat"].started_at and state.plan_gate == "pending"


async def test_a_plan_that_fails_the_gate_spends_nothing(parsed_job, tmp_path: Path, fake_ffmpeg, monkeypatch) -> None:
    """The PRE gate runs before the cost gate and before any submit; a
    metronome floor no template can meet proves the order."""
    from ai_studio.core.errors import GateFailure
    from ai_studio.editing.rhythm import PacingPolicy

    from fun_workflow.core import drama_spec

    monkeypatch.setattr(drama_spec, "DRAMA_PACING", PacingPolicy(min_s=2.0, warn_s=8.0, fail_s=12.5, cv_floor=0.9))
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()
    with pytest.raises(GateFailure, match="R-CV"):
        await _render(job, {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger)}, tmp_path)
    assert ledger.events == [] and fake_ffmpeg == []
    run_dir = tmp_path / "runs" / "drama" / job.token
    assert (run_dir / "gates" / "plan_gate.json").is_file()
    assert drama.load_state(run_dir).plan_gate.startswith("failed: ")


async def test_an_oom_releases_comfyui_before_the_attempt_is_handed_back(parsed_job, tmp_path: Path, fake_ffmpeg) -> None:
    """After an OOM ComfyUI's own unload left 16 MiB free and the requeued
    attempt died in a second, twice (2026-08-29). The provider is evicted
    before the error propagates; any other failure is not."""
    q, job = parsed_job
    job = await _with_screenplay(q, job)
    ledger = Ledger()

    class Oom(FakeClipProvider):
        async def submit(self, request: Any) -> ClipJob:
            job_ = await super().submit(request)
            if request.shot_id.endswith("_shot2"):
                return job_.model_copy(update={"state": ClipState.FAILED,
                                               "error": "Allocation on device 0 would exceed allowed memory. (out of memory)"})
            return job_

    clip = Oom(ledger)
    with pytest.raises(ProviderError, match="out of memory"):
        await _render(job, {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: clip}, tmp_path)
    assert ledger.events.count("evict:video") == 1
    assert ledger.events.index("evict:video") > ledger.events.index(f"clip:job{job.id}_shot2")

    ledger.events.clear()
    with pytest.raises(ProviderError, match="boom"):
        await _render(job, {MediaKind.IMAGE: FakeImageProvider(ledger), MediaKind.VIDEO: FakeClipProvider(ledger, fail_on={f"job{job.id}_shot2"})}, tmp_path)
    assert "evict:video" not in ledger.events
