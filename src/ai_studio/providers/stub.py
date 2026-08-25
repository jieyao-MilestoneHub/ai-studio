"""Offline synthetic clip provider.

Exists so the entire pipeline — planning, format policy, assembly, gates — can
be developed and tested with no GPU, no RunPod account, and no money. It is the
provider CI runs against.

**It deliberately produces real motion.** A black frame would be simpler, and
would also make the pace gate pass vacuously: that gate counts visual events by
measuring frame-to-frame difference, so a static stub would report "no stillness
violations" on footage that is nothing but stillness, and hide the bug it exists
to catch. `testsrc2` moves, carries a frame counter, and hue-shifts per seed so
different shots are visually distinct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_studio import media
from ai_studio.config.settings import get_settings
from ai_studio.core.enums import GenMode, JobState
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ClipRequest, ProviderCapabilities
from ai_studio.storage.base import sha256_file

# Mirrors MiniMax H3 so that format policy, planner and gates see the same
# constraints offline as they will on real hardware.
STUB_CAPABILITIES = ProviderCapabilities(
    provider="stub",
    model_id="stub-testsrc2",
    native_width=864,
    native_height=480,
    native_fps=24,
    modes=frozenset({GenMode.T2V, GenMode.I2V, GenMode.KEYFRAME}),
    min_clip_s=1.0,
    max_clip_s=30.0,
    clip_duration_quantum=None,
    has_native_audio=True,
    cost_per_second_usd=0.0,
    expected_latency_s=2.0,
    max_concurrent_jobs=4,
    url_ttl_s=None,
)


class StubProvider:
    """Synthesises clips locally with ffmpeg. No network, no cost."""

    name = "stub"

    def __init__(self, work_dir: Path | str | None = None, **_: Any) -> None:
        settings = get_settings()
        self.work_dir = Path(work_dir) if work_dir else settings.runs_dir / "_stub"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._outputs: dict[str, Path] = {}
        self._requests: dict[str, ClipRequest] = {}

    def capabilities(self) -> ProviderCapabilities:
        return STUB_CAPABILITIES

    # ---------------------------------------------------------------- submit

    async def submit(self, request: ClipRequest) -> ClipJob:
        """Renders synchronously, then reports a completed job.

        The protocol is still honoured — callers poll and fetch exactly as they
        would against a real backend — so switching providers changes nothing
        about the calling code.
        """
        import time

        job_id = f"stub-{request.shot_id}"
        path = self.work_dir / f"{job_id}.mp4"
        now = time.time()

        try:
            self._render(request, path)
        except media.FFmpegError as exc:
            return ClipJob(
                provider=self.name,
                job_id=job_id,
                shot_id=request.shot_id,
                state=JobState.FAILED,
                submitted_at=now,
                updated_at=time.time(),
                error=str(exc),
            )

        self._outputs[job_id] = path
        self._requests[job_id] = request
        return ClipJob(
            provider=self.name,
            job_id=job_id,
            shot_id=request.shot_id,
            state=JobState.COMPLETED,
            submitted_at=now,
            updated_at=time.time(),
        )

    async def poll(self, job: ClipJob) -> ClipJob:
        return job

    async def fetch(self, job: ClipJob, dest: Path) -> ClipAsset:
        import shutil

        source = self._outputs.get(job.job_id)
        if source is None or not source.is_file():
            raise media.FFmpegError(f"no stub output for job {job.job_id}")

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)

        info = media.probe(dest)
        return ClipAsset(
            shot_id=job.shot_id,
            key=dest.name,
            sha256=sha256_file(dest),
            size_bytes=info.size_bytes,
            width=info.width,
            height=info.height,
            fps=info.fps,
            duration_s=info.duration_s,
            has_audio=info.has_audio,
            provider=self.name,
            job_id=job.job_id,
            cost_usd=0.0,
        )

    async def cancel(self, job: ClipJob) -> None:
        return None

    async def aclose(self) -> None:
        return None

    # --------------------------------------------------------------- private

    def _render(self, request: ClipRequest, dest: Path) -> None:
        settings = get_settings()
        seed = request.seed if request.seed is not None else abs(hash(request.shot_id))
        hue = seed % 360
        tone = 200 + (seed % 12) * 40  # a distinct pitch per shot

        argv = [
            settings.ffmpeg_bin,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi",
            "-i", f"testsrc2=size={request.width}x{request.height}"
                  f":rate={request.fps}:duration={request.duration_s}",
            "-f", "lavfi",
            "-i", f"sine=frequency={tone}:duration={request.duration_s}",
            "-vf", f"hue=h={hue}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            str(dest),
        ]
        media.run(argv, timeout_s=180.0)
