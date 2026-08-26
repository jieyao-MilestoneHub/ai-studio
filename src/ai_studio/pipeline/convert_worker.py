"""Wire the LLM to the queue.

Lives in `pipeline` rather than `prompts` because it touches both the queue (L4)
and a client (L2); `prompts.convert` stays pure so it can be tested with no
network and no database.
"""

from __future__ import annotations

from typing import Any

from ai_studio.config.settings import get_settings
from ai_studio.core.enums import MediaKind
from ai_studio.pipeline.queue import Job, JobQueue, JobState
from ai_studio.prompts import flux as flux_prompts
from ai_studio.prompts.convert import LlmClient, convert
from ai_studio.prompts.flux import FluxPrompt
from ai_studio.prompts.h3 import I2VA_INSTRUCTION, H3Mode, H3Prompt

FRAME_GRID = 17
"""The turbo node pack documents that frame counts snap to a 17k+5 grid."""

MIN_FRAMES = 124
"""17*7+5: the validated floor, the shortest clip the model actually produces."""

DEFAULT_FRAMES = 243
"""17*14+5 = 10.1 s at 24fps, the default clip length.

Measured on an RTX 4090 (864x480, int8, 2026-08-26): 243 frames rendered in
📏 79 s with a 📏 22.1 GB VRAM peak -- the same time and the same peak as
124 frames, because the turbo path's cost is dominated by the text encoder
and VAE rather than by frame count -- and the subject stayed consistent from
the first frame to the last. 294 frames is the next grid step; it was not
adopted because the community guide puts drift risk above ~10 s and nothing
here has measured it yet."""

DEFAULT_DURATION_S = DEFAULT_FRAMES / 24


def snap_frames(frames: int) -> int:
    """The nearest frame count the model accepts, at or above `frames`.

    17k+5, never below MIN_FRAMES. A count off the grid is not an error the
    node pack raises; it is a clip silently cut to the grid step below.
    """
    frames = max(frames, MIN_FRAMES)
    k = -(-(frames - 5) // FRAME_GRID)  # ceil
    return FRAME_GRID * k + 5


def raw_payload(job: Job, *, duration_s: float = DEFAULT_DURATION_S) -> dict[str, Any]:
    """The user's own words, verbatim, as the prompt. No LLM, no schema.

    What the group asked for: the person typing knows what they want, and
    every rewrite -- however faithful -- is a place their words can be lost.
    The one thing added is model protocol, not prompt-writing: an
    image-to-video request still needs the line that binds the picture to
    the first frame. `duration_s` and `mode` are here because the render
    reads them from the plan, the same as for a structured one.
    """
    text = job.text.strip()
    if job.media_kind is MediaKind.IMAGE:
        return {"text": text, "_rendered": text, "_built_by": "raw"}
    mode = H3Mode.I2VA if job.first_frame_path else H3Mode.T2VA
    rendered = f"{I2VA_INSTRUCTION}\n\n{text}" if mode is H3Mode.I2VA else text
    return {
        "text": text,
        "mode": mode.value,
        "duration_s": duration_s,
        "_rendered": rendered,
        "_built_by": "raw",
    }


async def convert_job(
    queue: JobQueue,
    job_id: int,
    client: LlmClient | None = None,
    *,
    duration_s: float = DEFAULT_DURATION_S,
    prompt_mode: str | None = None,
) -> str:
    """Convert one queued request into a validated prompt and mark it claimable.

    Returns how the prompt was built ("llm", "llm-retry", "template..."), which
    is worth keeping: it is the difference between a 367.6-quality prompt and a
    26.0 one, and it explains a disappointing clip after the fact.
    """
    job = next((j for j in queue.unparsed(limit=200) if j.id == job_id), None)
    if job is None:
        return "skipped: not queued"

    mode_setting = prompt_mode or get_settings().prompt_mode
    if mode_setting == "raw":
        queue.set_parsed(job.id, raw_payload(job, duration_s=duration_s))
        return "raw"

    prompt: FluxPrompt | H3Prompt
    if job.media_kind is MediaKind.IMAGE:
        prompt, how = await flux_prompts.convert(job.text, client)
    else:
        # A cached photo makes this image-to-video: the picture is the first
        # frame, and the prompt has to be about it rather than a fresh scene.
        mode = H3Mode.I2VA if job.first_frame_path else H3Mode.T2VA
        prompt, how = await convert(job.text, client, duration_s=duration_s, mode=mode)
    payload = prompt.model_dump(mode="json")
    payload["_rendered"] = prompt.render()
    payload["_built_by"] = how
    queue.set_parsed(job.id, payload)
    return how


async def convert_pending(
    queue: JobQueue,
    client: LlmClient | None = None,
    *,
    limit: int = 20,
    prompt_mode: str | None = None,
) -> dict[str, int]:
    """Convert everything still waiting. Used at window open as a catch-up.

    A request whose conversion failed while the web process was restarting would
    otherwise sit `queued` forever and never reach a GPU.
    """
    tally: dict[str, int] = {}
    for job in queue.unparsed(limit=limit):
        if job.state is not JobState.QUEUED:
            continue
        how = await convert_job(queue, job.id, client, prompt_mode=prompt_mode)
        key = how.split(" ")[0]
        tally[key] = tally.get(key, 0) + 1
    return tally
