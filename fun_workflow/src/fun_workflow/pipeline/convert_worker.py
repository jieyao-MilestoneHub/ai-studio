"""Wire the LLM to the queue.

Lives in `pipeline` rather than `prompts` because it touches both the queue (L4)
and a client (L2); `prompts.convert` stays pure so it can be tested with no
network and no database.
"""

from __future__ import annotations

from typing import Any

from ai_studio.config.settings import get_settings
from ai_studio.core.enums import MediaKind
from ai_studio.core.errors import AIStudioError
from ai_studio.prompts import flux as flux_prompts
from ai_studio.prompts import understanding as understanding_prompts
from ai_studio.prompts.convert import LlmClient, convert
from ai_studio.prompts.flux import FluxPrompt
from ai_studio.prompts.h3 import I2VA_INSTRUCTION, H3Mode, H3Prompt

from fun_workflow.pipeline.queue import Job, JobQueue, JobState
from fun_workflow.prompts.chat import CHAT_DEVELOPER_PROMPT
from fun_workflow.prompts.drama import ScreenplayError, screenplay_payload, write_screenplay

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

MAX_FRAMES = 362
"""17*21+5 = 15.08 s at 24fps. Measured on an RTX 4090 (2026-08-26): 362
frames rendered in 📏 317 s with a 📏 22.8 GB peak of 24 -- stable, no OOM,
the subject consistent first frame to last -- but four times the render time
of the 243-frame default, so it is the ceiling a user may ask for, not the
default. Above it VRAM headroom is unmeasured and the model's own limit is
near, so requests are clamped here."""

DEFAULT_DURATION_S = DEFAULT_FRAMES / 24
MAX_DURATION_S = MAX_FRAMES / 24


def clamp_duration(seconds: float | None) -> float:
    """A requested length, clamped to what the model and the card allow and
    snapped to the frame grid. None -> the default. Below the floor or above
    the ceiling is pulled into range rather than refused: a user who types a
    number just wants the nearest clip that exists."""
    if seconds is None:
        return DEFAULT_DURATION_S
    frames = snap_frames(round(seconds * 24))
    return min(max(frames, MIN_FRAMES), MAX_FRAMES) / 24


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
    if job.media_kind is MediaKind.DRAMA:
        raise AIStudioError("a drama has no raw form; it needs the screenwriter")
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

    if job.media_kind is MediaKind.CHAT:
        # The user's words are never rewritten (a chat is a chat), but the
        # reply gets the engineered developer prompt -- persona, language,
        # length -- which `drain.render_chat` carries as `extra["system"]`.
        queue.set_parsed(job.id, {"_built_by": "chat", "_system": CHAT_DEVELOPER_PROMPT})
        return "chat"

    if job.media_kind is MediaKind.DRAMA:
        # The screenwriter, in any prompt mode: a drama has nothing to be
        # "raw" from, and there is no template that is six shots of a story.
        # A failure here is terminal and *told* -- `worker.prepare` delivers
        # the failure -- rather than a job left queued forever.
        try:
            screenplay, how = await write_screenplay(job.text, client)
        except ScreenplayError as exc:
            queue.fail(job.id, f"編劇失敗:{exc}")
            return f"failed: {exc}"
        queue.set_parsed(job.id, screenplay_payload(screenplay, how))
        return how

    if job.media_kind.is_understanding:
        # The webhook already validated the media; what is built here is the
        # *question*: the engineered default when the trigger came bare, or
        # the user's own question rewritten into the model's best form
        # (structured mode) / sent as typed (raw mode). None means the image
        # caption path on the server -- see prompts/understanding.py.
        question, how = await understanding_prompts.convert_question(
            job.text, client if mode_setting == "structured" else None, modality=job.media_kind
        )
        queue.set_parsed(job.id, {"_built_by": how, "_question": question})
        return how

    length = clamp_duration(job.requested_seconds) if job.requested_seconds else duration_s
    if mode_setting == "raw":
        queue.set_parsed(job.id, raw_payload(job, duration_s=length))
        return "raw"

    prompt: FluxPrompt | H3Prompt
    if job.media_kind is MediaKind.IMAGE:
        prompt, how = await flux_prompts.convert(job.text, client)
    else:
        # A cached photo makes this image-to-video: the picture is the first
        # frame, and the prompt has to be about it rather than a fresh scene.
        mode = H3Mode.I2VA if job.first_frame_path else H3Mode.T2VA
        prompt, how = await convert(job.text, client, duration_s=length, mode=mode)
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


def needs_llm(job: Job, prompt_mode: str) -> bool:
    """Whether converting this job would call the rewriter at all.

    The worker uses this to split a batch: jobs that do not need the LLM are
    converted at once, the rest wait until gpt-oss is loaded -- one model
    swap for the whole batch, not one per job. Chat never needs it (the words
    go verbatim); an understanding job needs it only when the user added a
    question; generation needs it only in structured mode.
    """
    if job.media_kind is MediaKind.CHAT:
        return False
    if job.media_kind is MediaKind.DRAMA:
        return True  # the screenwriter is the feature, whatever the mode
    if prompt_mode != "structured":
        return False
    if job.media_kind.is_understanding:
        return bool(job.text.strip())
    return True
