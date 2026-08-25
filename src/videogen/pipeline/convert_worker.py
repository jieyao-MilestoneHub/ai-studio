"""Wire the LLM to the queue.

Lives in `pipeline` rather than `prompts` because it touches both the queue (L4)
and a client (L2); `prompts.convert` stays pure so it can be tested with no
network and no database.
"""

from __future__ import annotations

from videogen.core.enums import MediaKind
from videogen.pipeline.queue import JobQueue, JobState
from videogen.prompts import flux as flux_prompts
from videogen.prompts.convert import LlmClient, convert
from videogen.prompts.flux import FluxPrompt
from videogen.prompts.h3 import H3Prompt

DEFAULT_DURATION_S = 124 / 24
"""124 frames at 24fps. The turbo node pack documents that frame counts snap to
a 17k+5 grid and that 124 is the validated floor, so this is the shortest clip
the model will actually produce rather than a round number."""


async def convert_job(
    queue: JobQueue,
    job_id: int,
    client: LlmClient | None = None,
    *,
    duration_s: float = DEFAULT_DURATION_S,
) -> str:
    """Convert one queued request into a validated prompt and mark it claimable.

    Returns how the prompt was built ("llm", "llm-retry", "template..."), which
    is worth keeping: it is the difference between a 367.6-quality prompt and a
    26.0 one, and it explains a disappointing clip after the fact.
    """
    job = next((j for j in queue.unparsed(limit=200) if j.id == job_id), None)
    if job is None:
        return "skipped: not queued"

    prompt: FluxPrompt | H3Prompt
    if job.media_kind is MediaKind.IMAGE:
        prompt, how = await flux_prompts.convert(job.text, client)
    else:
        prompt, how = await convert(job.text, client, duration_s=duration_s)
    payload = prompt.model_dump(mode="json")
    payload["_rendered"] = prompt.render()
    payload["_built_by"] = how
    queue.set_parsed(job.id, payload)
    return how


async def convert_pending(
    queue: JobQueue, client: LlmClient | None = None, *, limit: int = 20
) -> dict[str, int]:
    """Convert everything still waiting. Used at window open as a catch-up.

    A request whose conversion failed while the web process was restarting would
    otherwise sit `queued` forever and never reach a GPU.
    """
    tally: dict[str, int] = {}
    for job in queue.unparsed(limit=limit):
        if job.state is not JobState.QUEUED:
            continue
        how = await convert_job(queue, job.id, client)
        key = how.split(" ")[0]
        tally[key] = tally.get(key, 0) + 1
    return tally
