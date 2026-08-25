"""Wire the LLM to the queue.

Lives in `pipeline` rather than `prompts` because it touches both the queue (L4)
and a client (L2); `prompts.convert` stays pure so it can be tested with no
network and no database.
"""

from __future__ import annotations

from ai_studio.core.enums import MediaKind
from ai_studio.core.model_profile import MINIMAX_H3
from ai_studio.pipeline.queue import JobQueue, JobState
from ai_studio.prompts import flux as flux_prompts
from ai_studio.prompts.convert import LlmClient, convert
from ai_studio.prompts.flux import FluxPrompt
from ai_studio.prompts.h3 import H3Prompt

DEFAULT_DURATION_S = MINIMAX_H3.shortest_useful_duration_s
"""The shortest clip worth generating, from the model profile.

124 frames at 24fps. The `17k+5` grid is `[verified]` against ComfyUI's own
`nodes_minimax_h3.py`; **124 is not the model's floor** — ComfyUI accepts 5 —
it is the turbo LoRA's trained lower bound `[reported]`, which is why the
profile records the two separately."""


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
