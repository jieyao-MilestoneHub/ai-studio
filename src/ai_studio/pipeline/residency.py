"""Who holds the card: the pull-based GPU hand-off between ComfyUI and the
inference server.

Its own module because two same-layer callers need it -- `drain` (per claimed
job) and `drama` (once per checkpoint side inside one job) -- and `drain`
also dispatches to `drama`. Living in either would make the other import it
lazily to dodge a cycle; living here, both import it at module level.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ai_studio.core.enums import MediaKind

_log = logging.getLogger("ai_studio.residency")


async def make_room_for(job_kind: MediaKind, providers: dict[MediaKind, Any]) -> None:
    """Evict whichever side's resident model the upcoming job does not need.

    Pull-based GPU hand-off: this is the one place that already knows which
    job is about to run, so it is the one place responsible for evicting the
    other workload's model before submitting -- one 24GB card holds at most
    one of {ComfyUI's H3/Flux checkpoint, an understanding model} at a time.
    Neither provider needs to know the other's endpoint.

    A provider whose kind is not in `providers` (this pod does not serve it)
    is skipped; a provider whose `evict()` is a no-op (the offline stubs)
    costs nothing extra to call.
    """
    other_kinds = (
        (MediaKind.IMAGE_UNDERSTAND, MediaKind.AUDIO_UNDERSTAND, MediaKind.VIDEO_UNDERSTAND, MediaKind.CHAT)
        if job_kind.is_generation
        else (MediaKind.VIDEO, MediaKind.IMAGE)
    )
    evicted: set[int] = set()
    for kind in other_kinds:
        provider = providers.get(kind)
        if provider is None or id(provider) in evicted:
            continue
        evicted.add(id(provider))
        evict = getattr(provider, "evict", None)
        if evict is not None:
            started = time.monotonic()
            await evict()
            # The model swap is the single most expensive thing that is not a
            # render (📏 15-90 s per side); it was never logged before 2026-08-28.
            _log.info(
                "evicted %s for %s", kind.value, job_kind.value,
                extra={"stage": "swap", "evicted": kind.value, "seconds": round(time.monotonic() - started, 1)},
            )

