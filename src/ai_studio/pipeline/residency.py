"""Who holds the card: the pull-based GPU hand-off between ComfyUI and the
inference server.

One 24GB card holds at most one of {ComfyUI's H3/Flux checkpoint, an
inference-server model} at a time. Every provider declares which side it
lives on (`residency_group`); before a job runs, `make_room_for` evicts
every provider on the *other* side. Neither side needs to know the other's
endpoint, and this module needs to know nothing about what the jobs are.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from ai_studio.core.errors import ProviderError

_log = logging.getLogger("ai_studio.residency")


def residency_group_of(provider: Any) -> str:
    """The side of the card a provider lives on. Raises rather than guess:
    a provider that does not say is one whose eviction nobody planned."""
    group = getattr(provider, "residency_group", None)
    if not isinstance(group, str) or not group:
        raise ProviderError(
            f"{type(provider).__name__} declares no residency_group; every provider "
            "must say which side of the GPU it lives on"
        )
    return group


async def make_room_for(target: Any, providers: Mapping[Any, Any]) -> None:
    """Evict every provider that is not on `target`'s side of the card.

    Pull-based: this is called by whoever is about to submit to `target`,
    the one place that knows which job is about to run. A provider that
    appears under several keys is evicted once; one whose `evict()` is a
    no-op (the offline stubs) costs nothing extra to call.
    """
    group = residency_group_of(target)
    evicted: set[int] = set()
    for key, provider in providers.items():
        if id(provider) in evicted or residency_group_of(provider) == group:
            continue
        evicted.add(id(provider))
        evict = getattr(provider, "evict", None)
        if evict is None:
            continue
        started = time.monotonic()
        await evict()
        # The model swap is the single most expensive thing that is not a
        # render (📏 15-90 s per side); it was never logged before 2026-08-28.
        label = str(getattr(key, "value", key))
        _log.info(
            "evicted %s for %s", label, group,
            extra={"stage": "swap", "evicted": label, "seconds": round(time.monotonic() - started, 1)},
        )
