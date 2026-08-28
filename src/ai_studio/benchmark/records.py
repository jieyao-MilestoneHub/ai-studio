"""The one record shape every real render logs, and the report reads.

Owned here so the producer (whoever drives a provider and fetches the
result) and the consumer (`benchmark.report`) cannot drift apart: a caller
builds the `extra=` dict with `render_record` and logs it under `msg_for`.
Anything else on a `stage="render"` line is ignored by the fold.
"""

from __future__ import annotations

from typing import Any

RENDER_STAGE = "render"

BENCHMARK_MSGS = {"fetched clip", "fetched image"}
"""The two log messages the monthly fold collects. Real generation jobs
only -- a stub or dry run never logs these."""

BENCHMARK_FIELDS = ("seconds", "cost_usd", "vram_gb")
"""Numeric fields averaged per `(kind, gpu_tier)` group. `frames_per_s` is
derived separately -- it needs two fields (`frames` / `seconds`) together."""


def msg_for(kind: str) -> str:
    """The log message for a finished render of `kind` ("video"/"image")."""
    if kind == "video":
        return "fetched clip"
    if kind == "image":
        return "fetched image"
    raise ValueError(f"no benchmark record for kind {kind!r}; only video and image renders are measured")


def render_record(
    kind: str,
    *,
    seconds: float,
    cost_usd: float | None,
    vram_gb: float | None,
    gpu_tier: str | None,
    polls: int = 0,
    frames: int | None = None,
) -> dict[str, Any]:
    """The `extra=` dict for one finished render. Only these keys are on the
    observability allow-list (`core.observability.EXTRA_FIELDS`)."""
    record: dict[str, Any] = {
        "stage": RENDER_STAGE,
        "kind": kind,
        "seconds": round(seconds, 1),
        "polls": polls,
        "cost_usd": cost_usd,
        "vram_gb": vram_gb,
        "gpu_tier": gpu_tier,
    }
    if frames is not None:
        record["frames"] = frames
    return record
