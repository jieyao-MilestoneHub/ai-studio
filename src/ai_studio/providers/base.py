"""The clip-provider protocol.

Four methods rather than one blocking `generate()`, for three reasons that are
all about money and none about taste:

1. **The backend contract is already submit-then-poll.** An H3 clip runs for
   2-6 minutes; RunPod's pod proxy is cut at ~100s. Modelling generation as one
   awaitable call means every implementation reinvents the same poll loop
   around the same constraint.

2. **`ClipJob` is serialisable, so a crash is survivable.** Jobs are written to
   `clips.json` on every state change. If the orchestrator dies mid-run, resume
   reattaches to jobs still executing on the GPU instead of paying to generate
   them a second time.

3. **`fetch` has to be separable.** Outputs live on the pod's container disk,
   which is destroyed by `pod down`; a RunPod public endpoint's URL expires
   after seven days. Making the copy-into-our-own-storage step its own method
   is what stops it being the step everyone forgets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ai_studio.core.image_provider_spec import (
    ImageAsset,
    ImageJob,
    ImageProviderCapabilities,
    ImageRequest,
)
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ClipRequest, ProviderCapabilities


@runtime_checkable
class ClipProvider(Protocol):
    """A backend that turns a `ClipRequest` into a video file."""

    name: str

    def capabilities(self) -> ProviderCapabilities:
        """Declare native size, clip-length limits, cost, and latency.

        Published into `provider_manifest.json` so the format policy and planner
        can reason about the model without importing this package.
        """
        ...

    async def submit(self, request: ClipRequest) -> ClipJob:
        """Enqueue one clip. Returns immediately with a job handle."""
        ...

    async def poll(self, job: ClipJob) -> ClipJob:
        """Refresh a job's state. Never blocks longer than one HTTP call."""
        ...

    async def fetch(self, job: ClipJob, dest: Path) -> ClipAsset:
        """Copy a completed job's output to `dest` and describe it."""
        ...

    async def cancel(self, job: ClipJob) -> None:
        """Best-effort cancellation. Must not raise if the job already ended."""
        ...

    async def aclose(self) -> None:
        """Release any connections."""
        ...


@runtime_checkable
class ImageProvider(Protocol):
    """A backend that turns an `ImageRequest` into a still image.

    Same shape as `ClipProvider` — submit/poll/fetch/cancel over the same
    ComfyUI HTTP surface — but for a still image, which has no frame count.
    """

    name: str

    def capabilities(self) -> ImageProviderCapabilities: ...

    async def submit(self, request: ImageRequest) -> ImageJob: ...

    async def poll(self, job: ImageJob) -> ImageJob: ...

    async def fetch(self, job: ImageJob, dest: Path) -> ImageAsset: ...

    async def cancel(self, job: ImageJob) -> None: ...

    async def aclose(self) -> None: ...


def write_provider_manifest(
    dest_dir: Path | str,
    capabilities: ProviderCapabilities | ImageProviderCapabilities,
    *,
    profile_key: str,
    workflow: Path | str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Publish a capabilities snapshot to `provider_manifest.json`.

    This file has been named in `core/provider_spec.py`, in this module's own
    docstring above, in `docs/architecture.md` and in `CLAUDE.md` ("Do not move
    it") since the beginning — and nothing has ever written it. The dependency
    inversion was real at the type level (`ProviderCapabilities` genuinely lives
    in `core`, and `editing` genuinely does not import `providers`); only the
    serialisation half was missing.

    It matters now because gates are, by contract, *pure functions of on-disk
    artifacts* and may not import `providers`. Without this file a gate has no
    way to learn what the model was configured to produce, which is exactly what
    checking a render against its own promises requires.

    Written once per pod rather than per job: it describes the provider, and the
    provider does not change while the pod lives. The workflow digest is what
    makes that claim checkable — two runs with the same manifest really did
    submit the same graph.
    """
    workflow_path = Path(workflow)
    digest = ""
    if workflow_path.is_file():
        digest = hashlib.sha256(workflow_path.read_bytes()).hexdigest()[:16]

    payload: dict[str, Any] = {
        "profile": profile_key,
        "workflow": workflow_path.name,
        "workflow_sha256_16": digest,
        "capabilities": capabilities.model_dump(mode="json"),
    }
    if extra:
        payload.update(extra)

    dest = Path(dest_dir) / "provider_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return dest
