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

from pathlib import Path
from typing import Protocol, runtime_checkable

from videogen.core.provider_spec import ClipAsset, ClipJob, ClipRequest, ProviderCapabilities


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
