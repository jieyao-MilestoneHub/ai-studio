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

from ai_studio.core.chat_spec import ChatAsset, ChatCapabilities, ChatJob, ChatRequest
from ai_studio.core.image_provider_spec import (
    ImageAsset,
    ImageJob,
    ImageProviderCapabilities,
    ImageRequest,
)
from ai_studio.core.provider_spec import ClipAsset, ClipJob, ClipRequest, ProviderCapabilities
from ai_studio.core.understanding_spec import (
    UnderstandingAsset,
    UnderstandingCapabilities,
    UnderstandingJob,
    UnderstandingRequest,
)


@runtime_checkable
class ClipProvider(Protocol):
    """A backend that turns a `ClipRequest` into a video file."""

    name: str
    residency_group: str
    """Which side of the one GPU this provider\'s model lives on ("comfyui" or
    "inference"); `pipeline.residency.make_room_for` evicts the other side."""

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
    residency_group: str
    """Which side of the one GPU this provider\'s model lives on ("comfyui" or
    "inference"); `pipeline.residency.make_room_for` evicts the other side."""

    def capabilities(self) -> ImageProviderCapabilities:
        """Declare what this backend can do, its cost and its latency, without
        touching the network."""
        ...

    async def submit(self, request: ImageRequest) -> ImageJob:
        """Enqueue one request. Returns immediately with a job handle."""
        ...

    async def poll(self, job: ImageJob) -> ImageJob:
        """Refresh a job's state. Never blocks longer than one HTTP call."""
        ...

    async def fetch(self, job: ImageJob, dest: Path) -> ImageAsset:
        """Copy a completed job's image to `dest` and describe it."""
        ...

    async def cancel(self, job: ImageJob) -> None:
        """Best-effort cancellation. Must not raise if the job already ended."""
        ...

    async def aclose(self) -> None:
        """Release any connections."""
        ...


@runtime_checkable
class UnderstandingProvider(Protocol):
    """A backend that turns an `UnderstandingRequest` into a text description.

    Same submit/poll/fetch/cancel shape as `ClipProvider`/`ImageProvider` for
    the same reason: a cold model load for a >16GB understanding model could
    plausibly exceed the ~100s proxy window even though a warm caption call
    is fast. `fetch()` does no network I/O of its own here -- the result is
    text already captured in `job.raw` at poll time, not a file to download.
    """

    name: str
    residency_group: str
    """Which side of the one GPU this provider\'s model lives on ("comfyui" or
    "inference"); `pipeline.residency.make_room_for` evicts the other side."""

    def capabilities(self) -> UnderstandingCapabilities:
        """Declare what this backend can do, its cost and its latency, without
        touching the network."""
        ...

    async def submit(self, request: UnderstandingRequest) -> UnderstandingJob:
        """Enqueue one request. Returns immediately with a job handle."""
        ...

    async def poll(self, job: UnderstandingJob) -> UnderstandingJob:
        """Refresh a job's state. Never blocks longer than one HTTP call."""
        ...

    async def fetch(self, job: UnderstandingJob) -> UnderstandingAsset:
        """Describe a completed job. No file to download: the text was captured
        into `job.raw` at poll time."""
        ...

    async def cancel(self, job: UnderstandingJob) -> None:
        """Best-effort cancellation. Must not raise if the job already ended."""
        ...

    async def aclose(self) -> None:
        """Release any connections."""
        ...


@runtime_checkable
class ChatProvider(Protocol):
    """A backend that turns a `ChatRequest` into a text reply.

    Same submit/poll/fetch/cancel shape as `UnderstandingProvider` for the
    same reason: a cold gpt-oss-20b load could plausibly exceed the ~100s
    proxy window even though a warm reply is fast. Kept as its own Protocol
    rather than reusing `UnderstandingProvider`'s, because a chat request has
    no input media to require.
    """

    name: str
    residency_group: str
    """Which side of the one GPU this provider\'s model lives on ("comfyui" or
    "inference"); `pipeline.residency.make_room_for` evicts the other side."""

    def capabilities(self) -> ChatCapabilities:
        """Declare what this backend can do, its cost and its latency, without
        touching the network."""
        ...

    async def submit(self, request: ChatRequest) -> ChatJob:
        """Enqueue one request. Returns immediately with a job handle."""
        ...

    async def poll(self, job: ChatJob) -> ChatJob:
        """Refresh a job's state. Never blocks longer than one HTTP call."""
        ...

    async def fetch(self, job: ChatJob) -> ChatAsset:
        """Describe a completed job. No file to download: the text was captured
        into `job.raw` at poll time."""
        ...

    async def cancel(self, job: ChatJob) -> None:
        """Best-effort cancellation. Must not raise if the job already ended."""
        ...

    async def aclose(self) -> None:
        """Release any connections."""
        ...
