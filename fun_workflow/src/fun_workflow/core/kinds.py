"""What a request *is*, as opposed to what model serves it.

`JobKind` is the queue's discriminator: every row, trigger, status wording
and cap is keyed on it. `MediaKind` (ai-studio) is what a provider serves.
The two coincide for every kind except DRAMA, which is not a model at all
but a pipeline over the IMAGE and VIDEO providers -- so `model_kind` is
None there and the drama renderer picks its providers itself.

The values are the strings the queue has always stored (`jobs.media_kind`),
so no row needs rewriting.
"""

from __future__ import annotations

from enum import Enum

from ai_studio.core.enums import MediaKind


class JobKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    IMAGE_UNDERSTAND = "image_understand"
    """/說圖: describe a photo."""
    AUDIO_UNDERSTAND = "audio_understand"
    """/說音: describe an audio clip."""
    VIDEO_UNDERSTAND = "video_understand"
    """/說影: describe a video clip -- two models, two answers."""
    CHAT = "chat"
    """/himonkey: a plain-text reply. Not an understanding kind: it describes
    nothing. It shares the inference server's GPU slot, but that is the
    provider's `residency_group`, not a property of the request."""
    DRAMA = "drama"
    """/短劇: a ~60 s six-shot story with one recurring character. Not a
    single provider call: `pipeline.drama.render_drama` drives the IMAGE
    provider (character sheet, keyframes) and then the VIDEO provider (six
    image-to-video clips) and concatenates."""

    @property
    def is_understanding(self) -> bool:
        return self in _UNDERSTANDING_KINDS

    @property
    def is_generation(self) -> bool:
        """Every GPU second is a Flux or an H3 job: ComfyUI's side of the card."""
        return self in _GENERATION_KINDS

    @property
    def model_kind(self) -> MediaKind | None:
        """The provider key this job renders on, or None for a kind that
        drives several providers itself (DRAMA)."""
        return None if self is JobKind.DRAMA else MediaKind(self.value)


_UNDERSTANDING_KINDS = frozenset(
    {JobKind.IMAGE_UNDERSTAND, JobKind.AUDIO_UNDERSTAND, JobKind.VIDEO_UNDERSTAND}
)
_GENERATION_KINDS = frozenset({JobKind.VIDEO, JobKind.IMAGE, JobKind.DRAMA})
