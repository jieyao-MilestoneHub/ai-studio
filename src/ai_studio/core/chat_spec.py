"""The chat-provider contract. Sibling to `understanding_spec.py`, not a
promotion of `prompts.convert.LlmClient`.

A chat turn has to sit inside the caller's window-deadline poll loop
the same way a clip or a description does, so a turn straddling the window's
close can be cancelled before billing more GPU-seconds -- that needs an
externally pollable job handle, not one opaque awaited call. It also has no
input media (`UnderstandingRequest.input_media_path` is required; a chat
message is text in, text out), which is why this is its own quartet rather
than a widened `UnderstandingRequest`.

`ChatRequest` deliberately carries no `user_id`. Per-user isolation is
enforced entirely host-side, keyed on `Job.user_id` directly
(the caller reads/writes its own store's
`chat_turns` table before a `ChatRequest` is ever built) -- the pod-side
wire protocol has no need to know *whose* conversation it is answering, only
what to say next, so the field would sit on this type unread. If a future
caller genuinely needs it on the wire, add it deliberately then.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_studio.core.enums import JobState


class ChatCapabilities(BaseModel):
    """What the chat backend can actually do. One instance, not one per
    modality -- unlike understanding's three backing models, there is only
    ever one chat model."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model_id: str

    max_output_chars: int = Field(
        default=1000,
        gt=0,
        description="Ceiling on the returned reply. The caller's delivery "
        "channel decides the number; 1000 is a conservative default.",
    )
    cost_per_call_usd: float = Field(default=0.0, ge=0)
    expected_latency_s: float = Field(default=20.0, gt=0)
    max_concurrent_jobs: int = Field(
        default=1,
        gt=0,
        description="Kept at 1: one 24GB card holds at most one of "
        "{ComfyUI's resident checkpoint, an understanding model, gpt-oss-20b} "
        "at a time.",
    )


class ChatRequest(BaseModel):
    """One reply to produce. Provider-agnostic."""

    model_config = ConfigDict(frozen=True)

    shot_id: str
    text: str
    """The new message, verbatim. No LLM rewrite step -- unlike H3/Flux
    prompts, gpt-oss-20b gets the user's own words directly."""

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific knobs, kept out of the typed surface "
        "on purpose. Carries `history`: the rendered prior turns for this "
        "user, fetched host-side before submit and never persisted on the "
        "pod itself.",
    )


class ChatJob(BaseModel):
    """A submitted chat job. Same lifecycle shape as `UnderstandingJob`."""

    model_config = ConfigDict(frozen=True)

    provider: str
    job_id: str
    shot_id: str
    state: JobState = JobState.PENDING

    submitted_at: float
    updated_at: float
    queue_position: int | None = None

    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.updated_at - self.submitted_at)

    def with_state(self, state: JobState, *, now: float, **changes: Any) -> ChatJob:
        return self.model_copy(update={"state": state, "updated_at": now, **changes})


class ChatAsset(BaseModel):
    """A produced reply. No output file -- the same reason
    `UnderstandingAsset` has none."""

    model_config = ConfigDict(frozen=True)

    shot_id: str
    provider: str
    job_id: str
    result_text: str
    reasoning_exhausted: bool = False
    """The model thought until its budget ran out and wrote no answer;
    `result_text` is "". The caller words what to say about that."""
    cost_usd: float = Field(default=0.0, ge=0)
