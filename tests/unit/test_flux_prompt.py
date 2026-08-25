"""The image-side prompt path.

The load-bearing assertion in this file is the boring one: **the trigger
message is always Chinese and Flux's text encoders are very weak at it.** So
translation is not a refinement here, it is the difference between the picture
somebody asked for and a picture of something else — and a translation that
fails must fall back rather than drop the request, because the user is already
waiting.
"""

from __future__ import annotations

import pytest

from ai_studio.prompts.convert import ConversionError
from ai_studio.prompts.flux import (
    DEFAULT_MAX_CHARS,
    QUALITY_SUFFIX,
    FluxPrompt,
    build_prompt,
    convert,
)

CHINESE = "一隻橘貓坐在窗邊看雨"
ENGLISH = "an orange tabby cat sitting by a window watching the rain"


class FakeLlm:
    """Replays canned replies and records what it was asked."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
        self.calls.append((system, user))
        reply = self.replies.pop(0) if self.replies else '{"prompt": "fallback"}'
        if isinstance(reply, Exception):
            raise reply
        return reply


# ------------------------------------------------------------------ building


def test_an_empty_request_raises_rather_than_submitting_nothing() -> None:
    with pytest.raises(ConversionError, match="empty prompt"):
        build_prompt("   \n\t  ")


def test_whitespace_is_collapsed() -> None:
    assert build_prompt("a   cat\n\nby a  window").text == "a cat by a window"


def test_render_is_the_only_place_the_final_string_is_made() -> None:
    """Same rule as the video side: the model returns data, `render()` returns
    the string a sampler sees."""
    prompt = build_prompt(ENGLISH)

    assert prompt.render() == f"{ENGLISH}, {QUALITY_SUFFIX}"
    assert prompt.text == ENGLISH, "the subject is stored without the tail"


def test_the_quality_tail_is_never_what_gets_truncated() -> None:
    """It is appended after truncation, so the budget is spent on the subject.
    A half-sentence of quality words at the end would be worse than none."""
    prompt = build_prompt("cat " * 2000)

    rendered = prompt.render()
    assert len(rendered) <= DEFAULT_MAX_CHARS
    assert rendered.endswith(QUALITY_SUFFIX)


def test_truncation_happens_at_the_documented_ceiling() -> None:
    prompt = build_prompt("x" * 5000, max_chars=200)

    assert len(prompt.render()) <= 200
    assert len(prompt.text) == 200 - len(QUALITY_SUFFIX) - 2


def test_a_ceiling_too_small_for_the_tail_raises_rather_than_producing_junk() -> None:
    with pytest.raises(ConversionError, match="no room for the subject"):
        build_prompt(ENGLISH, max_chars=10)


def test_the_subject_comes_first_because_clip_l_only_reads_the_start() -> None:
    """CLIP-L sees ~77 tokens and ignores the rest; T5 reads far more. Putting
    the generic quality words first would spend the whole CLIP window on
    boilerplate."""
    rendered = build_prompt(ENGLISH).render()

    assert rendered.index(ENGLISH) < rendered.index(QUALITY_SUFFIX)


# --------------------------------------------------------------- translation


@pytest.mark.asyncio
async def test_chinese_is_translated_to_english() -> None:
    llm = FakeLlm(f'{{"prompt": "{ENGLISH}"}}')

    prompt, how = await convert(CHINESE, llm)

    assert how == "llm"
    assert prompt.text == ENGLISH
    assert CHINESE not in prompt.render()


@pytest.mark.asyncio
async def test_the_request_is_what_the_model_is_asked_about() -> None:
    llm = FakeLlm(f'{{"prompt": "{ENGLISH}"}}')

    await convert(CHINESE, llm)

    (system, user) = llm.calls[0]
    assert CHINESE in user
    assert "English" in system


@pytest.mark.asyncio
async def test_a_code_fence_around_the_json_is_tolerated() -> None:
    """Models add fences however firmly you ask them not to."""
    llm = FakeLlm(f'```json\n{{"prompt": "{ENGLISH}"}}\n```')

    prompt, how = await convert(CHINESE, llm)

    assert how == "llm"
    assert prompt.text == ENGLISH


@pytest.mark.asyncio
async def test_a_bad_reply_is_retried_once_then_accepted() -> None:
    llm = FakeLlm("I cannot help with that.", f'{{"prompt": "{ENGLISH}"}}')

    prompt, how = await convert(CHINESE, llm)

    assert how == "llm-retry"
    assert prompt.text == ENGLISH


@pytest.mark.asyncio
async def test_a_model_that_never_returns_json_falls_back_to_the_original() -> None:
    """A worse picture beats no picture. The user is already waiting, and the
    reason is recorded so a disappointing image can be explained afterwards."""
    llm = FakeLlm("sorry", "still sorry")

    prompt, how = await convert(CHINESE, llm)

    assert how.startswith("template (llm failed:")
    assert prompt.text == CHINESE, "the original is submitted, not nothing"


@pytest.mark.asyncio
async def test_a_network_failure_falls_back_too() -> None:
    llm = FakeLlm(TimeoutError("the endpoint was cold"), TimeoutError("still cold"))

    prompt, how = await convert(CHINESE, llm)

    assert "TimeoutError" in how
    assert prompt.text == CHINESE


@pytest.mark.asyncio
async def test_an_empty_translation_is_rejected_not_submitted() -> None:
    """An empty string is a valid JSON reply and a completely invalid prompt."""
    llm = FakeLlm('{"prompt": ""}', '{"prompt": ""}')

    prompt, how = await convert(CHINESE, llm)

    assert how.startswith("template (llm failed:")
    assert prompt.text == CHINESE


@pytest.mark.asyncio
async def test_no_client_means_no_llm_call_and_no_cost() -> None:
    """`convert_worker` passes `None` when no endpoint is configured; that must
    stay a working path, not a crash."""
    prompt, how = await convert(CHINESE, None)

    assert how == "template"
    assert prompt.text == CHINESE


@pytest.mark.asyncio
async def test_an_empty_request_still_raises_through_convert() -> None:
    with pytest.raises(ConversionError, match="empty prompt"):
        await convert("   ", None)


# ------------------------------------------------------------------ storage


def test_the_prompt_round_trips_through_the_queues_json() -> None:
    """`convert_worker` stores `model_dump(mode="json")` and the worker reads
    `_rendered` back out. A field that does not survive that trip is a field
    the GPU never sees."""
    prompt = build_prompt(ENGLISH)

    payload = prompt.model_dump(mode="json")
    payload["_rendered"] = prompt.render()

    assert FluxPrompt(**{k: v for k, v in payload.items() if not k.startswith("_")}) == prompt
    assert payload["_rendered"].startswith(ENGLISH)
