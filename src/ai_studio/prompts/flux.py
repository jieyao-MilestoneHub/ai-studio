"""Free text → a Flux.1-dev prompt.

Still much thinner than `prompts/h3.py`, and for the same reason as before: H3's
structured shot schema exists because it is *measured* to matter (26.0 -> 367.6
on the same scene, see `prompts/convert.py`), while Flux has no equivalent
published schema — it takes plain natural-language T5/CLIP prose. There is no
shot list to build here and inventing one would be cargo cult.

What it is no longer is a passthrough. **The trigger message is always Chinese
and Flux's text encoders are very weak at it**, so a translation step is not a
nice-to-have on this path, it is the difference between the picture somebody
asked for and a picture of something else. That is the whole of "prompt
engineering" on the image side, and it is why `convert()`'s `client` argument —
which this module used to accept and deliberately ignore — is now connected.

Two properties carried over from the video side on purpose:

- **The LLM never produces the final string.** It returns JSON, which is
  validated into `FluxPrompt`; `render()` is the only thing that produces text
  for a sampler. The rules `FluxPrompt` enforces are thinner than H3's, but the
  shape is the same and so is the reason: a model that returns three paragraphs
  of commentary fails validation instead of being submitted.
- **A failure falls back, it does not drop the request.** `built_by` records
  which happened, so a disappointing image can be traced to a prompt that never
  got translated.

The LoRA in play (`Heartsync/Flux-NSFW-uncensored`) needs **no trigger word** —
its model card declares no `instance_prompt`, and it is an "unrestrain" adapter
rather than a concept LoRA. That is verified 📏, and it is why this module is
smaller than it was originally scoped to be: there is no magic token to inject.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ai_studio.prompts.convert import ConversionError, LlmClient, extract_json

DEFAULT_MAX_CHARS = 2000
"""Ceiling on the string `render()` produces — **characters, not tokens.**

Deliberately blunt, and worth knowing exactly what it is and is not protecting:

- CLIP-L, one of Flux's two text encoders, reads only the first **77 tokens**
  and silently ignores the rest. T5-XXL reads far more (512 in the usual
  ComfyUI configuration).
- So this ceiling protects the *request* (a pathological 50KB chat message
  should not be submitted), not the *quality*. Nothing here can stop CLIP-L
  from ignoring the tail, because nothing can.
- What follows from that is the ordering in `render()`: the subject goes first,
  where both encoders see it, and the quality tail goes last, where only T5
  does. Splitting these into two separate bindings — one string per encoder —
  is the real fix and is deliberately not this change.

Truncation happens to the *subject* in `build_prompt`, before the quality tail
is appended, so the tail can never be the part that gets cut off.
"""

QUALITY_SUFFIX = "highly detailed, sharp focus, natural lighting, photorealistic"
"""Appended to every prompt by `render()`.

Generic on purpose. These are the terms Flux.1-dev responds to across subjects;
anything more opinionated is a style choice that belongs to the person asking,
not to the plumbing. It sits at the end because of the CLIP-L window above.
"""


class FluxPrompt(BaseModel):
    """A prompt ready to submit to Flux. No shots, no camera, no schema."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    """The subject, in English, already cleaned and truncated."""

    quality: str = QUALITY_SUFFIX
    """The tail `render()` appends. A field rather than a constant reference so
    a run's stored prompt records what was actually appended to it."""

    def render(self) -> str:
        """The one place a final prompt string is produced."""
        return f"{self.text}, {self.quality}" if self.quality else self.text


MAX_WORDS = 90
"""A reply longer than this is rejected and retried rather than accepted.

Flux.1-dev's guidance is 40-60 words; past ~200 the T5 encoder summarises and
detail is dropped silently, and CLIP-L only ever sees the first 77 tokens. A
long answer is the model ignoring the rules, so it is a failed conversion,
not a prompt to truncate."""


class _Translated(BaseModel):
    """What the LLM is allowed to return. Anything else is a failed conversion."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1, max_length=DEFAULT_MAX_CHARS)

    @model_validator(mode="after")
    def _not_too_long(self) -> _Translated:
        words = len(self.prompt.split())
        if words > MAX_WORDS:
            raise ValueError(f"prompt is {words} words; the ceiling is {MAX_WORDS}")
        return self


SYSTEM_PROMPT = """\
You rewrite a casual description of a picture (usually Traditional Chinese) as
an English image-generation prompt for the Flux.1-dev model. Reply with JSON
only, no prose and no code fence.

Schema:
{"prompt": "<the English prompt>"}

Rules:
- Always English, whatever language the request is in. Proper nouns stay
  verbatim; on-screen text the user asked for stays verbatim in double quotes.
- The request is the spec: translate faithfully, do not embellish, do not
  refuse or sanitise. Every concrete word the user used (objects, colours,
  clothing, styles, places, names, numbers) survives as its plain English
  equivalent. A thin request gives a short prompt, never one padded with
  invented detail.
- Write 1-3 flowing natural-language sentences, 40-60 words, in this order:
  subject (who/what, appearance, clothing) -> what they are doing / pose ->
  environment and time of day -> materials and textures (be physical:
  brushed steel, coarse wool, cracked leather) -> lighting as direction +
  colour + quality ("soft window light from the left") -> style or medium
  ("35mm photo", "watercolour illustration") only if the user named one.
- Only visible things. No mood words, no praise ("beautiful", "masterpiece",
  "8k"), no tag lists, no headings, no weights or parentheses like (word:1.2).
- No negatives: phrase everything positively ("an empty street", not "no
  people"). Never write the phrase "white background"; if the user wants a
  plain one, say "a plain light-grey studio backdrop".
- Do not append quality words; they are added separately.
- Output ONE line of minified JSON: nothing before the opening brace or
  after the closing one.
"""


def build_prompt(
    text: str, *, max_chars: int = DEFAULT_MAX_CHARS, quality: str = QUALITY_SUFFIX
) -> FluxPrompt:
    """Strip/collapse whitespace and truncate. Raises on empty input.

    The budget is spent on the subject: `max_chars` bounds what `render()`
    finally produces, so the quality tail is subtracted first and the subject
    truncated to what is left. The tail is never the part that gets cut.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ConversionError("empty prompt")
    budget = max_chars - (len(quality) + 2 if quality else 0)
    if budget <= 0:
        raise ConversionError(
            f"max_chars={max_chars} leaves no room for the subject after a "
            f"{len(quality)}-character quality tail"
        )
    return FluxPrompt(text=cleaned[:budget], quality=quality)


async def convert(
    text: str,
    client: LlmClient | None = None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    attempts: int = 2,
) -> tuple[FluxPrompt, str]:
    """Convert `text` into a `FluxPrompt`. Returns `(prompt, how)`.

    `how` is "llm", "llm-retry", or "template..." with the reason attached, and
    is stored on the job so a disappointing image can be traced to a prompt
    that was never actually translated.

    **Never raises on a translation failure.** Submitting the original Chinese
    produces a worse picture; dropping the request produces none at all, and
    the user is already waiting. Same trade as the video path.
    """
    if client is None:
        return build_prompt(text, max_chars=max_chars), "template"

    user = f"Request: {text.strip()}\nReply with the JSON object only."

    last_error = ""
    for attempt in range(attempts):
        try:
            reply = await client.complete(SYSTEM_PROMPT, user, max_tokens=400)
            payload: dict[str, Any] = extract_json(reply)
            translated = _Translated(**payload)
            return (
                build_prompt(translated.prompt, max_chars=max_chars),
                "llm" if attempt == 0 else "llm-retry",
            )
        except (ConversionError, ValidationError, TypeError) as exc:
            last_error = str(exc)
        except Exception as exc:  # network, timeout, anything at all
            last_error = f"{type(exc).__name__}: {exc}"

    return (
        build_prompt(text, max_chars=max_chars),
        f"template (llm failed: {last_error[:200]})",
    )
