"""Free text → a Flux.1-dev prompt.

Deliberately thin, unlike `prompts/h3.py`. H3's structured shot schema exists
because it is *measured* to matter (26.0 -> 367.6 on the same scene, see
`prompts/convert.py`); Flux has no equivalent published schema; it takes plain
natural-language T5/CLIP prose. So this module does not build a schema — it
does the minimum that makes a LINE message safe to submit: strip, collapse
whitespace, enforce a length ceiling, and refuse an empty request.

`convert()` mirrors `prompts.convert.convert()`'s `(prompt, how)` return shape
so `pipeline.convert_worker.convert_job()` can branch on `media_kind` without
a different call shape either side. `client` is accepted but unused in this
version — there is no LLM step in the image path, so no serverless cost is
added by supporting it.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from ai_studio.prompts.convert import ConversionError, LlmClient

DEFAULT_MAX_CHARS = 2000


class FluxPrompt(BaseModel):
    """A prompt ready to submit to Flux. No shots, no camera, no schema."""

    model_config = ConfigDict(frozen=True)

    text: str

    def render(self) -> str:
        return self.text


def build_prompt(text: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> FluxPrompt:
    """Strip/collapse whitespace and truncate. Raises on empty input."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ConversionError("empty prompt")
    return FluxPrompt(text=cleaned[:max_chars])


async def convert(
    text: str,
    client: LlmClient | None = None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[FluxPrompt, str]:
    """Convert `text` into a `FluxPrompt`. Returns `(prompt, how)`.

    `client` is accepted for call-shape symmetry with `prompts.convert.convert`
    but ignored in this version — v1 always returns `"template"`. The LLM hook
    is reserved for a future "translate + tidy" pass, not built now.
    """
    return build_prompt(text, max_chars=max_chars), "template"
