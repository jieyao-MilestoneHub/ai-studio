"""Questions for the understanding models: the engineered defaults, and the
rewriter that turns a group member's own question into each model's best form.

Pure, like the rest of `prompts`: the only outside contact is the `LlmClient`
protocol from `convert.py`, injected by the pipeline. The LLM behind it is
gpt-oss-20b on the pod (`pipeline.pod_llm.PodLlmClient`) -- never a serverless
endpoint (decision 2026-08-27).

Why per-model shapes, from each model's docs and its first live run:

- `/說影` is two models: Qwen2.5-VL on the frames, then Qwen2-Audio on the
  ffmpeg-extracted track (composed in `deploy/inference_server.py::_run_job`,
  headings 【畫面】/【聲音】). Qwen2.5-VL alone is deaf -- `process_vision_info`
  samples frames only -- so a user asking「他說了什麼」would have got a guess
  from lip shapes. The same question reaches both models.
- moondream3-preview answers **only in English** (📏 2026-08-27, asked three
  ways). It has two skills: `caption(length=...)` for a description with no
  question, and `query(question)` for a specific one. So the default sends
  *no* question (the server picks the long caption) and a user question is
  rewritten as one compact English question about what is visible.
- Qwen2-Audio-7B-Instruct takes a text instruction alongside the audio and
  follows it well, but 📏 dropped the second half of a two-language ask -- so
  one language, a fixed shape.
- Qwen2.5-VL-7B-Instruct likewise; 📏 it drifted to 简体 once without the
  台灣用字 instruction.

`AUDIO_DEFAULT_QUESTION` and `VIDEO_DEFAULT_QUESTION` are duplicated verbatim
in `deploy/inference_server.py` (which cannot import this package). A unit
test pins the two copies together.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_studio.core.enums import MediaKind
from ai_studio.prompts.convert import ConversionError, LlmClient, extract_json

AUDIO_DEFAULT_QUESTION = (
    "請只用繁體中文,依下列格式條列描述這段聲音,不要加開場白:\n"
    "類型:(人聲說話 / 唱歌 / 音樂 / 環境音 / 混合)\n"
    "內容:(若有人說話或唱歌,盡量逐字寫出說的內容;若是音樂,寫出樂器、曲風、節奏、有無人聲)\n"
    "細節:(說話者人數與性別、語言、語氣情緒;背景聲、環境、音質)\n"
    "一句話總結:"
)

VIDEO_DEFAULT_QUESTION = (
    "請只用繁體中文(台灣用字,不要簡體字)描述這段影片,依下列項目條列,不要加開場白:\n"
    "場景:(地點、時間、天氣、環境)\n"
    "人物/主體:(人數、外觀、衣著)\n"
    "動作與變化:(依時間順序,開頭 → 中段 → 結尾 發生了什麼)\n"
    "畫面文字:(字幕、招牌、標誌上的文字,逐字寫出;沒有就寫「無」)\n"
    "鏡頭:(靜止或移動、拍攝角度、有無剪接)\n"
    "一句話總結:"
)

DEFAULT_QUESTIONS: dict[MediaKind, str | None] = {
    MediaKind.IMAGE_UNDERSTAND: None,  # caption(length="long") on the server
    MediaKind.AUDIO_UNDERSTAND: AUDIO_DEFAULT_QUESTION,
    MediaKind.VIDEO_UNDERSTAND: VIDEO_DEFAULT_QUESTION,
}

_IMAGE_REWRITE = """\
You turn a group-chat question about a photo (usually Traditional Chinese)
into ONE clear English question for a small vision-language model that
answers only in English. Reply with JSON only, no prose and no code fence:
{"question": "..."}

Rules:
- Translate the user's intent faithfully; do not answer it, do not add asks.
- Make it specific and answerable from pixels alone: what is visible, how
  many, what colour, what is written, where things are.
- One question, under 30 words, ending with a question mark. If the user
  asks several things, join them into one sentence with "and".
- If they ask something the pixels cannot show (who a person is, the price,
  the date), rephrase to what is observable ("describe the person's
  appearance and clothing").
- If they ask for a general description, ask for "a detailed description
  of ..." naming the aspect they care about.
- Never add "answer in Chinese" -- the model cannot.
"""

_AUDIO_REWRITE = """\
You turn a group-chat question about an audio clip (Traditional Chinese)
into one precise instruction for an audio-understanding model that listens
to the clip and answers in Traditional Chinese. Reply with JSON only, no
prose and no code fence: {"question": "..."}

Rules:
- Write the instruction in Traditional Chinese (台灣用字), starting with
  「請只用繁體中文」.
- Keep the user's intent exactly; say what to report and in what form
  (逐字稿 / 條列 / 一句話).
- Cover only the relevant facets among: 說話內容(逐字)、說話者人數與性別、
  語言與口音、語氣情緒、音樂的樂器/曲風/節奏、環境聲、音質.
- Do not ask for two languages; do not ask for anything the audio alone
  cannot show. Under 80 characters.
"""

_VIDEO_REWRITE = """\
You turn a group-chat question about a short video (Traditional Chinese)
into one precise instruction for a video-understanding pipeline: one model
watches the clip at about one frame per second, a second model listens to
its audio track, and each answers in Traditional Chinese.
Reply with JSON only, no prose and no code fence: {"question": "..."}

Rules:
- Write in Traditional Chinese (台灣用字,不要簡體字), starting with
  「請只用繁體中文」.
- Keep the user's intent exactly; make it answerable from the frames: what
  is visible, what changes over time, on-screen text, counts, colours,
  camera movement.
- Ask for a short bullet list when the user asks for more than one thing,
  otherwise one or two sentences.
- The clip's sound is analysed too, by a second model that hears the
  track: a question about speech, music or noises is fine -- phrase it so
  it makes sense for both what is seen and what is heard. Under 80
  characters.
"""

REWRITE_SYSTEM: dict[MediaKind, str] = {
    MediaKind.IMAGE_UNDERSTAND: _IMAGE_REWRITE,
    MediaKind.AUDIO_UNDERSTAND: _AUDIO_REWRITE,
    MediaKind.VIDEO_UNDERSTAND: _VIDEO_REWRITE,
}


class _Question(BaseModel):
    """What the rewriter is allowed to return."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1, max_length=400)


def default_question(modality: MediaKind) -> str | None:
    """The engineered question a bare trigger sends. Raises on a kind that is
    not an understanding kind -- fail loudly, never a blank prompt."""
    if modality not in DEFAULT_QUESTIONS:
        raise ValueError(f"{modality!r} is not an understanding kind")
    return DEFAULT_QUESTIONS[modality]


async def convert_question(
    text: str,
    client: LlmClient | None,
    *,
    modality: MediaKind,
    attempts: int = 2,
) -> tuple[str | None, str]:
    """The question to send for one understanding job. Returns `(question, how)`.

    `how` is one of `understanding-default` (no user text: the engineered
    default, or None for the image caption path), `understanding-raw` (user
    text, no client: sent as typed), `understanding-llm` / `understanding-
    llm-retry` (rewritten), or `understanding-template (llm failed: ...)`
    (user text sent as typed, with the failure attached). Never raises on a
    rewrite failure -- the user's own words are a worse question than the
    rewritten one, but no question at all would answer nothing.
    """
    if modality not in REWRITE_SYSTEM:
        raise ValueError(f"{modality!r} is not an understanding kind")
    text = text.strip()
    if not text:
        return default_question(modality), "understanding-default"
    if client is None:
        return text, "understanding-raw"

    user = f"User's question: {text}\nReply with the JSON object only."
    last_error = ""
    for attempt in range(attempts):
        try:
            reply = await client.complete(REWRITE_SYSTEM[modality], user, max_tokens=300)
            payload: dict[str, Any] = extract_json(reply)
            question = _Question(**payload).question.strip()
            return question, ("understanding-llm" if attempt == 0 else "understanding-llm-retry")
        except (ConversionError, ValidationError, TypeError) as exc:
            last_error = str(exc)
        except Exception as exc:  # network, timeout, anything at all
            last_error = f"{type(exc).__name__}: {exc}"
    return text, f"understanding-template (llm failed: {last_error[:200]})"
