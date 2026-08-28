"""`prompts.understanding`: the engineered default questions and the rewriter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ai_studio.core.enums import MediaKind
from ai_studio.llm.scripted import ScriptedLlmClient
from ai_studio.prompts import understanding as und

REPO = Path(__file__).resolve().parents[2]


def test_defaults_are_per_model_and_image_means_caption_mode() -> None:
    assert und.default_question(MediaKind.IMAGE_UNDERSTAND) is None
    audio = und.default_question(MediaKind.AUDIO_UNDERSTAND)
    video = und.default_question(MediaKind.VIDEO_UNDERSTAND)
    assert audio and audio.startswith("請只用繁體中文")
    assert video and "不要簡體字" in video
    # One language, a fixed shape: the first live run dropped a second language.
    assert "英文" not in audio and "English" not in audio
    with pytest.raises(ValueError):
        und.default_question(MediaKind.VIDEO)


def test_the_server_carries_byte_identical_copies_of_the_default_questions() -> None:
    """deploy/inference_server.py cannot import ai_studio, so the defaults are
    duplicated there. If the two drift, a bare trigger and a CLI call would
    ask different questions and nobody would see why."""
    path = REPO / "deploy" / "inference_server.py"
    spec = importlib.util.spec_from_file_location("srv_for_prompt_pin", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(module)
    assert module.AUDIO_DEFAULT_QUESTION == und.AUDIO_DEFAULT_QUESTION
    assert module.VIDEO_DEFAULT_QUESTION == und.VIDEO_DEFAULT_QUESTION


@pytest.mark.asyncio
async def test_no_text_is_the_default_and_calls_no_llm() -> None:
    client = ScriptedLlmClient()  # any call would raise: no replies scripted
    q, how = await und.convert_question("  ", client, modality=MediaKind.AUDIO_UNDERSTAND)
    assert how == "understanding-default" and q == und.AUDIO_DEFAULT_QUESTION
    assert client.calls == []


@pytest.mark.asyncio
async def test_text_without_a_client_is_sent_as_typed() -> None:
    q, how = await und.convert_question("這是誰", None, modality=MediaKind.IMAGE_UNDERSTAND)
    assert (q, how) == ("這是誰", "understanding-raw")


@pytest.mark.asyncio
async def test_a_question_is_rewritten_with_the_models_own_system_prompt() -> None:
    client = ScriptedLlmClient('{"question": "What breed is the dog in the photo?"}')
    q, how = await und.convert_question("這是什麼品種的狗", client, modality=MediaKind.IMAGE_UNDERSTAND)
    assert how == "understanding-llm"
    assert q == "What breed is the dog in the photo?"
    system, user = client.calls[0]
    assert system is und.REWRITE_SYSTEM[MediaKind.IMAGE_UNDERSTAND]
    assert "answers only in English" in system
    assert "這是什麼品種的狗" in user


@pytest.mark.asyncio
async def test_a_bad_reply_is_retried_then_falls_back_to_the_users_words_labelled() -> None:
    client = ScriptedLlmClient("not json", "still not json")
    q, how = await und.convert_question("他在唱什麼", client, modality=MediaKind.AUDIO_UNDERSTAND)
    assert q == "他在唱什麼"
    assert how.startswith("understanding-template (llm failed:")
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_a_second_attempt_that_succeeds_is_a_retry() -> None:
    client = ScriptedLlmClient("garbage", '{"question": "請只用繁體中文說明影片最後發生了什麼"}')
    q, how = await und.convert_question("最後怎麼了", client, modality=MediaKind.VIDEO_UNDERSTAND)
    assert how == "understanding-llm-retry" and q.startswith("請只用繁體中文")


@pytest.mark.asyncio
async def test_a_transport_failure_falls_back_rather_than_raising() -> None:
    class _Down:
        async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
            raise ConnectionError("pod unreachable")

    q, how = await und.convert_question("誰在說話", _Down(), modality=MediaKind.AUDIO_UNDERSTAND)
    assert q == "誰在說話" and "ConnectionError" in how


def test_rewriter_system_prompts_name_the_models_language() -> None:
    assert "only in English" in und.REWRITE_SYSTEM[MediaKind.IMAGE_UNDERSTAND]
    for kind in (MediaKind.AUDIO_UNDERSTAND, MediaKind.VIDEO_UNDERSTAND):
        assert "請只用繁體中文" in und.REWRITE_SYSTEM[kind]
    # /說影 hears too (a second model on the track); the rewriter must know.
    assert "listens to\nits audio track" in und.REWRITE_SYSTEM[MediaKind.VIDEO_UNDERSTAND]
    assert "cannot hear" not in und.REWRITE_SYSTEM[MediaKind.VIDEO_UNDERSTAND]
