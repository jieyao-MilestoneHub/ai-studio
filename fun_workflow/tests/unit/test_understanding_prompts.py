"""`prompts.understanding`: the engineered default questions, the rewriter,
and the wording of what comes back."""

from __future__ import annotations

import pytest
from ai_studio.core.enums import MediaKind
from ai_studio.core.understanding_spec import UnderstandingAsset, VideoSections
from ai_studio.llm.scripted import ScriptedLlmClient

from fun_workflow.prompts import understanding as und


def test_defaults_are_per_model_and_image_means_caption_mode() -> None:
    assert und.default_questions(MediaKind.IMAGE_UNDERSTAND) == (None, None)
    audio, no_second = und.default_questions(MediaKind.AUDIO_UNDERSTAND)
    video, track = und.default_questions(MediaKind.VIDEO_UNDERSTAND)
    assert audio and audio.startswith("請只用繁體中文") and no_second is None
    assert video and "不要簡體字" in video
    # The audio model gets its own, shorter question about the track -- not
    # the frame-oriented one (the bug the split found).
    assert track == und.VIDEO_AUDIO_QUESTION and "場景" not in track
    # One language, a fixed shape: the first live run dropped a second language.
    assert "英文" not in audio and "English" not in audio
    with pytest.raises(ValueError):
        und.default_questions(MediaKind.VIDEO)


@pytest.mark.asyncio
async def test_no_text_is_the_default_and_calls_no_llm() -> None:
    client = ScriptedLlmClient()  # any call would raise: no replies scripted
    q, aq, how = await und.convert_question("  ", client, modality=MediaKind.AUDIO_UNDERSTAND)
    assert how == "understanding-default" and q == und.AUDIO_DEFAULT_QUESTION and aq is None
    q, aq, how = await und.convert_question("", client, modality=MediaKind.VIDEO_UNDERSTAND)
    assert (q, aq) == (und.VIDEO_DEFAULT_QUESTION, und.VIDEO_AUDIO_QUESTION)
    assert client.calls == []


@pytest.mark.asyncio
async def test_text_without_a_client_is_sent_as_typed() -> None:
    q, aq, how = await und.convert_question("這是誰", None, modality=MediaKind.IMAGE_UNDERSTAND)
    assert (q, aq, how) == ("這是誰", None, "understanding-raw")
    q, aq, how = await und.convert_question("他說了什麼", None, modality=MediaKind.VIDEO_UNDERSTAND)
    assert (q, aq) == ("他說了什麼", "他說了什麼"), "a user's own question reaches both models"


def test_compose_answer_joins_the_video_sections_and_notes_silence() -> None:
    common = dict(shot_id="j", provider="p", job_id="1", modality=MediaKind.VIDEO_UNDERSTAND)
    heard = UnderstandingAsset(sections=VideoSections(visual="a cat", audio="purring", has_audio_track=True), **common)
    silent = UnderstandingAsset(sections=VideoSections(visual="a cat", audio=None, has_audio_track=False), **common)
    plain = UnderstandingAsset(result_text="a dog", **{**common, "modality": MediaKind.IMAGE_UNDERSTAND})

    assert und.compose_answer(heard) == "【畫面】\na cat\n\n【聲音】\npurring"
    assert und.compose_answer(silent) == f"【畫面】\na cat\n\n【聲音】\n{und.AUDIO_TRACK_SILENT}"
    assert und.compose_answer(plain) == "a dog"


@pytest.mark.asyncio
async def test_a_question_is_rewritten_with_the_models_own_system_prompt() -> None:
    client = ScriptedLlmClient('{"question": "What breed is the dog in the photo?"}')
    q, _, how = await und.convert_question("這是什麼品種的狗", client, modality=MediaKind.IMAGE_UNDERSTAND)
    assert how == "understanding-llm"
    assert q == "What breed is the dog in the photo?"
    system, user = client.calls[0]
    assert system is und.REWRITE_SYSTEM[MediaKind.IMAGE_UNDERSTAND]
    assert "answers only in English" in system
    assert "這是什麼品種的狗" in user


@pytest.mark.asyncio
async def test_a_bad_reply_is_retried_then_falls_back_to_the_users_words_labelled() -> None:
    client = ScriptedLlmClient("not json", "still not json")
    q, _, how = await und.convert_question("他在唱什麼", client, modality=MediaKind.AUDIO_UNDERSTAND)
    assert q == "他在唱什麼"
    assert how.startswith("understanding-template (llm failed:")
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_a_second_attempt_that_succeeds_is_a_retry() -> None:
    client = ScriptedLlmClient("garbage", '{"question": "請只用繁體中文說明影片最後發生了什麼"}')
    q, aq, how = await und.convert_question("最後怎麼了", client, modality=MediaKind.VIDEO_UNDERSTAND)
    assert how == "understanding-llm-retry" and q.startswith("請只用繁體中文") and aq == q


@pytest.mark.asyncio
async def test_a_transport_failure_falls_back_rather_than_raising() -> None:
    class _Down:
        async def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
            raise ConnectionError("pod unreachable")

    q, _, how = await und.convert_question("誰在說話", _Down(), modality=MediaKind.AUDIO_UNDERSTAND)
    assert q == "誰在說話" and "ConnectionError" in how


def test_rewriter_system_prompts_name_the_models_language() -> None:
    assert "only in English" in und.REWRITE_SYSTEM[MediaKind.IMAGE_UNDERSTAND]
    for kind in (MediaKind.AUDIO_UNDERSTAND, MediaKind.VIDEO_UNDERSTAND):
        assert "請只用繁體中文" in und.REWRITE_SYSTEM[kind]
    # /說影 hears too (a second model on the track); the rewriter must know.
    assert "listens to\nits audio track" in und.REWRITE_SYSTEM[MediaKind.VIDEO_UNDERSTAND]
    assert "cannot hear" not in und.REWRITE_SYSTEM[MediaKind.VIDEO_UNDERSTAND]
