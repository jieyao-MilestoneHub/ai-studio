"""train.loss_mask: assistant-only masking, verified against real chat-
template resolution behavior (not mocked) — see the module's docstring for
why this exists (TRL 1.12's `assistant_only_loss` resolves the chat template
by exact string match, and only fails loudly once `SFTTrainer.__init__` runs)."""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from twin.train.loss_mask import resolve_training_chat_template, verify_assistant_masking


def _toy_tokenizer(chat_template: str) -> PreTrainedTokenizerFast:
    words = [f"tok{i}" for i in range(50)]
    corpus = ["available tools recall web_search reply call none", " ".join(words)]
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=200)
    tokenizer.train_from_iterator(corpus, trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )
    fast_tokenizer.chat_template = chat_template
    return fast_tokenizer


# Mirrors the toy chat template in test_train_checkpoint_kill_resume.py:
# assistant turns wrapped in {% generation %}, everything else rendered
# plainly — the minimal shape `assistant_only_loss=True` needs.
GENERATION_TAGGED_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "assistant: {% generation %}"
    "{% if message['tool_calls'] %}call {{ message['tool_calls'][0]['function']['name'] }}"
    "{% else %}{{ message['content'] if message['content'] else 'none' }}{% endif %}"
    "{% endgeneration %}\n"
    "{% else %}"
    "{{ message['role'] }}: {{ message['content'] if message['content'] else '' }}\n"
    "{% endif %}"
    "{% endfor %}"
)

NO_GENERATION_MARKERS_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] if message['content'] else '' }}\n{% endfor %}"
)

NO_ASSISTANT_TURN_MESSAGES = [
    {"role": "system", "content": "available tools: recall, web_search, reply"},
    {"role": "user", "content": "tok1 tok2 tok3"},
]

TOOL_CALL_MESSAGES = [
    {"role": "system", "content": "available tools: recall, web_search, reply"},
    {"role": "user", "content": "tok1 tok2 tok3"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"type": "function", "function": {"name": "reply", "arguments": "{}"}}],
    },
    {"role": "tool", "name": "reply", "content": "digest0000"},
]

NO_ACTION_MESSAGES = [
    {"role": "system", "content": "available tools: recall, web_search, reply"},
    {"role": "user", "content": "tok1 tok2 tok3"},
    {"role": "assistant", "content": "", "tool_calls": []},
]


def test_resolve_training_chat_template_is_noop_when_markers_already_present() -> None:
    tokenizer = _toy_tokenizer(GENERATION_TAGGED_TEMPLATE)
    assert resolve_training_chat_template(tokenizer) is None


def test_verify_assistant_masking_isolates_only_the_assistant_tool_call_span() -> None:
    tokenizer = _toy_tokenizer(GENERATION_TAGGED_TEMPLATE)
    report = verify_assistant_masking(tokenizer, TOOL_CALL_MESSAGES)

    assert 0 < report.assistant_tokens < report.total_tokens
    # The masked span is the tool-call rendering ("call reply"), not the
    # user's observation or the tool result digest.
    assert "tok1" not in report.assistant_text
    assert "digest0000" not in report.assistant_text


def test_verify_assistant_masking_handles_the_no_action_turn() -> None:
    tokenizer = _toy_tokenizer(GENERATION_TAGGED_TEMPLATE)
    report = verify_assistant_masking(tokenizer, NO_ACTION_MESSAGES)

    # NoActionStep formats to an empty-content assistant turn; the template's
    # else-branch renders the literal "none" placeholder inside the
    # generation block, so at least one real token is still masked — an
    # empty-but-present assistant turn is not the same as no assistant turn.
    assert report.assistant_tokens > 0


def test_verify_assistant_masking_raises_when_template_has_no_generation_markers() -> None:
    tokenizer = _toy_tokenizer(NO_GENERATION_MARKERS_TEMPLATE)
    with pytest.raises(ValueError):
        verify_assistant_masking(tokenizer, TOOL_CALL_MESSAGES)


def test_verify_assistant_masking_raises_when_no_assistant_turn_is_present() -> None:
    # Exercises verify_assistant_masking's own zero-token guard, distinct
    # from the "no generation markers at all" case above (which fails
    # earlier, inside TRL's own template resolution) — every real trajectory
    # from formatting.trajectory_to_messages has at least one assistant
    # turn, so this input shape should never reach a real training run.
    tokenizer = _toy_tokenizer(GENERATION_TAGGED_TEMPLATE)
    with pytest.raises(ValueError, match="zero assistant-masked tokens"):
        verify_assistant_masking(tokenizer, NO_ASSISTANT_TURN_MESSAGES)
