"""Assistant-only loss masking, verified — not a SPEC.md-mandated rule (no
section of SPEC.md governs chat-template masking resolution specifically;
§11 item G is about the base-model-capacity decision, not this), but the
same "explicit and recorded, not a silently-trusted default" discipline
this codebase applies elsewhere (e.g. `train/model.py`'s
`LORA_RANK_FALLBACK_LADDER`, `TrainingConfig`'s no-default fields).

`train.run.main` sets `SFTConfig(assistant_only_loss=True)`; internally TRL
1.12 resolves the tokenizer's chat template to one of its own
generation-tagged patch templates (`trl.chat_template_utils.
get_training_chat_template`) by *exact string equality* against a template
baked into that TRL release. Verified live against the real pinned
`Qwen/Qwen3-8B` tokenizer (`train.model.DEFAULT_BASE_MODEL_REVISION`) with
trl==1.12.0: the match holds today. It is still a brittle mechanism — a
future TRL upgrade, or a change to the pinned base model/revision, can
silently break that exact-string match. TRL itself fails loudly when no
patch matches and the tokenizer's own template also lacks
`{% generation %}` markers (`get_training_chat_template` raises
`ValueError`), but only once `SFTTrainer.__init__` runs — i.e. after the
quantized model has already been constructed. This module exists to run
the same resolution as an explicit, cheap, tokenizer-only pre-flight step
in `train.run.main`, before any model/GPU cost is paid.

Adapted from `twin/reference/llm-twin`'s
`packages/twin-train/src/twin_train/common.py`
(`RESPONSE_MARKERS`/`detect_markers`/`verify_masking`) — but built on
transformers' native `return_assistant_tokens_mask` mechanism rather than
hardcoded chat-template marker strings: twin's pinned `trl>=1.12`/
`transformers>=5.16` support it directly, and it is authoritative where
llm-twin's regex-marker approach (written against older library versions)
is a heuristic. See twin/docs/llm-twin-reference-notes.md for the full
compatibility write-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from transformers import PreTrainedTokenizerBase
from trl.chat_template_utils import get_training_chat_template, has_generation_markers


@dataclass(frozen=True)
class MaskReport:
    """`assistant_text` is the human-eyeball equivalent of llm-twin's
    `verify_masking()` print step — read it once per new trajectory shape,
    the same way that project's Colab notebook insists on reading a dry-run
    before ever starting a real training run."""

    total_tokens: int
    assistant_tokens: int
    assistant_text: str

    @property
    def assistant_fraction(self) -> float:
        return self.assistant_tokens / self.total_tokens if self.total_tokens else 0.0


def resolve_training_chat_template(tokenizer: PreTrainedTokenizerBase) -> str | None:
    """The chat template `assistant_only_loss=True` needs. `None` means the
    tokenizer's own template already qualifies (has `{% generation %}`
    markers). Raises `ValueError` (propagated from TRL) if the tokenizer's
    template neither already qualifies nor matches one of TRL's known
    patchable templates — this MUST surface as an explicit failure, not a
    silently-wrong mask."""
    if has_generation_markers(tokenizer.chat_template):
        return None
    return get_training_chat_template(tokenizer)


def verify_assistant_masking(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, Any]]) -> MaskReport:
    """Renders `messages` through the resolved training chat template and
    reports exactly which tokens would receive loss under
    `assistant_only_loss=True` — the automated, tokenizer-only, no-GPU-
    needed equivalent of llm-twin's manual "read the printed masked tokens"
    dry-run step.

    Raises if the resulting mask is empty: every trajectory
    `train.formatting.trajectory_to_messages` produces has at least one
    assistant turn (a tool call, or the empty no-action turn — never zero
    turns), so an all-empty mask means the template/message shape has
    silently stopped matching, not that the trajectory was legitimately
    action-free.
    """
    chat_template = resolve_training_chat_template(tokenizer)
    # `apply_chat_template`'s return type is overloaded on `tokenize`/`return_dict`;
    # with both True (as here) it's always a dict of encoded fields (`BatchEncoding`),
    # never the `str`/`list[str]` variants mypy's overload resolution otherwise offers.
    encoded = cast(
        "dict[str, Any]",
        tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        ),
    )
    mask: list[int] = encoded["assistant_masks"]
    ids: list[int] = encoded["input_ids"]
    assistant_token_count = sum(mask)
    if assistant_token_count == 0:
        raise ValueError(
            "verify_assistant_masking: resolved chat template produced zero "
            "assistant-masked tokens for a non-empty message list — "
            "assistant_only_loss would silently train on zero examples. "
            "This means the chat-template resolution above has stopped "
            "matching this message shape; do not proceed to a real run."
        )
    # decode()'s return type is overloaded on whether its input is batched;
    # a single flat list of ints (as built here) always yields a single str.
    assistant_text = cast(
        str, tokenizer.decode([token_id for token_id, keep in zip(ids, mask, strict=True) if keep])
    )
    return MaskReport(total_tokens=len(ids), assistant_tokens=assistant_token_count, assistant_text=assistant_text)
