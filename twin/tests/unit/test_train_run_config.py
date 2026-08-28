"""TrainingConfig's effective-batch-size ceiling. The field comment on
`gradient_accumulation_steps` documented this LoRA-specific ceiling (~32)
without enforcing it; this turns that comment into a raised error — see
twin/docs/llm-twin-reference-notes.md for why (inspired by llm-twin's
`twin_train/common.py:guard_config()` pattern)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from twin.train.run import MAX_EFFECTIVE_BATCH_SIZE, TrainingConfig


def _config(*, per_device_train_batch_size: int, gradient_accumulation_steps: int) -> TrainingConfig:
    return TrainingConfig(
        lora_rank=8,
        learning_rate=2e-4,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=10,
    )


def test_effective_batch_size_at_ceiling_is_accepted() -> None:
    per_device = 4
    accumulation = MAX_EFFECTIVE_BATCH_SIZE // per_device
    config = _config(per_device_train_batch_size=per_device, gradient_accumulation_steps=accumulation)
    assert config.per_device_train_batch_size * config.gradient_accumulation_steps == MAX_EFFECTIVE_BATCH_SIZE


def test_effective_batch_size_over_ceiling_is_rejected() -> None:
    per_device = 4
    accumulation = MAX_EFFECTIVE_BATCH_SIZE // per_device + 1
    with pytest.raises(ValidationError, match="exceeds the"):
        _config(per_device_train_batch_size=per_device, gradient_accumulation_steps=accumulation)
