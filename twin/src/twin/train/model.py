"""Base model + QLoRA construction. SPEC.md §5.1 (base model), §5.3 (LoRA/
QLoRA only, no RLHF/DPO — this file only configures adapters, never a
training loop), §7.6 (existing framework — this file hands its output
straight to TRL's SFTTrainer, never implements training itself). SPEC.md §11
item G: the base model size is spec-decided-not-mandated, and any capacity
downgrade (the rank ladder below) MUST be a recorded human decision, never a
silent runtime fallback.
"""

from __future__ import annotations

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase

from twin.core.adapter import ModelSpec

# Selected 2026-08-27 after comparing current (Apache-2.0-or-equivalent,
# Traditional-Chinese-capable, 8B-class) candidates — see twin/PLAN.md Phase 4
# discussion. Revision pinned to the exact commit SHA fetched from the HF Hub
# API on the same date, not "main": a floating ref could change upstream
# without changing this string, silently invalidating SPEC.md §7.5
# reproducibility without tripping config_hash.
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_BASE_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

DEFAULT_MODEL_SPEC = ModelSpec(
    base_model_id=DEFAULT_BASE_MODEL_ID,
    base_model_revision=DEFAULT_BASE_MODEL_REVISION,
)

# TRL's current "LoRA Without Regret" reference config for SFT at
# post-training data scale uses r=256 — but that reference targets A100-class
# hardware. On a 16GB T4 (this project's actual target, SPEC.md §7.3) it is
# unverified. `run.main()` MUST NOT silently retry at a lower rank on OOM —
# that is exactly the silent downgrade SPEC.md §11 item G forbids. Instead,
# `examples/probe_lora_rank.py` smoke-tests each rung on real hardware, a
# human reads the result, and the chosen rank is recorded explicitly as
# `TrainingConfig.lora_rank` — that record IS item G's required decision
# trail.
LORA_RANK_FALLBACK_LADDER: tuple[int, ...] = (256, 128, 64, 32)

# Note alpha < r here, matching TRL's published reference config exactly —
# a real departure from the older "alpha = 2*r" heuristic. Don't re-derive it.
LORA_ALPHA_DEFAULT = 16


def build_quantization_config() -> BitsAndBytesConfig:
    """QLoRA 4-bit config for a T4. `bnb_4bit_compute_dtype` MUST be
    `torch.float16`, not `torch.bfloat16`: T4 (Turing, compute capability 7.5)
    has no Ampere-or-later bf16 tensor cores. This is the QLoRA-side half of
    the same T4 fix `train.run` applies on the `SFTConfig` side
    (`fp16=True, bf16=False`) — TRL's current default (`bf16=True` when `fp16`
    isn't set) crashes outright on this hardware if either half is missed."""
    return BitsAndBytesConfig(  # type: ignore[no-untyped-call]  # transformers ships BitsAndBytesConfig.__init__ without annotations
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def build_lora_config(*, rank: int, lora_alpha: int = LORA_ALPHA_DEFAULT, lora_dropout: float = 0.0) -> LoraConfig:
    """`target_modules="all-linear"` (attention q/k/v/o + MLP gate/up/down),
    per TRL's own current guidance: attention-only LoRA underperforms even at
    matched parameter count. Plain vanilla LoRA on purpose — no rsLoRA, no
    DoRA, despite both being available in the pinned peft version: the
    current highest-authority reference config (TRL's "LoRA Without Regret"
    doc) uses neither."""
    return LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def load_tokenizer(model_spec: ModelSpec) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_spec.base_model_id, revision=model_spec.base_model_revision)
