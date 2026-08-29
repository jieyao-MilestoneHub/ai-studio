#!/usr/bin/env python3
"""Manual, GPU-only smoke test for `twin.train.model.LORA_RANK_FALLBACK_LADDER`.
NOT pytest, NOT CI — SPEC.md §11 item G forbids silently downgrading LoRA
capacity for cost/memory reasons; this script exists so a human can pick a
rank on real target hardware and record that choice explicitly, instead of
`twin.train.run` auto-retrying at a lower rank on OOM.

Usage (on the actual T4/target GPU, not CI):

    uv run python examples/probe_lora_rank.py

For each rank in the ladder (highest first — TRL's current reference config
for SFT at post-training data scale is r=256, but that reference targets
A100-class hardware), attempts to build the quantized model + LoRA adapter
and run one forward+backward pass with a short synthetic batch, reporting
peak CUDA memory and whether it succeeded. Does not train anything and does
not touch real trajectory data — this measures hardware headroom only.

Read the output, then set `TrainingConfig.lora_rank` in the real training
config to the highest rank that succeeded with comfortable headroom. That
choice, made here by a human reading real numbers, IS the SPEC.md §11 item G
decision trail — this script's output is meant to be recorded (e.g. in a
commit message or twin/PLAN.md), not silently acted on by code.
"""

from __future__ import annotations

import sys

import torch

from twin.train.model import (
    DEFAULT_MODEL_SPEC,
    LORA_RANK_FALLBACK_LADDER,
    build_lora_config,
    build_quantization_config,
)


def probe_rank(rank: int) -> tuple[bool, float]:
    """Returns (succeeded, peak_memory_gib). Reloads the model fresh per rank
    — this is a one-shot manual script, not a hot loop; correctness over
    speed. Runs one real optimizer step (AdamW keeps two fp32 moments per
    trainable param, which at r=256 all-linear on an 8B model is several GiB
    on its own) so a pass here means a training step fits, not just a
    forward/backward. Frees everything in `finally`: the first real run
    (2026-08-29, Modal T4) leaked the OOM'd model into the next rung, so all
    four rungs reported the same ~14.3 GiB peak."""
    import gc

    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = None
    optimizer = None
    output = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_SPEC.base_model_id,
            revision=DEFAULT_MODEL_SPEC.base_model_revision,
            quantization_config=build_quantization_config(),
            device_map="auto",
        )
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, build_lora_config(rank=rank))
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

        batch = torch.randint(0, model.config.vocab_size, (1, 512), device=model.device)
        output = model(input_ids=batch, labels=batch)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        return True, torch.cuda.max_memory_allocated() / (1024**3)
    except torch.cuda.OutOfMemoryError:
        return False, torch.cuda.max_memory_allocated() / (1024**3)
    finally:
        del output, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("This probe needs a real GPU (T4 or better) — nothing to measure on CPU.")

    print(f"Probing {DEFAULT_MODEL_SPEC.base_model_id}@{DEFAULT_MODEL_SPEC.base_model_revision} "
          f"on {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB)")
    for rank in LORA_RANK_FALLBACK_LADDER:
        succeeded, peak_gib = probe_rank(rank)
        status = "OK" if succeeded else "OOM"
        print(f"  r={rank:>4}: {status} (peak {peak_gib:.2f} GiB)")


if __name__ == "__main__":
    main()
