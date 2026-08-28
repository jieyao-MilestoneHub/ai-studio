#!/usr/bin/env python3
"""A real `harness.baseline.InferenceBackend` binding — bare, un-fine-tuned
`twin.train.model.DEFAULT_MODEL_SPEC` weights, no LoRA adapter. Same
deferred-hardware status as `examples/probe_lora_rank.py`: written now, not
run against real hardware in this pass. B0/B1/B2 (EVAL.md §3.4) all run the
same unmodified base model — only the prompt differs, which is
`harness.baseline`'s job, not this file's.

Usage (on a real GPU, once real S1 item-bank/persona/transcript data exist —
see examples/prepare_s1_eval_round.py):

    uv run python examples/run_baseline_inference.py
"""

from __future__ import annotations

import sys

import torch

from twin.train.model import DEFAULT_MODEL_SPEC, load_tokenizer


class HFBaselineBackend:
    """Loads the bare base model once, reuses it for every `complete()`
    call. No quantization/LoRA config here — unlike `probe_lora_rank.py` and
    `train.model.build_quantization_config`, which exist specifically for
    QLoRA fine-tuning, a baseline-inference-only load doesn't need bnb at
    all: it's the plain weights, full precision the hardware supports."""

    def __init__(self) -> None:
        from transformers import AutoModelForCausalLM

        self._tokenizer = load_tokenizer(DEFAULT_MODEL_SPEC)
        self._model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_SPEC.base_model_id,
            revision=DEFAULT_MODEL_SPEC.base_model_revision,
            device_map="auto",
        )
        self._model.eval()

    def complete(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("This needs a real GPU (T4 or better) — nothing to run baseline inference on CPU.")

    backend = HFBaselineBackend()
    print(f"Loaded {DEFAULT_MODEL_SPEC.base_model_id}@{DEFAULT_MODEL_SPEC.base_model_revision}.")
    print("Use examples/prepare_s1_eval_round.py to drive it against a real S1 item bank.")
    _ = backend  # constructed to prove the load path works; wiring into a real round is prepare_s1_eval_round.py's job


if __name__ == "__main__":
    main()
