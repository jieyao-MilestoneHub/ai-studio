#!/usr/bin/env python3
"""The real `harness.baseline.InferenceBackend` binding: Qwen3-8B loaded the
same way training loaded it (4-bit, `twin.train.model.build_quantization_
config`) so B0/B1/B2 and T differ only in prompt/adapter, never in numerics
— and so it fits a 16 GB T4, which the fp16 weights (~16.4 GB) do not.

Prompts go through the chat template with thinking disabled: Qwen3-8B is a
post-trained chat model, and T was SFT'd through that same template
(`train.formatting`); feeding raw text would measure template confusion, not
persona. "空白 prompt" (EVAL.md §3.4 B0) means no persona/context is added,
not that the model's own I/O format is bypassed.

`adapter_dir` (optional) is an already-decrypted local PEFT adapter — see
examples/generate_s1_candidates.py for the R2 download + decrypt step. GPU
only; `examples/generate_s1_candidates.py` is the driver.
"""

from __future__ import annotations

import sys

import torch

from twin.train.model import DEFAULT_MODEL_SPEC, build_quantization_config, load_tokenizer


class HFBaselineBackend:
    def __init__(self, *, adapter_dir: str | None = None, max_new_tokens: int = 256) -> None:
        from transformers import AutoModelForCausalLM

        self._tokenizer = load_tokenizer(DEFAULT_MODEL_SPEC)
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_SPEC.base_model_id,
            revision=DEFAULT_MODEL_SPEC.base_model_revision,
            quantization_config=build_quantization_config(),
            device_map="auto",
        )
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        self._model = model
        self._max_new_tokens = max_new_tokens
        self.model_label = f"{DEFAULT_MODEL_SPEC.base_model_id}@{DEFAULT_MODEL_SPEC.base_model_revision}"

    def complete(self, prompt: str) -> str:
        return self.complete_messages([{"role": "user", "content": prompt}])

    def complete_messages(self, messages: list[dict[str, str]]) -> str:
        """Same generation path, caller-shaped turns — T's S1 answers need the
        training-time shape (system tool list + stimulus, `train.formatting`)."""
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("This needs a real GPU — nothing to run baseline inference on CPU.")
    backend = HFBaselineBackend()
    print(f"Loaded {backend.model_label} (4-bit). Reply: {backend.complete('請用一句話介紹你自己。')!r}")


if __name__ == "__main__":
    main()
