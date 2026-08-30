"""Orchestration: data -> model/LoRA -> SFTConfig -> SFTTrainer -> resume ->
train() -> AdapterManifest. SPEC.md §5.3, §7.4-§7.6.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import fsspec
import torch
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, model_validator
from trl import SFTConfig, SFTTrainer

from twin.core.adapter import AdapterManifest, ModelSpec, write_adapter_manifest
from twin.core.hashing import config_hash, dataset_hash
from twin.train import checkpoint
from twin.train import model as train_model
from twin.train.formatting import build_sft_dataset
from twin.train.loss_mask import verify_assistant_masking
from twin.train.reproducibility import derive_run_id, seed_everything

# LoRA-specific ceiling on effective batch size (see TrainingConfig's
# `gradient_accumulation_steps` field comment below). Enforced by
# `TrainingConfig._check_effective_batch_ceiling` instead of staying a
# comment nobody re-reads — inspired by llm-twin's
# `twin_train/common.py:guard_config()` pattern (see
# twin/docs/llm-twin-reference-notes.md).
MAX_EFFECTIVE_BATCH_SIZE = 32


class TrainingConfig(BaseModel):
    """Every field here MUST be exactly and only what materially changes
    training semantics/output — this whole model (minus `seed`, bound
    separately per SPEC.md §7.5) is `config_hash()`'s input. Storage
    locations (`trajectories_uri`, `checkpoint_store_uri`, passed separately
    to `main()`) are deliberately NOT fields here: changing where checkpoints
    live is an operational choice, not a training-semantics change, and MUST
    NOT perturb `run_id`."""

    model_config = ConfigDict(frozen=True)

    base_model_id: str = train_model.DEFAULT_BASE_MODEL_ID
    base_model_revision: str = train_model.DEFAULT_BASE_MODEL_REVISION
    lora_rank: int  # No default on purpose — SPEC.md §11 item G requires an explicit, recorded choice.
    lora_alpha: int = train_model.LORA_ALPHA_DEFAULT
    lora_dropout: float = 0.0
    # T4/QLoRA production defaults (see train/model.py's docstrings for why).
    # bitsandbytes 4-bit quantization needs CUDA — tests/unit/test_train_checkpoint_kill_resume.py's
    # toy CPU run is the one legitimate reason to flip use_quantization=False and
    # fp16=False; that's an explicit, recorded config difference (and correctly
    # hashes differently via config_hash_fields()), not a silent fallback.
    use_quantization: bool = True
    fp16: bool = True
    bf16: bool = False
    learning_rate: float  # No default on purpose — needs a real sweep, see train/model.py's docstring.
    per_device_train_batch_size: int
    gradient_accumulation_steps: int  # Keep per_device_train_batch_size * this < ~32 (LoRA-specific ceiling).
    max_steps: int  # Not num_train_epochs: fixes an exact step count so the kill/resume
    # test can compare an interrupted run against an uninterrupted control run 1:1.
    # How many times each self-report (interview) trajectory is repeated in the
    # SFT set. 1 = no upsampling. The interview yields ~30 trajectories against
    # ~15.7k LINE ones (0.2%): at 1 the adapter sees them as noise, yet D19
    # names self-report the dominant persona-fidelity source. Repetition is the
    # only lever that keeps every LINE sample (no downsampling of behaviour
    # data) — its cost is verbatim memorisation of interview phrasing, which
    # EVAL.md S4 would surface as "more articulate than the principal".
    # Part of config_hash (it changes what the model is trained on).
    self_report_upsample: int = 1
    seed: int = 42

    def config_hash_fields(self) -> dict[str, Any]:
        return self.model_dump(exclude={"seed"})

    @model_validator(mode="after")
    def _check_effective_batch_ceiling(self) -> TrainingConfig:
        effective = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if effective > MAX_EFFECTIVE_BATCH_SIZE:
            raise ValueError(
                f"per_device_train_batch_size ({self.per_device_train_batch_size}) * "
                f"gradient_accumulation_steps ({self.gradient_accumulation_steps}) = {effective} exceeds the "
                f"LoRA-specific ceiling of {MAX_EFFECTIVE_BATCH_SIZE} documented above. This was previously "
                "only a comment; enforcing it turns a config typo into a startup failure instead of a "
                "silently-suboptimal run."
            )
        return self


def main(
    config: TrainingConfig,
    *,
    principal_id: str,
    trajectories_uri: str,
    checkpoint_store_uri: str,
    encryption_key: bytes,  # SPEC.md §8: adapter weights MUST be stored encrypted — see core.encryption.
    resume: bool = True,
    checkpoint_interval_seconds: float = 600.0,  # SPEC.md §7.4 SHOULD 10-15 min (600-900s).
    # Deliberately NOT a TrainingConfig field: cadence is operational (how often
    # we checkpoint), not a training-semantics change (what the model converges
    # to) — same reasoning TrainingConfig's own docstring gives for excluding
    # storage URIs, so it stays out of config_hash(). tests/unit/test_train_checkpoint_kill_resume.py
    # passes a small value here so a toy CI run doesn't have to wait 10 minutes
    # for its first checkpoint.
) -> AdapterManifest:
    dataset, trajectory_ids = build_sft_dataset(
        trajectories_uri, seed=config.seed, self_report_upsample=config.self_report_upsample
    )
    dataset_hash_value = dataset_hash(trajectory_ids)
    config_hash_value = config_hash(config.config_hash_fields())
    run_id = derive_run_id(seed=config.seed, dataset_hash=dataset_hash_value, config_hash=config_hash_value)
    run_root_uri = f"{checkpoint_store_uri.rstrip('/')}/{principal_id}/{run_id}"

    local_checkpoint: str | None = None
    if resume:
        latest = checkpoint.find_latest_complete_checkpoint(run_root_uri)
        if latest is not None:
            local_checkpoint = checkpoint.download_checkpoint_for_resume(
                latest, local_scratch_dir="./.twin_train_scratch", encryption_key=encryption_key
            )

    seed_everything(config.seed)

    quantization_config = train_model.build_quantization_config() if config.use_quantization else None
    lora_config = train_model.build_lora_config(
        rank=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout
    )
    model_spec = ModelSpec(base_model_id=config.base_model_id, base_model_revision=config.base_model_revision)
    tokenizer = train_model.load_tokenizer(model_spec)

    # Cheap, tokenizer-only pre-flight check for `assistant_only_loss` below —
    # runs before the quantized model is even constructed, let alone loaded
    # onto a GPU. See train/loss_mask.py's module docstring: TRL 1.12
    # resolves the chat template needed for assistant-only masking by exact
    # string match against a template baked into that TRL release, and only
    # fails loudly once SFTTrainer.__init__ runs. This surfaces the same
    # failure earlier. Checks every example, not just one — a shape-specific
    # mismatch (e.g. one negative-class variant) could otherwise sit past the
    # first example undetected. Printed rather than only returned: this
    # trainer runs unattended on Modal/Kaggle/Lightning, so the decoded
    # eyeball sample only has value if it actually lands in the captured job
    # log, not just in a value nothing reads.
    mask_reports = []
    for index, example in enumerate(dataset):
        try:
            mask_reports.append(verify_assistant_masking(tokenizer, example["messages"]))
        except ValueError as exc:
            raise ValueError(
                f"loss-mask pre-flight failed on trajectory index {index} "
                f"(trajectory_id={example['trajectory_id']}): {exc}"
            ) from exc
    print(
        f"[loss-mask pre-flight] {len(mask_reports)} example(s) verified; "
        f"assistant-token fraction min={min(r.assistant_fraction for r in mask_reports):.3f} "
        f"max={max(r.assistant_fraction for r in mask_reports):.3f}; "
        f"sample decoded assistant span: {mask_reports[0].assistant_text!r}"
    )

    sft_config = SFTConfig(
        output_dir="./.twin_train_scratch/output",
        fp16=config.fp16,
        bf16=config.bf16,  # Production default bf16=False MUST NOT change for a T4 run —
        # TRL's own SFTConfig default (bf16=True when fp16 isn't set) crashes outright
        # on a T4, which has no Ampere+ bf16 tensor cores.
        assistant_only_loss=True,  # Not optional: without this, loss is computed over
        # every token (system/user/tool turns included, per TRL's default), so the
        # model spends capacity learning to predict the *other party's* turns and the
        # injected tool-name vocabulary instead of the principal's own replies —
        # trains without error, loss still drops, converges to the wrong objective.
        # `verify_assistant_masking` above is this same codebase's pre-flight check
        # that the chat-template resolution this flag depends on actually holds for
        # the pinned base model (train/loss_mask.py).
        gradient_checkpointing=True,  # Memory-only (recompute activations), numerically
        # identical training — hence not a TrainingConfig/config_hash field. Required for
        # an 8B QLoRA step to fit a 16 GB T4 at all (rank probe, Modal, 2026-08-29).
        gradient_checkpointing_kwargs={"use_reentrant": False},  # PEFT + checkpointing
        # with the reentrant impl silently yields no grads for the adapter.
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        seed=config.seed,
        logging_steps=1,  # Free-tier runs are short enough per session that per-step
        # visibility matters more than log noise; also gives tests/unit/test_train_checkpoint_kill_resume.py
        # a per-step log_history to assert continuity against.
        save_strategy="steps",
        save_steps=10**9,  # Effectively "never on step count alone" — real cadence
        # comes from R2CheckpointCallback's wall-clock timer instead (SPEC.md §7.4 SHOULD).
        save_total_limit=3,  # Bound ephemeral local disk; full history lives in R2.
        ignore_data_skip=False,  # MUST NOT be flipped to True — this is the actual
        # switch that makes resume seek the dataloader cursor instead of silently
        # re-iterating already-seen data (SPEC.md §7.4 item 6 / D11's failure mode).
    )

    trainer = SFTTrainer(
        model=config.base_model_id,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        quantization_config=quantization_config,
        processing_class=tokenizer,
        callbacks=[
            checkpoint.R2CheckpointCallback(
                remote_run_root_uri=run_root_uri,
                encryption_key=encryption_key,
                interval_seconds=checkpoint_interval_seconds,
            )
        ],
    )

    if config.fp16:
        # fp16 AMP keeps fp32 master weights and unscales fp32 grads; the LoRA
        # params are otherwise created in the base checkpoint's dtype (Qwen3-8B:
        # bf16), and the first real T4 run (Modal, 2026-08-29) died at step 0 with
        # "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
        # 'BFloat16'". Upcasting only the trainable params is the standard QLoRA
        # fix (what peft.prepare_model_for_kbit_training does) — memory cost is a
        # few hundred MiB at r=64, well inside the probe's headroom.
        assert trainer.model is not None
        for param in trainer.model.parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()

    trainer.train(resume_from_checkpoint=local_checkpoint)

    final_adapter_uri = f"{run_root_uri}/final"
    assert isinstance(trainer.model, PeftModel), "SFTTrainer always wraps model in a PeftModel when peft_config is set"
    # save_pretrained is not fsspec-aware — it writes to a literal local path,
    # not a URI (a `file://`/`r2://`-prefixed string would silently create a
    # bogus local directory named after the scheme rather than resolving it).
    # Save locally first, then upload+encrypt via the same mechanism
    # checkpoints use (SPEC.md §8).
    local_final_adapter_dir = "./.twin_train_scratch/final"
    # Weights only, fp16: `save_pretrained` on a PeftModel writes just the
    # adapter tensors + adapter_config.json (no optimizer/scheduler/RNG — those
    # live in the resume checkpoints, which are pruned separately). Halving to
    # fp16 is lossless for inference on the fp16 T4 target and keeps the
    # artifact ~0.35 GB at r=64 instead of ~0.7 GB (R2 free tier, 2026-08-29).
    # Done AFTER training and after the last checkpoint, so nothing that
    # resumes ever sees the downcast.
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)
    trainer.model.save_pretrained(local_final_adapter_dir)
    checkpoint.upload_adapter(local_final_adapter_dir, final_adapter_uri, encryption_key=encryption_key)

    manifest = AdapterManifest(
        run_id=run_id,
        principal_id=principal_id,
        adapter_uri=final_adapter_uri,
        model_spec=model_spec,
        seed=config.seed,
        dataset_hash=dataset_hash_value,
        config_hash=config_hash_value,
        global_step=trainer.state.global_step,
        created_at=datetime.now(UTC),
    )
    write_adapter_manifest(manifest, f"{run_root_uri}/manifest.json")
    # The loss curve is the only evidence a run converged; the first real run's
    # (run_e6a366ee73958e69) survived nowhere but Modal's app log. Plain JSON,
    # not encrypted: per-step loss/lr/grad_norm reveal nothing about the
    # principal (same reasoning as the unencrypted AdapterManifest).
    with fsspec.open(f"{run_root_uri}/log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False)
    return manifest
