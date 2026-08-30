"""SPEC.md §7.4's hard checkpoint contract, verified end to end: a real
subprocess running `train.py --resume auto` is really `SIGKILL`ed mid-training
(not simulated), and resume MUST continue from `global_step + 1` with a loss
curve that matches an uninterrupted control run — the "dataloader cursor
wasn't saved" failure mode (§7.4 item 6, D11) makes resume silently re-see
already-consumed examples, which shows up as a step-continuity or loss-curve
break, not an exception.

Per twin/PLAN.md §3.5/§3.8: this test deliberately stays in the flat
`tests/unit/` tree (not a separate `tests/integration/`) and carries no
skippable marker — SPEC §7.4's acceptance bar is binary and MUST run by
default. It is heavier than this project's other unit tests (real subprocess
launches, real (tiny) model construction) by necessity, not oversight.

Uses a fully local, from-scratch tiny Qwen3 model + tokenizer — no network
access, no Hub download. `Qwen3ForCausalLM`/`Qwen3Config` are used (not a
generic architecture) so the LoRA `target_modules="all-linear"` attach path
under test is the real one `twin.train.model.build_lora_config` uses.
`bitsandbytes` 4-bit quantization is CUDA-only, so this test trains with
`use_quantization=False, fp16=False, bf16=False` (`TrainingConfig`'s explicit,
documented CPU-testing escape hatch) — it verifies TRL/Transformers/
Accelerate's own `resume_from_checkpoint` mechanics, which are precision-
agnostic, not the T4-specific fp16+QLoRA numeric path (that needs a separate,
non-CI GPU smoke test — see twin/PLAN.md's open items for Phase 4).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from twin.core.adapter import read_adapter_manifest
from twin.core.encryption import generate_key
from twin.core.enums import ExposureEvidence, GroundTruthSource, NegativeClass, Split
from twin.core.hashing import config_hash, dataset_hash
from twin.core.trajectory import ActionStep, Exposure, Trajectory
from twin.ingest.store import write_trajectories_jsonl
from twin.train import checkpoint as checkpoint_module
from twin.train.formatting import build_sft_dataset
from twin.train.reproducibility import derive_run_id
from twin.train.run import TrainingConfig

REPO_TWIN_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ENTRYPOINT = REPO_TWIN_ROOT / "train.py"

CHECKPOINT_POLL_TIMEOUT_S = 60
SUBPROCESS_TIMEOUT_S = 60
# Empirically calibrated (see twin/PLAN.md Phase 4 notes): an uninterrupted
# control run and a correctly-resumed run differ by ~0.001-0.003 (pure CPU
# cross-process floating-point noise). Deliberately breaking `ignore_data_skip`
# to True (the exact SPEC §7.4 item 6 / D11 failure mode) reliably pushed the
# max per-step diff to ~0.01-0.016 on this same fixture. 0.01 sits between the
# two with comfortable margin on both sides.
MAX_PER_STEP_LOSS_DIFF = 0.01


def _build_toy_model_dir(tmp_path: Path, subdir: str) -> Path:
    """A tiny, fully local, randomly-initialized Qwen3 model + a from-scratch
    word-level tokenizer with a minimal chat template — zero network access.
    `torch.manual_seed(0)` makes two separate calls produce byte-identical
    initial weights, which the control-vs-interrupted comparison relies on.
    """
    model_dir = tmp_path / subdir
    model_dir.mkdir()

    words = [f"tok{i}" for i in range(200)]
    corpus = ["available tools recall web_search reply", " ".join(words)]
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    word_trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=400)
    tokenizer.train_from_iterator(corpus, word_trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )
    # Assistant turns are wrapped in {% generation %}/{% endgeneration %} so
    # `train.run.main`'s `assistant_only_loss=True` (twin/src/twin/train/loss_mask.py)
    # has real generation markers to resolve against — without them,
    # `verify_assistant_masking`'s pre-flight check (and TRL's own internal
    # resolution) would raise for this toy tokenizer, since a from-scratch
    # custom template never matches any of TRL's known patchable templates.
    fast_tokenizer.chat_template = (
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
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    fast_tokenizer.save_pretrained(str(model_dir))

    config = Qwen3Config(
        vocab_size=fast_tokenizer.vocab_size + 10,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=fast_tokenizer.pad_token_id,
        bos_token_id=fast_tokenizer.bos_token_id,
        eos_token_id=fast_tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    Qwen3ForCausalLM(config).save_pretrained(str(model_dir))
    return model_dir


def _build_toy_trajectories(tmp_path: Path, subdir: str, *, n: int = 12) -> str:
    """Content varies per trajectory (not identical placeholder text) so a
    dataloader-cursor bug (re-seeing already-consumed examples) actually
    perturbs the loss curve instead of being masked by every batch looking
    the same."""
    trajectories = [
        Trajectory(
            principal_id="default",
            context_time="2026-01-01T00:00:00Z",
            split=Split.TRAIN,
            exposure=Exposure(occurred=True, stimulus="msg", evidence=ExposureEvidence.READ_RECEIPT),
            observation=" ".join(f"tok{(i * 7 + j) % 200}" for j in range(6)),
            available_tools=["recall", "web_search", "reply"],
            steps=[ActionStep(surface="line", content=" ".join(f"tok{(i * 13 + j) % 200}" for j in range(6)))],
            negative_class=NegativeClass.NONE,
            ground_truth_source=GroundTruthSource.OBSERVED,
        )
        for i in range(n)
    ]
    uri = f"file://{tmp_path / subdir}/trajectories.jsonl"
    write_trajectories_jsonl(trajectories, uri)
    return uri


def _toy_config(base_model_id: str) -> TrainingConfig:
    return TrainingConfig(
        base_model_id=base_model_id,
        base_model_revision="main",  # meaningless for a local path, harmless
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        use_quantization=False,  # bitsandbytes 4-bit needs CUDA; see module docstring
        fp16=False,
        bf16=False,
        learning_rate=1e-3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=8,
        seed=123,
    )


def _run_train_subprocess(
    *,
    config_path: Path,
    trajectories_uri: str,
    checkpoint_store_uri: str,
    principal_id: str,
    encryption_key: bytes,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["TWIN_TRAJECTORY_STORE_URI"] = trajectories_uri
    env["TWIN_CHECKPOINT_STORE_URI"] = checkpoint_store_uri
    env["TWIN_PRINCIPAL_ID"] = principal_id
    env["TWIN_ADAPTER_ENCRYPTION_KEY"] = encryption_key.decode("utf-8")
    cmd = [
        sys.executable,
        str(TRAIN_ENTRYPOINT),
        "--allow-no-self-report",  # toy LINE-only set; train/preflight.py's D19 gate is tested on its own
        "--resume",
        "auto",
        "--config",
        str(config_path),
        "--checkpoint-interval-seconds",
        "0",  # save on (almost) every step so the test doesn't wait minutes for one
    ]
    return subprocess.run(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=SUBPROCESS_TIMEOUT_S
    )


def _log_history_by_step(output_dir: Path) -> dict[int, float]:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text(encoding="utf-8"))
    return {entry["step"]: entry["loss"] for entry in state["log_history"] if "loss" in entry}


def test_kill_and_resume_preserves_step_continuity_and_loss_curve(tmp_path: Path) -> None:
    model_dir = _build_toy_model_dir(tmp_path, "toy_model")
    trajectories_uri = _build_toy_trajectories(tmp_path, "data")
    config = _toy_config(str(model_dir))
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    checkpoint_store_uri = f"file://{tmp_path / 'checkpoints'}"

    # Predict the deterministic run_root_uri the same way twin.train.run.main
    # computes it, so the test can poll it without parsing subprocess output.
    _, trajectory_ids = build_sft_dataset(trajectories_uri, seed=config.seed)
    run_id = derive_run_id(
        seed=config.seed,
        dataset_hash=dataset_hash(trajectory_ids),
        config_hash=config_hash(config.config_hash_fields()),
    )
    run_root_uri = f"{checkpoint_store_uri}/default/{run_id}"
    encryption_key = generate_key()

    env = dict(os.environ)
    env["TWIN_TRAJECTORY_STORE_URI"] = trajectories_uri
    env["TWIN_CHECKPOINT_STORE_URI"] = checkpoint_store_uri
    env["TWIN_PRINCIPAL_ID"] = "default"
    env["TWIN_ADAPTER_ENCRYPTION_KEY"] = encryption_key.decode("utf-8")
    cmd = [
        sys.executable,
        str(TRAIN_ENTRYPOINT),
        "--allow-no-self-report",  # toy LINE-only set; train/preflight.py's D19 gate is tested on its own
        "--resume",
        "auto",
        "--config",
        str(config_path),
        "--checkpoint-interval-seconds",
        "0",
    ]
    run_cwd = tmp_path / "run"
    run_cwd.mkdir()

    process = subprocess.Popen(cmd, cwd=run_cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + CHECKPOINT_POLL_TIMEOUT_S
        checkpoint_uri = None
        while time.monotonic() < deadline:
            checkpoint_uri = checkpoint_module.find_latest_complete_checkpoint(run_root_uri)
            if checkpoint_uri is not None:
                break
            if process.poll() is not None:
                raise AssertionError(f"train.py exited before producing a checkpoint:\n{process.stdout.read()}")
            time.sleep(0.1)
        if checkpoint_uri is None:
            raise AssertionError("timed out waiting for the first checkpoint to appear")

        step_at_kill = int(checkpoint_uri.rsplit("checkpoint-", 1)[-1])
        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    resumed = _run_train_subprocess(
        config_path=config_path,
        trajectories_uri=trajectories_uri,
        checkpoint_store_uri=checkpoint_store_uri,
        principal_id="default",
        encryption_key=encryption_key,
        cwd=run_cwd,
    )
    assert resumed.returncode == 0, resumed.stdout

    manifest = read_adapter_manifest(f"{run_root_uri}/manifest.json")
    assert manifest.global_step == config.max_steps

    # `PeftModel.save_pretrained` is not fsspec-aware — it would silently
    # write to a bogus local directory literally named after the URI scheme
    # if `run.main()` ever passed it a `file://`/`r2://` string directly
    # again. Verify the final adapter is genuinely readable back from
    # `manifest.adapter_uri` (encrypted, per SPEC.md §8), not just that the
    # manifest points somewhere.
    assert checkpoint_module.is_checkpoint_complete(manifest.adapter_uri)
    downloaded_adapter_dir = checkpoint_module.download_checkpoint_for_resume(
        manifest.adapter_uri, str(tmp_path / "downloaded_final_adapter"), encryption_key=encryption_key
    )
    assert (Path(downloaded_adapter_dir) / "adapter_model.safetensors").exists()
    assert (Path(downloaded_adapter_dir) / "adapter_config.json").exists()
    # Final adapter is weights-only fp16 (R2 free-tier budget, 2026-08-29);
    # resume checkpoints stay fp32 — this must never leak into them.
    from safetensors import safe_open

    with safe_open(str(Path(downloaded_adapter_dir) / "adapter_model.safetensors"), framework="pt") as f:
        dtypes = {f.get_tensor(k).dtype for k in f.keys()}  # noqa: SIM118 -- safe_open is not iterable
    assert dtypes == {torch.float16}, dtypes
    assert not (Path(downloaded_adapter_dir) / "optimizer.pt").exists()

    resumed_by_step = _log_history_by_step(run_cwd / ".twin_train_scratch" / "output")
    post_resume_steps = sorted(step for step in resumed_by_step if step > step_at_kill)
    assert post_resume_steps, "no logged steps after the kill point"
    assert post_resume_steps[0] == step_at_kill + 1, (
        f"resume should continue at step {step_at_kill + 1}, first post-resume step logged was "
        f"{post_resume_steps[0]} — a dataloader cursor that wasn't restored shows up here."
    )

    # Control: an uninterrupted run against a byte-identical fresh model +
    # dataset, same seed/config. Compares against the resumed run's full
    # per-step loss curve, not just the final aggregate — a broken
    # `ignore_data_skip` re-consumes already-seen batches, which perturbs
    # individual post-resume steps far more visibly than the run's overall
    # final loss (verified empirically, see MAX_PER_STEP_LOSS_DIFF's comment).
    control_model_dir = _build_toy_model_dir(tmp_path, "control_toy_model")
    control_trajectories_uri = _build_toy_trajectories(tmp_path, "control_data")
    control_config = config.model_copy(update={"base_model_id": str(control_model_dir)})
    control_config_path = tmp_path / "control_config.json"
    control_config_path.write_text(control_config.model_dump_json(), encoding="utf-8")
    control_cwd = tmp_path / "control_run"
    control_cwd.mkdir()

    control = _run_train_subprocess(
        config_path=control_config_path,
        trajectories_uri=control_trajectories_uri,
        checkpoint_store_uri=f"file://{tmp_path / 'control_checkpoints'}",
        principal_id="control",
        encryption_key=generate_key(),
        cwd=control_cwd,
    )
    assert control.returncode == 0, control.stdout

    control_by_step = _log_history_by_step(control_cwd / ".twin_train_scratch" / "output")
    common_steps = sorted(set(control_by_step) & set(resumed_by_step))
    assert common_steps == list(range(1, config.max_steps + 1))

    per_step_diffs = {step: abs(control_by_step[step] - resumed_by_step[step]) for step in common_steps}
    max_diff = max(per_step_diffs.values())
    assert max_diff <= MAX_PER_STEP_LOSS_DIFF, (
        f"interrupted+resumed run's loss curve diverged from the uninterrupted control run "
        f"by {max_diff:.5f} (threshold {MAX_PER_STEP_LOSS_DIFF}) — per-step diffs: {per_step_diffs}. "
        "This is the RNG-state / dataloader-cursor 'fake convergence' failure mode SPEC.md §7.4/D11 "
        "exists to catch, not measurement noise (see this module's calibration notes)."
    )
