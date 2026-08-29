"""Checkpoint save-sync-resume. SPEC.md §7.4 (the six-artifact contract, the
`kill -9`/resume CI gate), §7.2 (fsspec URIs only, R2 as the cross-cloud
hub), §8 (adapter weights are personal data — MUST be encrypted at rest, see
`core.encryption`). Modal/Kaggle local disks are ephemeral — especially
under spot/preemptible interruption, and Kaggle sessions additionally carry
a hard wall-clock cutoff independent of preemption — so every checkpoint
Trainer writes locally MUST be synced out before it can be relied on for
`--resume auto` on a fresh machine.

HF Trainer/Accelerate (current pinned versions) already restore five of the
six SPEC.md §7.4 items via `resume_from_checkpoint` — adapter weights,
optimizer state, LR scheduler state, RNG state, `global_step` — provided
`ignore_data_skip=False` (the default; `twin.train.run` MUST NOT flip it).
The sixth item, the dataloader sample cursor, is only cheaply and correctly
restored for a map-style dataset (see `twin.train.formatting.build_sft_dataset`,
which is why it insists on one). This module's actual job is narrower than
"implement checkpointing" — it's "make sure a checkpoint Trainer already
wrote correctly survives an encrypted trip through ephemeral remote storage".

Every checkpoint (and the final adapter) is uploaded as a single encrypted
tar archive, not as individual plaintext files — this is what SPEC.md §8's
"MUST 加密儲存" actually requires, and as a side effect a remote directory
listing no longer leaks anything about internal structure (filenames,
sizes per artifact) either.
"""

from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass, field

import fsspec
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from twin.core import encryption

# SPEC.md §7.4 items 1-5: the on-disk artifact names transformers'
# Trainer.save_state / PEFT's save_pretrained currently write for a PEFT
# (LoRA) model *checkpoint* (not the bare final adapter save — see
# ADAPTER_SAVE_ARTIFACTS below for that narrower case). Confirmed 2026-08-27
# against a real local run of the pinned transformers/accelerate/peft
# versions, not guessed.
REQUIRED_CHECKPOINT_ARTIFACTS: tuple[str, ...] = (
    "adapter_model.safetensors",  # 1. adapter weights
    "adapter_config.json",
    "optimizer.pt",  # 2. optimizer state
    "scheduler.pt",  # 3. LR scheduler state
    "rng_state.pth",  # 4. RNG state
    "trainer_state.json",  # 5. global_step (and dataloader-skip bookkeeping)
)

# What a bare `PeftModel.save_pretrained(...)` writes (twin.train.run's final
# adapter save) — no optimizer/scheduler/RNG/trainer-state, those are
# Trainer-checkpoint-specific artifacts a plain save doesn't produce.
ADAPTER_SAVE_ARTIFACTS: tuple[str, ...] = ("adapter_model.safetensors", "adapter_config.json")

# The encrypted blob every checkpoint/adapter upload actually consists of —
# one file, not the artifact list above; those names describe what's inside
# the tar, not what appears in a remote directory listing.
ARCHIVE_FILENAME = "adapter_weights.tar.enc"

# Written last, after the archive has finished uploading. fsspec puts to an
# object store are not atomic — a kill mid-upload could otherwise leave a
# directory containing a torn, half-written archive that looks present at a
# glance. Checking for this marker, not just the archive's existence, is
# this module's own addition on top of SPEC's literal text, in service of
# what the six-item contract actually intends.
COMPLETION_MARKER = "_TWIN_CHECKPOINT_COMPLETE"

# Remote retention. A resume-capable checkpoint (SPEC.md §7.4: adapter +
# optimizer + scheduler + RNG + state) is ~2.6 GB at r=64 on Qwen3-8B, and
# R2's free tier is 10 GB-month — the first real run reached 17 GB after six
# uploads (2026-08-29). Two are kept, not one: a kill *during* an upload
# leaves the newest directory without its COMPLETION_MARKER, and the one
# before it is then the only resumable state.
REMOTE_CHECKPOINTS_TO_KEEP = 2


def _fs_and_path(uri: str) -> tuple[fsspec.AbstractFileSystem, str]:
    return fsspec.core.url_to_fs(uri)  # type: ignore[no-any-return]


def _tar_directory_to_bytes(local_dir: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        tar.add(local_dir, arcname=".")
    return buffer.getvalue()


def _untar_bytes_to_directory(data: bytes, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        tar.extractall(dest_dir, filter="data")  # "data" filter: refuses absolute paths/symlinks escaping dest_dir


def _upload_encrypted_directory(local_dir: str, remote_uri: str, *, encryption_key: bytes) -> None:
    """Tar `local_dir`, encrypt (SPEC.md §8), upload as one blob to
    `remote_uri`/`ARCHIVE_FILENAME`, then write `COMPLETION_MARKER` last."""
    archive_bytes = encryption.encrypt_bytes(_tar_directory_to_bytes(local_dir), encryption_key)
    fs, remote_path = _fs_and_path(remote_uri)
    fs.makedirs(remote_path, exist_ok=True)
    fs.pipe_file(f"{remote_path.rstrip('/')}/{ARCHIVE_FILENAME}", archive_bytes)
    fs.pipe_file(f"{remote_path.rstrip('/')}/{COMPLETION_MARKER}", b"done")


def _download_and_decrypt_directory(remote_uri: str, local_dir: str, *, encryption_key: bytes) -> None:
    fs, remote_path = _fs_and_path(remote_uri)
    archive_bytes = fs.cat_file(f"{remote_path.rstrip('/')}/{ARCHIVE_FILENAME}")
    _untar_bytes_to_directory(encryption.decrypt_bytes(archive_bytes, encryption_key), local_dir)


def upload_checkpoint(local_checkpoint_dir: str, remote_checkpoint_uri: str, *, encryption_key: bytes) -> None:
    """Used by `R2CheckpointCallback.on_save`. Validates SPEC.md §7.4's six
    required artifacts are actually present locally before bothering to
    tar/encrypt/upload anything — fail loudly on a malformed local checkpoint
    rather than silently ship an incomplete one."""
    local_names = set(os.listdir(local_checkpoint_dir))
    missing = [artifact for artifact in REQUIRED_CHECKPOINT_ARTIFACTS if artifact not in local_names]
    if missing:
        raise AssertionError(f"local checkpoint at {local_checkpoint_dir} is missing required artifacts: {missing}")
    _upload_encrypted_directory(local_checkpoint_dir, remote_checkpoint_uri, encryption_key=encryption_key)


def upload_adapter(local_adapter_dir: str, remote_adapter_uri: str, *, encryption_key: bytes) -> None:
    """Used by `twin.train.run` for the final adapter save. Narrower
    validation than `upload_checkpoint`: a bare `PeftModel.save_pretrained`
    output has no optimizer/scheduler/RNG/trainer-state files."""
    local_names = set(os.listdir(local_adapter_dir))
    missing = [artifact for artifact in ADAPTER_SAVE_ARTIFACTS if artifact not in local_names]
    if missing:
        raise AssertionError(f"local adapter save at {local_adapter_dir} is missing required artifacts: {missing}")
    _upload_encrypted_directory(local_adapter_dir, remote_adapter_uri, encryption_key=encryption_key)


def is_checkpoint_complete(checkpoint_uri: str) -> bool:
    """The archive AND `COMPLETION_MARKER` both present. Missing directory
    (nothing uploaded yet) is `False`, not an error — that's the expected
    state before the first checkpoint lands."""
    fs, path = _fs_and_path(checkpoint_uri)
    if not fs.exists(path):
        return False
    names = {entry.rsplit("/", 1)[-1] for entry in fs.ls(path, detail=False)}
    return ARCHIVE_FILENAME in names and COMPLETION_MARKER in names


def list_checkpoints(run_root_uri: str) -> list[tuple[int, str]]:
    """`(step, checkpoint_uri)` pairs for `checkpoint-<N>` subdirectories
    under `run_root_uri`, ascending by step. Empty list if `run_root_uri`
    doesn't exist yet — a genuinely fresh run, not an error."""
    fs, path = _fs_and_path(run_root_uri)
    if not fs.exists(path):
        return []
    pairs: list[tuple[int, str]] = []
    for entry in fs.ls(path, detail=False):
        name = entry.rsplit("/", 1)[-1]
        if name.startswith("checkpoint-"):
            step_str = name.removeprefix("checkpoint-")
            if step_str.isdigit():
                pairs.append((int(step_str), f"{run_root_uri.rstrip('/')}/{name}"))
    return sorted(pairs, key=lambda pair: pair[0])


def prune_remote_checkpoints(run_root_uri: str, *, keep: int = REMOTE_CHECKPOINTS_TO_KEEP) -> list[str]:
    """Deletes all but the `keep` highest-step checkpoints under
    `run_root_uri`. Only complete checkpoints count toward `keep`; an
    incomplete (torn) one is never what protects resume, so it is removed
    too unless it is newer than everything kept. Returns the removed URIs."""
    if keep < 1:
        raise ValueError("keep must be >= 1 — pruning every checkpoint would destroy resume")
    complete = [(step, uri) for step, uri in list_checkpoints(run_root_uri) if is_checkpoint_complete(uri)]
    kept_steps = {step for step, _ in complete[-keep:]}
    removed: list[str] = []
    for step, uri in list_checkpoints(run_root_uri):
        if step in kept_steps or (complete and step > max(kept_steps)):
            continue
        fs, path = _fs_and_path(uri)
        fs.rm(path, recursive=True)
        removed.append(uri)
    return removed


def find_latest_complete_checkpoint(run_root_uri: str) -> str | None:
    """Newest-first scan via `list_checkpoints`, skipping incomplete
    directories (a checkpoint mid-upload when the run was killed). Returns
    `None` when nothing valid exists — the correct, expected result on a
    genuinely fresh `run_id`, not an error `--resume auto` should surface."""
    for _step, checkpoint_uri in reversed(list_checkpoints(run_root_uri)):
        if is_checkpoint_complete(checkpoint_uri):
            return checkpoint_uri
    return None


def download_checkpoint_for_resume(remote_checkpoint_uri: str, local_scratch_dir: str, *, encryption_key: bytes) -> str:
    """Decrypts and unpacks into a local directory —
    `Trainer.train(resume_from_checkpoint=...)` expects a local filesystem
    path, not an arbitrary fsspec URI. Returns that local path."""
    _fs, remote_path = _fs_and_path(remote_checkpoint_uri)
    local_path = f"{local_scratch_dir.rstrip('/')}/{remote_path.rsplit('/', 1)[-1]}"
    _download_and_decrypt_directory(remote_checkpoint_uri, local_path, encryption_key=encryption_key)
    return local_path


@dataclass
class R2CheckpointCallback(TrainerCallback):
    """Two responsibilities: (a) requests an out-of-cadence save roughly
    every `interval_seconds` (SPEC.md §7.4 SHOULD 10-15 min) via
    `TrainerControl.should_save`, since Trainer's built-in `save_strategy` is
    step-count-based, not wall-clock-based; (b) on `on_save`, encrypts and
    syncs the checkpoint Trainer just wrote locally up to
    `remote_run_root_uri` (SPEC.md §8; see `upload_checkpoint`).
    """

    remote_run_root_uri: str
    encryption_key: bytes
    interval_seconds: float = 600.0  # 10 min; tune within SPEC's 10-15 min SHOULD once real upload timing is measured.
    # Tests pass a small value (e.g. 0) so a toy CI run doesn't wait for a real interval.
    _last_save_monotonic: float | None = field(default=None, repr=False)

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> TrainerControl:
        import time

        now = time.monotonic()
        if self._last_save_monotonic is None:
            # First step of this process's lifetime (fresh start or just-resumed):
            # record the clock, don't force an immediate extra save.
            self._last_save_monotonic = now
            return control
        if now - self._last_save_monotonic >= self.interval_seconds:
            control.should_save = True
            self._last_save_monotonic = now
        return control

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> TrainerControl:
        assert args.output_dir is not None, "SFTConfig always sets output_dir explicitly (see train.run.main)"
        local_checkpoint_dir = f"{args.output_dir.rstrip('/')}/checkpoint-{state.global_step}"
        remote_checkpoint_uri = f"{self.remote_run_root_uri.rstrip('/')}/checkpoint-{state.global_step}"
        upload_checkpoint(local_checkpoint_dir, remote_checkpoint_uri, encryption_key=self.encryption_key)
        # Only after the new one is complete: the retention floor above is
        # measured in *complete* checkpoints, so pruning never runs ahead.
        prune_remote_checkpoints(self.remote_run_root_uri)
        return control
