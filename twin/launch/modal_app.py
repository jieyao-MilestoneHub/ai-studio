"""Modal infrastructure glue ONLY — the one allowed home for `import modal`
outside `launch/*.sh` itself (SPEC.md §7.1/D12). Deliberately does not import
anything from `twin.train`: the function body only shells out to the same
`train.py --resume auto` entrypoint every other launch path uses, so
`train.py` never has to know it's running on Modal.

Lives in `launch/` but outside `src/twin/` — import-linter's
`root_package = "twin"` never scans this file, so importing `modal` here
cannot violate the "train.py stays cloud-agnostic" contract even though this
file itself is unapologetically Modal-specific.

VERIFY AT IMPLEMENTATION TIME (not yet confirmed against a real Modal
account): the exact `Volume`/`Secret` names below, the GPU string
("T4" vs a more specific SKU), and the CUDA-matched torch install step that
launch/modal.sh comments out — this file is a best-reasoned skeleton, not a
tested deployment.
"""

from __future__ import annotations

import subprocess

import modal

app = modal.App("twin-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("pip install uv")
    .add_local_dir(
        ".",
        remote_path="/twin",
        copy=True,
        # SPEC.md §8 guardrail 2: data/adapters/transcripts/eval MUST NOT
        # leave local storage. twin/.gitignore + the repo-root pre-commit
        # hook only block that path into *version control* — this
        # add_local_dir call is a completely separate channel that would
        # otherwise bake those directories straight into a Modal image
        # layer if they happened to be populated locally when this runs.
        # .env is excluded because it holds real credentials (Gemini/R2/
        # Kaggle) — this image already gets TWIN_GEMINI_* via the
        # `twin-gemini` Modal Secret, not by shipping the plaintext file.
        ignore=[".env", ".git", ".venv", "__pycache__", "data", "adapters", "transcripts", "eval"],
    )
)

checkpoint_volume = modal.Volume.from_name("twin-checkpoints", create_if_missing=True)
gemini_secret = modal.Secret.from_name("twin-gemini")  # unused by train.py itself; kept for parity with teacher.py runs sharing this image


@app.function(
    image=image,
    gpu="T4",
    timeout=6 * 60 * 60,  # SPEC.md §7.3: Modal Starter is the primary loop; long runs go to Kaggle instead of a longer timeout here
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[gemini_secret],
    # SPEC.md §7.3 "MUST 優先使用 spot/preemptible": no opt-in needed here —
    # checked live against Modal's docs (2026-08-27): "All Modal Functions
    # are subject to preemption by default," and the `nonpreemptible`
    # parameter is explicitly NOT supported for GPU Functions. Every GPU
    # Function on Modal is preemptible unconditionally; this MUST is already
    # satisfied by the plain decorator above, not something to additionally
    # configure. This is exactly why train.py's checkpoint/resume contract
    # (SPEC.md §7.4) matters here specifically, not just in theory.
)
def train_entrypoint(*args: str) -> None:
    subprocess.run(["uv", "sync"], cwd="/twin", check=True)
    subprocess.run(["python", "train.py", "--resume", "auto", *args], cwd="/twin", check=True)
