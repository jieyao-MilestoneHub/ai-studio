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
    # Resolve the venv at image-build time (cached across calls) — the first
    # real run (2026-08-29) did `uv sync` per call and then invoked bare
    # `python`, i.e. the system interpreter without torch. Every entrypoint
    # below MUST go through `uv run` for the same reason.
    # torch's ~900 MB wheel exceeded uv's default 30 s HTTP timeout once on
    # Modal's builder (2026-08-30, "Failed to download torch==2.13.0 ...
    # network timeout"); a generous timeout is cheaper than a rebuilt image.
    .env({"UV_HTTP_TIMEOUT": "600"})
    .run_commands("cd /twin && uv sync --no-dev")
)

checkpoint_volume = modal.Volume.from_name("twin-checkpoints", create_if_missing=True)
gemini_secret = modal.Secret.from_name("twin-gemini")  # unused by train.py itself; kept for parity with teacher.py runs sharing this image
# Everything train.py reads from twin/.env locally (TWIN_CHECKPOINT_STORE_URI,
# TWIN_TRAJECTORY_STORE_URI, TWIN_ADAPTER_ENCRYPTION_KEY, AWS_* for R2,
# TWIN_PRINCIPAL_ID) — the image deliberately excludes .env, so on Modal these
# arrive as a Secret. Create it once from the local .env:
#   uv run modal secret create twin-train $(grep -E '^(TWIN_(CHECKPOINT|TRAJECTORY)_STORE_URI|TWIN_ADAPTER_ENCRYPTION_KEY|TWIN_PRINCIPAL_ID|AWS_)' .env | xargs)
train_secret = modal.Secret.from_name("twin-train")


@app.function(
    image=image,
    # Ordered fallback: the first real run was preempted at step 180 and then
    # sat >7h "waiting to be scheduled on a GPU_T4 worker". Any of these fits
    # r=64 QLoRA (probe: 9.2 GiB peak on T4); L4/A10G are 24 GB and only
    # modestly pricier per hour, far cheaper than an idle day.
    gpu=["T4", "L4", "A10G"],
    timeout=6 * 60 * 60,  # SPEC.md §7.3: Modal Starter is the primary loop; long runs go to Kaggle instead of a longer timeout here
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[gemini_secret, train_secret],
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
    subprocess.run(["uv", "run", "--no-sync", "python", "train.py", "--resume", "auto", *args], cwd="/twin", check=True)


@app.function(image=image, gpu="T4", timeout=30 * 60, secrets=[train_secret])
def probe_entrypoint() -> None:
    """examples/probe_lora_rank.py on the real target GPU (SPEC.md §11 item G:
    a human reads the output and records the chosen rank in TrainingConfig)."""
    subprocess.run(["uv", "run", "--no-sync", "python", "examples/probe_lora_rank.py"], cwd="/twin", check=True)


@app.function(
    image=image,
    gpu=["T4", "L4", "A10G"],
    timeout=2 * 60 * 60,
    secrets=[train_secret],
)
def s1_candidates_fn(
    item_bank_jsonl: str,
    manifest_json: str,
    labels: str,
    persona: str | None,
    transcript: str | None,
    adapter_uri: str | None,
    consistency_probe: int,
) -> dict[str, str]:
    """examples/generate_s1_candidates.py on a GPU. Inputs arrive as function
    arguments and are written to the container's ephemeral /tmp — never to a
    Volume, never to R2: the B2 transcript is self-report (INTERVIEW.md
    §6.3 forbids it in cross-cloud *storage*; a container's memory for the
    duration of one call is the minimum exposure that lets B2 exist at all
    without a local GPU — recorded in twin/PLAN.md Phase 5). Outputs return
    the same way, as file contents, for the local caller to persist."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "s1").mkdir()
        (root / "s1" / "item_bank.jsonl").write_text(item_bank_jsonl, encoding="utf-8")
        (root / "s1" / "manifest.json").write_text(manifest_json, encoding="utf-8")
        cmd = [
            "uv", "run", "--no-sync", "python", "examples/generate_s1_candidates.py",
            "--s1-root", (root / "s1").as_uri(), "--out-dir", str(root / "out"),
            "--labels", labels, "--consistency-probe", str(consistency_probe),
        ]
        if persona is not None:
            (root / "persona.txt").write_text(persona, encoding="utf-8")
            cmd += ["--persona-file", str(root / "persona.txt")]
        if transcript is not None:
            (root / "transcript.txt").write_text(transcript, encoding="utf-8")
            cmd += ["--transcript-file", str(root / "transcript.txt")]
        if adapter_uri:
            cmd += ["--adapter-uri", adapter_uri]
        subprocess.run(cmd, cwd="/twin", check=True)
        return {p.name: p.read_text(encoding="utf-8") for p in sorted((root / "out").glob("*.jsonl"))}


@app.local_entrypoint()
def s1_candidates(labels: str = "B0", adapter_uri: str = "", consistency_probe: int = 0) -> None:
    """Local side: read the frozen bank (+ persona / self-report transcript
    when B1/B2 are requested) from the URIs in twin/.env, run the GPU
    function, persist results under <TWIN_S1_EVAL_ROOT_URI>/candidates/.

        uv run modal run launch/modal_app.py::s1_candidates --labels B0,T \\
            --adapter-uri s3://twin-checkpoints/default/run_.../final --consistency-probe 8
    """
    import fsspec
    from dotenv import load_dotenv

    load_dotenv()
    from twin.config.settings import get_settings
    from twin.harness.baseline import default_persona_uri, load_self_report_transcript

    settings = get_settings()
    s1_root = settings.s1_eval_root_uri.rstrip("/")
    with fsspec.open(f"{s1_root}/item_bank.jsonl", "r", encoding="utf-8") as f:
        bank = f.read()
    with fsspec.open(f"{s1_root}/manifest.json", "r", encoding="utf-8") as f:
        manifest = f.read()
    wanted = {lbl.strip() for lbl in labels.split(",")}
    persona = transcript = None
    if "B1" in wanted:
        with fsspec.open(default_persona_uri(settings.fragment_store_uri), "r", encoding="utf-8") as f:
            persona = f.read()
    if "B2" in wanted:
        transcript = load_self_report_transcript(settings.fragment_store_uri)
        if transcript is None:
            raise SystemExit("B2 requested but the fragment store has no self-report yet (Phase 3)")
        print(f"B2 transcript: {len(transcript)} chars from the local fragment store (sent to the GPU container's memory only)")

    outputs = s1_candidates_fn.remote(bank, manifest, labels, persona, transcript, adapter_uri or None, consistency_probe)
    for name, content in outputs.items():
        uri = f"{s1_root}/candidates/{name}"
        fs, path = fsspec.core.url_to_fs(uri)
        fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        with fsspec.open(uri, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"wrote {uri} ({content.count(chr(10))} lines)")
