"""The pod-side understanding server: describes a photo/audio/video clip.

Runs ON the pod, alongside ComfyUI, as a second always-resident process on a
second port (8189). Deposited and started by `deploy/pod_setup.sh` via
`runtime.session.provision()` -- see that module's docstring for why this is
a *second* file over the same one-file-at-a-time SSH transport rather than
being embedded inline.

**Lazy load/unload, one model at a time.** Only one of {moondream3-preview,
Qwen3-Omni-30B-A3B-Captioner (Q4), Tarsier2-7b-0115} is ever resident in
VRAM, and never at the same time as ComfyUI's own checkpoint -- the whole
card is 24GB and none of these comfortably coexist with an H3/Flux model.
`POST /submit` evicts whatever is currently loaded (if it is not what this
request needs) before loading the requested model; `POST /unload` is called
by the pipeline's pull-based GPU hand-off (`pipeline.drain.make_room_for`)
right before a ComfyUI generation job runs.

**Concurrency is 1, inherited rather than enforced here.** The shared FIFO
`JobQueue` on the ai-studio side already serializes every job kind, so this
server never receives two concurrent `/submit` calls in practice; the queue
upstream is what guarantees that, not a lock in this file.

⚠️ `[speculative]`: none of the three model-loading code paths below have
run against real weights on this project's own hardware yet. Each is a
best-effort reading of the model's published API surface (moondream3 and
Tarsier2 both ship custom `trust_remote_code=True` modeling code; Qwen3-Omni-
Captioner's Q4 form may turn out to require `llama-cpp-python`/AWQ bindings
rather than a plain `transformers` 4-bit load, depending on which quantized
artifact is actually published) -- verify each against the actual model
card before the first real deployment, and promote to measured (📏) only
after it has actually produced a description on this hardware. See
`docs/model-moondream3.md`, `docs/model-qwen3-omni-captioner.md`,
`docs/model-tarsier2.md`.

Not part of the `ai_studio` Python package -- this file is copied to the pod
and run standalone (`python3 inference_server.py`), so it has no access to
`ai_studio.core`/`ai_studio.config` and intentionally does not import them.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="[inference] %(message)s")
_log = logging.getLogger("inference_server")

UPLOAD_DIR = Path("/workspace/inference_uploads")
MAX_OUTPUT_CHARS = 1000
"""Ceiling on a returned description -- mirrors
`ai_studio.core.understanding_spec.UnderstandingCapabilities.max_output_chars`
on the other side of the wire; kept here too so a runaway generation cannot
return an unbounded response."""


# --------------------------------------------------------------- backends


class ModelBackend(Protocol):
    """One modality's load/infer/unload cycle. Implemented per model below."""

    modality: str
    model_id: str
    accepts_prompt: bool

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def infer(self, media_path: Path, prompt: str | None) -> str: ...


class MoondreamBackend:
    """moondream/moondream3-preview -- image captioning/VQA.

    `[speculative]` loader: moondream ships custom modeling code with a
    `.query(image, question)` method on the loaded model object, per its
    model card. `Florence-2-large` (the documented VRAM-constrained fallback,
    <2GB) would use a different call shape (`AutoProcessor` + a task prompt
    like `"<MORE_DETAILED_CAPTION>"`) -- swap `MODEL_ID` and the `infer`
    body together if moondream3's real >16GB footprint is too much.
    """

    modality = "image"
    model_id = "moondream/moondream3-preview"
    accepts_prompt = True

    def __init__(self) -> None:
        self._model: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=torch.float16
        ).to("cuda")

    def unload(self) -> None:
        import torch

        self._model = None
        torch.cuda.empty_cache()

    def infer(self, media_path: Path, prompt: str | None) -> str:
        from PIL import Image

        image = Image.open(media_path).convert("RGB")
        question = prompt or "Describe this image in detail."
        result = self._model.query(image, question)
        return str(result.get("answer", result) if isinstance(result, dict) else result)


class Qwen3OmniCaptionerBackend:
    """Qwen/Qwen3-Omni-30B-A3B-Captioner -- audio captioning, Q4 quantized.

    `[speculative]` loader, and the least certain of the three: **rejects a
    text prompt outright** per its own model card, so `infer` ignores
    `prompt` entirely rather than raising -- the caller-side
    `UnderstandingCapabilities.accepts_prompt = False` is what should stop a
    prompt arriving here in the first place. The Q4 form is loaded via
    `transformers`' own 4-bit quantization here; if the published artifact
    is GGUF rather than a quantizable safetensors checkpoint, this loader
    needs to switch to `llama-cpp-python` bindings instead -- check the
    actual repo contents before deploying.
    """

    modality = "audio"
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
    accepts_prompt = False
    max_input_seconds = 30.0

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        # The exact model class for Qwen3-Omni is not yet pinned here --
        # `AutoModelForCausalLM` is the generic fallback; swap in the
        # model's own class (as its README specifies) if this errors.
        from transformers import AutoModelForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, quantization_config=quant_config,
            device_map="cuda",
        )

    def unload(self) -> None:
        import torch

        self._model = None
        self._processor = None
        torch.cuda.empty_cache()

    def infer(self, media_path: Path, prompt: str | None) -> str:
        import soundfile as sf

        audio, sample_rate = sf.read(str(media_path))
        inputs = self._processor(audio=audio, sampling_rate=sample_rate, return_tensors="pt").to(
            "cuda"
        )
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        return str(self._processor.batch_decode(output_ids, skip_special_tokens=True)[0])


class Tarsier2Backend:
    """omni-research/Tarsier2-7b-0115 -- dense video captioning, FP16.

    `[speculative]` loader: Tarsier ships its own `trust_remote_code=True`
    modeling code with a chat-style video-QA interface. The lightest of the
    three (~14-16GB `[reported]`), and the only one with no stated input
    length ceiling -- see `docs/model-tarsier2.md`'s flagged-unresolved
    `MAX_VIDEO_UNDERSTAND_S`.
    """

    modality = "video"
    model_id = "omni-research/Tarsier2-7b-0115"
    accepts_prompt = True

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=torch.float16
        ).to("cuda")

    def unload(self) -> None:
        import torch

        self._model = None
        self._processor = None
        torch.cuda.empty_cache()

    def infer(self, media_path: Path, prompt: str | None) -> str:
        question = prompt or "Describe what happens in this video in detail."
        inputs = self._processor(text=question, videos=str(media_path), return_tensors="pt").to(
            "cuda"
        )
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        return str(self._processor.batch_decode(output_ids, skip_special_tokens=True)[0])


_BACKENDS: dict[str, type[ModelBackend]] = {
    "image": MoondreamBackend,
    "audio": Qwen3OmniCaptionerBackend,
    "video": Tarsier2Backend,
}


# ------------------------------------------------------------------- state


@dataclass
class Job:
    job_id: str
    modality: str
    media_path: Path
    prompt: str | None
    state: str = "queued"  # queued | running | completed | failed
    result_text: str | None = None
    error: str | None = None


@dataclass
class ModelSlot:
    """The one currently-loaded backend, if any. Guarded by `_lock` so a
    submit and an /unload cannot race each other onto the same GPU."""

    backend: ModelBackend | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_slot = ModelSlot()
_jobs: dict[str, Job] = {}
_running_tasks: set[asyncio.Task[None]] = set()
"""Keeps a strong reference to every in-flight `_run_job` task -- asyncio
only holds a weak reference to a task created with `create_task`, so an
unreferenced one can be garbage-collected mid-run, silently killing the job.
Discarded via the task's own done-callback once it finishes."""


async def _ensure_loaded(modality: str) -> ModelBackend:
    async with _slot.lock:
        if _slot.backend is not None and _slot.backend.modality == modality:
            return _slot.backend
        if _slot.backend is not None:
            _log.info("evicting %s to load %s", _slot.backend.modality, modality)
            _slot.backend.unload()
            _slot.backend = None
        backend_cls = _BACKENDS[modality]
        backend = backend_cls()
        _log.info("loading %s (%s)", backend.model_id, modality)
        started = time.monotonic()
        backend.load()
        _log.info("loaded %s in %.0fs", backend.model_id, time.monotonic() - started)
        _slot.backend = backend
        return backend


async def _run_job(job: Job) -> None:
    job.state = "running"
    try:
        backend = await _ensure_loaded(job.modality)
        # Run the blocking model call off the event loop -- this is the one
        # call in the whole server that can take tens of seconds.
        result = await asyncio.to_thread(backend.infer, job.media_path, job.prompt)
        job.result_text = result[:MAX_OUTPUT_CHARS]
        job.state = "completed"
    except Exception as exc:  # a job's own failure must not crash the server
        _log.exception("job %s failed", job.job_id)
        job.error = str(exc)
        job.state = "failed"


# --------------------------------------------------------------------- app

app = FastAPI(title="ai-studio-inference")


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Ready the moment the process is up -- no model needs to be loaded.
    Distinct from ComfyUI's node-pack readiness probe on purpose; see
    `runtime.session.wait_understanding_ready`."""
    return {"ok": True, "loaded": _slot.backend.modality if _slot.backend else None}


@app.post("/submit")
async def submit(
    media: UploadFile = File(...),
    modality: str = Form(...),
    prompt: str | None = Form(None),
) -> JSONResponse:
    if modality not in _BACKENDS:
        raise HTTPException(400, f"unknown modality {modality!r}, expected one of {list(_BACKENDS)}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{media.filename or 'upload'}"
    dest.write_bytes(await media.read())

    job_id = uuid.uuid4().hex
    job = Job(job_id=job_id, modality=modality, media_path=dest, prompt=prompt)
    _jobs[job_id] = job
    # Fire-and-forget: the caller polls. A cold model load can plausibly
    # exceed the ~100s Cloudflare window on RunPod's pod proxy even though a
    # warm inference call would not, so this must never block the request.
    task = asyncio.create_task(_run_job(job))
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)
    return JSONResponse({"job_id": job_id})


@app.get("/poll/{job_id}")
async def poll(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")
    return {"state": job.state, "result_text": job.result_text, "error": job.error}


@app.post("/cancel/{job_id}")
async def cancel(job_id: str) -> dict[str, Any]:
    """Best-effort. A job already `running` cannot be preempted mid-generate
    -- there is no cancellation hook into a blocking `model.generate()` call
    -- so this only prevents a still-`queued` job from starting."""
    job = _jobs.get(job_id)
    if job is not None and job.state == "queued":
        job.state = "failed"
        job.error = "cancelled"
    return {"ok": True}


@app.post("/unload")
async def unload() -> dict[str, Any]:
    """Evict the resident model so a ComfyUI generation job can use the same
    card. Called by `pipeline.drain.make_room_for` before every generation
    submit -- see that function's docstring for the pull-based hand-off."""
    async with _slot.lock:
        if _slot.backend is not None:
            _log.info("unloading %s (GPU hand-off to ComfyUI)", _slot.backend.modality)
            _slot.backend.unload()
            _slot.backend = None
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8189)
