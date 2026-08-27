"""The pod-side understanding + chat server: describes a photo/audio/video
clip, or answers a /himonkey chat message.

Runs ON the pod, alongside ComfyUI, as a second always-resident process on a
second port (8189). Deposited and started by `deploy/pod_setup.sh` via
`runtime.session.provision()` -- see that module's docstring for why this is
a *second* file over the same one-file-at-a-time SSH transport rather than
being embedded inline.

**Lazy load/unload, one model at a time.** Only one of {moondream3-preview,
Qwen3-Omni-30B-A3B-Captioner (Q4), Tarsier2-7b-0115, gpt-oss-20b} is ever
resident in VRAM, and never at the same time as ComfyUI's own checkpoint --
the whole card is 24GB and none of these comfortably coexist with an H3/Flux
model (nor, for gpt-oss-20b specifically, with H3 itself: H3 alone measures
22.1-22.8GB peak on the actual RTX 4090 this project targets, so there is
never room regardless of gpt-oss-20b's ~16GB native-MXFP4 footprint). `POST
/submit` evicts whatever is currently loaded (if it is not what this request
needs) before loading the requested model; `POST /unload` is called by the
pipeline's pull-based GPU hand-off (`pipeline.drain.make_room_for`) right
before a ComfyUI generation job runs.

**Concurrency is 1, inherited rather than enforced here.** The shared FIFO
`JobQueue` on the ai-studio side already serializes every job kind, so this
server never receives two concurrent `/submit` calls in practice; the queue
upstream is what guarantees that, not a lock in this file.

**Eviction waits for the in-flight `infer()` call to finish before unloading
its model — this is required, not defensive polish.** `POST /cancel` is
best-effort and cannot actually stop a running `model.generate()` (see its
own docstring); if the caller gives up on a slow job and moves on, the
abandoned background thread keeps running against the loaded model. Without
`ModelSlot.in_flight`, the *next* `/submit`/`/unload` would tear that model's
CUDA state out from under the still-running thread -- a use-after-free at
the CUDA level that can crash this whole process (taking every modality down
with it, not just whichever one was mid-generation), not something Python's
own exception handling can catch. gpt-oss-20b is what makes this likely to
actually fire: its generation length isn't bounded by a fixed input shape
the way a single-image caption or scored fixed-length transcript is, so it
is both the most likely modality to run long and the reason this fix
exists. `GptOssChatBackend.MAX_NEW_TOKENS` is the primary defense (bound the
thing that can hang); `_wait_for_idle()`/`UNLOAD_WAIT_TIMEOUT_S` is the
secondary one and must stay comfortably above it.

⚠️ `[speculative]`: none of the four model-loading code paths below have
run against real weights on this project's own hardware yet. Each is a
best-effort reading of the model's published API surface (moondream3 and
Tarsier2 both ship custom `trust_remote_code=True` modeling code; Qwen3-Omni-
Captioner's Q4 form may turn out to require `llama-cpp-python`/AWQ bindings
rather than a plain `transformers` 4-bit load, depending on which quantized
artifact is actually published; gpt-oss-20b's harmony channel-tag syntax in
`_final_channel()` below is a best-effort reading of the published format,
not yet verified against a real generation) -- verify each against the
actual model card before the first real deployment, and promote to measured
(📏) only after it has actually produced output on this hardware. See
`docs/model-moondream3.md`, `docs/model-qwen3-omni-captioner.md`,
`docs/model-tarsier2.md`, `docs/model-gpt-oss-20b.md`.

Not part of the `ai_studio` Python package -- this file is copied to the pod
and run standalone (`python3 inference_server.py`), so it has no access to
`ai_studio.core`/`ai_studio.config` and intentionally does not import them.
"""

from __future__ import annotations

import asyncio
import json
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
"""Ceiling on a returned description/reply -- mirrors
`ai_studio.core.understanding_spec.UnderstandingCapabilities.max_output_chars`
/ `ai_studio.core.chat_spec.ChatCapabilities.max_output_chars` on the other
side of the wire; kept here too so a runaway generation cannot return an
unbounded response."""


# --------------------------------------------------------------- backends


class ModelBackend(Protocol):
    """One modality's load/infer/unload cycle. Implemented per model below."""

    modality: str
    model_id: str
    accepts_prompt: bool

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def infer(
        self, media_path: Path | None, prompt: str | None, *, history: str | None = None
    ) -> str: ...


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

    def infer(self, media_path: Path | None, prompt: str | None, *, history: str | None = None) -> str:
        from PIL import Image

        assert media_path is not None  # only "chat" jobs ever omit it
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

    def infer(self, media_path: Path | None, prompt: str | None, *, history: str | None = None) -> str:
        import soundfile as sf

        assert media_path is not None  # only "chat" jobs ever omit it
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

    def infer(self, media_path: Path | None, prompt: str | None, *, history: str | None = None) -> str:
        assert media_path is not None  # only "chat" jobs ever omit it
        question = prompt or "Describe what happens in this video in detail."
        inputs = self._processor(text=question, videos=str(media_path), return_tensors="pt").to(
            "cuda"
        )
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        return str(self._processor.batch_decode(output_ids, skip_special_tokens=True)[0])


class GptOssChatBackend:
    """openai/gpt-oss-20b -- plain-text /himonkey chat, native MXFP4.

    `[speculative]` loader, same status as the three backends above: nothing
    here has run against real weights on this project's own hardware yet.
    See `docs/model-gpt-oss-20b.md`.

    gpt-oss-20b's chat template emits the "harmony" response format --
    separate analysis/commentary/final channels -- even with zero tool
    calling in play. `infer()` returns only the final channel via
    `_final_channel()`; a naive decode-and-return would leak the model's
    internal chain-of-thought into the LINE reply. `apply_chat_template()`
    renders the harmony-formatted prompt for us, which is why nothing here
    needs the separate `openai-harmony` pip package.
    """

    modality = "chat"
    model_id = "openai/gpt-oss-20b"
    accepts_prompt = True

    MAX_NEW_TOKENS = 512
    """The primary defense against an unbounded generation (see this file's
    module docstring on why `/unload` racing a still-running `infer()` call
    is the single highest-severity risk `/himonkey` introduces). Sized well
    above `MAX_OUTPUT_CHARS`'s character budget in tokens, with headroom --
    tune down once a real token-to-character ratio is measured on this
    model. A wall-clock `StoppingCriteria` cutoff would be a good second,
    independent backstop if the eventual serving stack exposes one; this
    token cap alone already guarantees termination regardless."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="cuda"
        )

    def unload(self) -> None:
        import torch

        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()

    def infer(self, media_path: Path | None, prompt: str | None, *, history: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if history:
            # (role, content) pairs, oldest first -- see
            # `ai_studio.pipeline.queue.JobQueue.recent_chat_turns`, the
            # host-side store this was fetched from.
            for role, content in json.loads(history):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt or ""})

        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        output_ids = self._model.generate(input_ids, max_new_tokens=self.MAX_NEW_TOKENS)
        decoded = self._tokenizer.decode(
            output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True
        )
        return _final_channel(decoded)


def _final_channel(text: str) -> str:
    """Pull only harmony's "final" channel out of a raw gpt-oss-20b decode.

    `[speculative]`: the exact channel-tag syntax has not been verified
    against a real generation on this hardware yet -- a best-effort reading
    of the published format, to be corrected against real output before the
    first deployment. Falls back to the whole decoded text if no channel
    marker is found rather than returning nothing -- a reply that leaked a
    thinking trace is a worse failure than skipped channel-splitting, but
    returning nothing at all is worse still.
    """
    marker = "<|channel|>final<|message|>"
    if marker in text:
        return text.split(marker, 1)[1].split("<|end|>", 1)[0].strip()
    return text.strip()


_BACKENDS: dict[str, type[ModelBackend]] = {
    "image": MoondreamBackend,
    "audio": Qwen3OmniCaptionerBackend,
    "video": Tarsier2Backend,
    "chat": GptOssChatBackend,
}


# ------------------------------------------------------------------- state


@dataclass
class Job:
    job_id: str
    modality: str
    media_path: Path | None
    prompt: str | None
    history: str | None = None
    """JSON-encoded (role, content) pairs, oldest first -- chat only. See
    `GptOssChatBackend.infer()`."""
    state: str = "queued"  # queued | running | completed | failed
    result_text: str | None = None
    error: str | None = None


@dataclass
class ModelSlot:
    """The one currently-loaded backend, if any. Guarded by `lock` so a
    submit and an /unload cannot race each other onto the same GPU.

    `in_flight` is the second half of that guarantee, and the more important
    one: `lock` only ever protects the brief evict/construct/load sequence,
    never the blocking `infer()` call itself (that runs in a background
    thread via `asyncio.to_thread`, deliberately off the lock so a slow
    generation cannot stall `/submit`/`/unload` for every other request).
    Without `in_flight`, eviction could unload a model while a background
    thread is still inside a live CUDA kernel using it -- see this file's
    module docstring for why that is the single highest-severity risk
    `/himonkey` introduces, and `_wait_for_idle()` for the wait this field
    backs.
    """

    backend: ModelBackend | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    in_flight: int = 0


_slot = ModelSlot()
_jobs: dict[str, Job] = {}
_running_tasks: set[asyncio.Task[None]] = set()
"""Keeps a strong reference to every in-flight `_run_job` task -- asyncio
only holds a weak reference to a task created with `create_task`, so an
unreferenced one can be garbage-collected mid-run, silently killing the job.
Discarded via the task's own done-callback once it finishes."""

UNLOAD_WAIT_TIMEOUT_S = 120.0
"""How long eviction waits for an in-flight `infer()` call to clear before
giving up and unloading anyway. Must stay comfortably above
`GptOssChatBackend.MAX_NEW_TOKENS`'s own worst-case generation time so this
only ever fires in the genuinely pathological case -- an ordinary reply
finishes well inside it. Past this point a truly stuck thread would
otherwise wedge the pod's GPU-hand-off forever; unloading anyway risks the
same CUDA-level crash this mechanism exists to avoid, but a permanently
unusable slot is strictly worse, so this is a last resort, not a target."""


async def _wait_for_idle(backend: ModelBackend, *, timeout_s: float = UNLOAD_WAIT_TIMEOUT_S) -> None:
    """Block until no `infer()` call is running against `backend`, or until
    `timeout_s` elapses (logged loudly, then proceeds anyway regardless).

    Must be called with `_slot.lock` NOT held: `infer()` never touches the
    lock (see `ModelSlot`'s docstring), so holding it here would only block
    this wait for no reason.
    """
    deadline = time.monotonic() + timeout_s
    while _slot.backend is backend and _slot.in_flight > 0:
        if time.monotonic() >= deadline:
            _log.warning(
                "unloading %s with %d in-flight infer() call(s) still running after "
                "%.0fs -- this can crash the process; see this file's module docstring",
                backend.modality, _slot.in_flight, timeout_s,
            )
            return
        await asyncio.sleep(0.5)


def _release_vram() -> None:
    """Best-effort VRAM release after a failed `backend.load()`.

    A partial load can claim CUDA memory before raising, and nothing else
    releases it on that path -- the next job of *any* kind can then fail to
    allocate VRAM it should have had, with a symptom ("random OOM on an
    ordinary job") that looks nothing like its actual cause. See this file's
    module docstring.
    """
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive; must never mask the real error
        _log.warning("could not release VRAM after a failed load", exc_info=True)


async def _ensure_loaded(modality: str) -> ModelBackend:
    async with _slot.lock:
        if _slot.backend is not None and _slot.backend.modality == modality:
            return _slot.backend
        current = _slot.backend

    if current is not None:
        # Never unload out from under a still-running infer() call -- see
        # ModelSlot's docstring. This wait happens outside `_slot.lock` on
        # purpose: infer() itself holds no lock, so holding one here would
        # just block the wait pointlessly.
        await _wait_for_idle(current)

    async with _slot.lock:
        if _slot.backend is not None:
            _log.info("evicting %s to load %s", _slot.backend.modality, modality)
            _slot.backend.unload()
            _slot.backend = None
        backend_cls = _BACKENDS[modality]
        backend = backend_cls()
        _log.info("loading %s (%s)", backend.model_id, modality)
        started = time.monotonic()
        try:
            backend.load()
        except Exception:
            _log.exception(
                "failed to load %s (%s) -- releasing any partial VRAM claim",
                backend.model_id, modality,
            )
            _release_vram()
            raise
        _log.info("loaded %s in %.0fs", backend.model_id, time.monotonic() - started)
        _slot.backend = backend
        return backend


async def _run_job(job: Job) -> None:
    job.state = "running"
    try:
        backend = await _ensure_loaded(job.modality)
        # Run the blocking model call off the event loop -- this is the one
        # call in the whole server that can take tens of seconds (longer,
        # unbounded until GptOssChatBackend.MAX_NEW_TOKENS caps it, for
        # chat). `_slot.in_flight` brackets it so eviction knows to wait --
        # see ModelSlot's docstring.
        _slot.in_flight += 1
        try:
            result = await asyncio.to_thread(
                backend.infer, job.media_path, job.prompt, history=job.history
            )
        finally:
            _slot.in_flight -= 1
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
    media: UploadFile | None = File(None),
    modality: str = Form(...),
    prompt: str | None = Form(None),
    history: str | None = Form(None),
) -> JSONResponse:
    if modality not in _BACKENDS:
        raise HTTPException(400, f"unknown modality {modality!r}, expected one of {list(_BACKENDS)}")

    # Chat has no media to upload -- every other modality still requires one.
    media_path: Path | None = None
    if modality != "chat":
        if media is None:
            raise HTTPException(400, f"modality {modality!r} requires a media file")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        media_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{media.filename or 'upload'}"
        media_path.write_bytes(await media.read())

    job_id = uuid.uuid4().hex
    job = Job(job_id=job_id, modality=modality, media_path=media_path, prompt=prompt, history=history)
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
    submit -- see that function's docstring for the pull-based hand-off.

    Waits for any in-flight `infer()` call to clear first -- see
    `ModelSlot`'s docstring and this file's module docstring on why unloading
    out from under a still-running generation is the single highest-severity
    risk `/himonkey` introduces.
    """
    async with _slot.lock:
        current = _slot.backend
    if current is not None:
        await _wait_for_idle(current)
    async with _slot.lock:
        if _slot.backend is not None:
            _log.info("unloading %s (GPU hand-off to ComfyUI)", _slot.backend.modality)
            _slot.backend.unload()
            _slot.backend = None
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8189)
