"""The pod-side understanding + chat server: describes a photo/audio/video
clip, or answers a chat message.

Runs ON the pod, alongside ComfyUI, as a second always-resident process on a
second port (8189). Deposited and started by `deploy/pod_setup.sh` via
`runtime.session.provision()` -- see that module's docstring for why this is
a *second* file over the same one-file-at-a-time SSH transport rather than
being embedded inline.

**Lazy load/unload, one model at a time.** Only one of {moondream3-preview,
Qwen2-Audio-7B-Instruct, Qwen2.5-VL-7B-Instruct, gpt-oss-20b} is ever
resident in VRAM, and never at the same time as ComfyUI's own checkpoint --
the whole card is 24GB and none of these comfortably coexist with an H3/Flux
model (nor, for gpt-oss-20b specifically, with H3 itself: H3 alone measures
22.1-22.8GB peak on the actual RTX 4090 this project targets, so there is
never room regardless of gpt-oss-20b's ~16GB native-MXFP4 footprint). `POST
/submit` evicts whatever is currently loaded (if it is not what this request
needs) before loading the requested model; `POST /unload` is called by the
caller's pull-based GPU hand-off (`ai_studio.pipeline.residency`) right
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
import os
import subprocess
import sys
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

INFERENCE_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(ctx)s] %(message)s"
"""Byte-identical to `ai_studio.core.observability.HUMAN_FORMAT` -- this file
cannot import the package, so the format is duplicated and a unit test on the
host pins the two together. Before 2026-08-28 lines here had no timestamp at
all ("[inference] ..."), so a pulled inference.log could not be lined up
against the worker's trace."""


class _CtxFilter(logging.Filter):
    """`ctx` = the pod-side job id while one is running, else "-"."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.ctx = getattr(record, "ctx", None) or "-"
        return True


class _IsoFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds")


_handler = logging.StreamHandler()
_handler.setFormatter(_IsoFormatter(INFERENCE_LOG_FORMAT))
_handler.addFilter(_CtxFilter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
_log = logging.getLogger("inference_server")

UPLOAD_DIR = Path("/workspace/inference_uploads")
MAX_OUTPUT_CHARS = 1600
"""Per answer (📏 ~700 chars at max_new_tokens 384/512 for the audio/video
models). A runaway generation cannot return an unbounded response; the
caller applies its own, usually tighter, ceiling on top."""
MAX_JSON_OUTPUT_CHARS = 8000
"""For `json_only` jobs (the prompt rewriter): an H3 shot plan is well over
1000 characters, and truncating it mid-object would fail every conversion."""
MAX_NEW_TOKENS_CEILING = 1536
"""The most a caller may ask for; keeps UNLOAD_WAIT_TIMEOUT_S honest."""


@dataclass
class GenerationOptions:
    """Per-request knobs a caller may set on /submit. All optional; every
    backend that does not understand one ignores it.

    `system` is the harmony developer/system instruction block for gpt-oss
    (a chat persona; "you are a prompt engineer, reply with JSON" for the
    rewriter). `json_only` switches to greedy decoding, trims the reply to
    its outermost braces, and lifts the output cap to MAX_JSON_OUTPUT_CHARS.
    """

    system: str | None = None
    max_new_tokens: int | None = None
    reasoning_effort: str | None = None
    json_only: bool = False


# --------------------------------------------------------------- backends


class ModelBackend(Protocol):
    """One modality's load/infer/unload cycle. Implemented per model below."""

    modality: str
    model_id: str
    accepts_prompt: bool

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def infer(
        self,
        media_path: Path | None,
        prompt: str | None,
        *,
        history: str | None = None,
        options: GenerationOptions | None = None,
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

    def infer(
        self,
        media_path: Path | None,
        prompt: str | None,
        *,
        history: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        from PIL import Image

        assert media_path is not None  # only "chat" jobs ever omit it
        image = Image.open(media_path).convert("RGB")
        # English only. 📏 2026-08-27: asked three ways (a bilingual
        # instruction, a Traditional-Chinese question, "Answer in Traditional
        # Chinese only"), moondream3 answered in English every time -- the
        # model does not write Chinese, so the caption stays English.
        # Two skills, per the model docs: `caption(length=...)` for a
        # description (no question), `query(question, reasoning=True)` for
        # a specific one. A bare "describe this" through query() is the
        # weaker path; the long caption is what the model was trained to do.
        if prompt:
            result = self._model.query(image, prompt, reasoning=True)
            return str(result.get("answer", result) if isinstance(result, dict) else result)
        result = self._model.caption(image, length="long")
        return str(result.get("caption", result) if isinstance(result, dict) else result)


class Qwen2AudioBackend:
    """Qwen/Qwen2-Audio-7B-Instruct -- audio understanding, fp16, ~17GB.

    Replaces Qwen3-Omni-30B-A3B-Captioner, which does not fit a 24GB card on
    this stack (transformers 5 keeps its fused MoE experts in fp16 under both
    bitsandbytes and compressed-tensors -- see docs/model-qwen3-omni-
    captioner.md). Loader and inference follow the model card:
    `Qwen2AudioForConditionalGeneration` + `AutoProcessor`, chat template
    with one audio part, librosa at the feature extractor's sampling rate.
    Instruction-tuned, so it takes a prompt and answers in Chinese when
    asked to -- weaker at fine-grained captioning than the 30B model, but it
    runs.
    """

    modality = "audio"
    model_id = os.environ.get("AI_STUDIO_AUDIO_MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")
    accepts_prompt = True
    max_input_seconds = 30.0

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id, device_map="cuda", dtype=torch.float16
        )

    def unload(self) -> None:
        import torch

        self._model = None
        self._processor = None
        torch.cuda.empty_cache()

    def infer(
        self,
        media_path: Path | None,
        prompt: str | None,
        *,
        history: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        import librosa

        assert media_path is not None  # only "chat" jobs ever omit it
        if not prompt:
            raise ValueError("the audio modality needs a prompt; the caller supplies the question")
        question = prompt
        conversation = [{
            "role": "user",
            "content": [{"type": "audio", "audio_url": str(media_path)}, {"type": "text", "text": question}],
        }]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        sr = self._processor.feature_extractor.sampling_rate
        # Phone audio is usually .m4a; librosa 1.0 has no audioread fallback, so
        # soundfile fails with "Format not recognised" (📏 2026-08-27).
        # Transcode with the pod's ffmpeg to a mono WAV at the model's rate.
        wav = media_path.with_suffix(".16k.wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(media_path), "-ac", "1", "-ar", str(sr), "-f", "wav", str(wav)],
            check=True, capture_output=True,
        )
        audio, _ = librosa.load(str(wav), sr=sr, mono=True)
        inputs = self._processor(text=text, audio=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        out = self._model.generate(**inputs, max_new_tokens=384)
        out = out[:, inputs["input_ids"].shape[1]:]
        return str(self._processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]).strip()


class Qwen25VLVideoBackend:
    """Qwen/Qwen2.5-VL-7B-Instruct -- video understanding, fp16, ~17GB.

    Replaces Tarsier2-7b-0115, whose `TarsierForConditionalGeneration` lives
    only in ByteDance's repo pinned to transformers 4.47 (see docs/model-
    tarsier2.md). Loader and inference follow the model card:
    `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor`, the video read
    and sampled by `qwen_vl_utils.process_vision_info` (decord). `max_pixels`
    caps each sampled frame so a phone clip does not blow the token budget.
    """

    modality = "video"
    model_id = os.environ.get("AI_STUDIO_VIDEO_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    accepts_prompt = True
    MAX_PIXELS = int(os.environ.get("AI_STUDIO_VIDEO_MAX_PIXELS", str(360 * 420)))
    """Per sampled frame. 480*480+ improves grounding per the Qwen docs; an
    env var so the experiment is a restart, not a redeploy."""
    FPS = 1.0

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, device_map="cuda", dtype=torch.float16
        )

    def unload(self) -> None:
        import torch

        self._model = None
        self._processor = None
        torch.cuda.empty_cache()

    def infer(
        self,
        media_path: Path | None,
        prompt: str | None,
        *,
        history: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        assert media_path is not None  # only "chat" jobs ever omit it
        if not prompt:
            raise ValueError("the video modality needs a prompt; the caller supplies the question")
        question = prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{media_path}", "max_pixels": self.MAX_PIXELS, "fps": self.FPS},
                {"type": "text", "text": question},
            ],
        }]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # qwen-vl-utils 0.0.8 (the card's pin) returns two values; the
        # `return_video_kwargs=True` form is a later release (📏 2026-08-27:
        # "unexpected keyword argument").
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True,
            return_tensors="pt",
        ).to("cuda")
        out = self._model.generate(**inputs, max_new_tokens=512)
        out = out[:, inputs["input_ids"].shape[1]:]
        return str(self._processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]).strip()


class GptOssChatBackend:
    """openai/gpt-oss-20b -- plain-text chat, native MXFP4.

    `[speculative]` loader, same status as the three backends above: nothing
    here has run against real weights on this project's own hardware yet.
    See `docs/model-gpt-oss-20b.md`.

    gpt-oss-20b's chat template emits the "harmony" response format --
    separate analysis/commentary/final channels -- even with zero tool
    calling in play. `infer()` returns only the final channel via
    `_final_channel()`; a naive decode-and-return would leak the model's
    internal chain-of-thought into the reply. `apply_chat_template()`
    renders the harmony-formatted prompt for us, which is why nothing here
    needs the separate `openai-harmony` pip package.
    """

    modality = "chat"
    model_id = "openai/gpt-oss-20b"
    accepts_prompt = True

    MAX_NEW_TOKENS = 512
    """The primary defense against an unbounded generation (see this file's
    module docstring on why `/unload` racing a still-running `infer()` call
    is the single highest-severity risk the chat modality introduces). Sized well
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

    def infer(
        self,
        media_path: Path | None,
        prompt: str | None,
        *,
        history: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        opts = options or GenerationOptions()
        messages: list[dict[str, str]] = []
        if opts.system:
            # The HF gpt-oss chat template renders a system-role message as
            # the harmony *developer* "# Instructions" block and synthesises
            # the real system header (identity, cutoff, date, Reasoning:
            # <effort>) itself from `reasoning_effort`.
            messages.append({"role": "system", "content": opts.system})
        if history:
            # (role, content) pairs, oldest first -- see
            # the caller's rolling chat history, the
            # host-side store this was fetched from.
            for role, content in json.loads(history):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt or ""})

        # 📏 2026-08-27, transformers 5.16: apply_chat_template(return_tensors=
        # "pt") hands back a BatchEncoding (a dict), not a tensor -- passed
        # positionally to generate() it died with KeyError: 'shape'. Ask for
        # the dict explicitly and unpack it.
        # reasoning_effort="low" is a documented gpt-oss chat-template
        # kwarg. 📏 2026-08-27: at the default effort a one-line question
        # spent the whole 512-token budget inside the analysis channel and
        # never reached "final" -- the reply was thinking or nothing. A group
        # chat wants the short answer, not the deliberation.
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            reasoning_effort=opts.reasoning_effort or "low",
        ).to("cuda")
        max_new = min(opts.max_new_tokens or self.MAX_NEW_TOKENS, MAX_NEW_TOKENS_CEILING)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new}
        if opts.json_only:
            gen_kwargs["do_sample"] = False  # a schema wants the argmax, not a sample
        output_ids = self._model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        # Keep the special tokens: harmony's channel markers ARE special
        # tokens, and with skip_special_tokens=True the decode came back as
        # "analysis<thinking>assistantfinal<reply>" -- the leak _final_channel
        # exists to prevent (📏 first real generation, 2026-08-27).
        decoded = self._tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
        # The raw harmony transcript, for the log: the one place channel
        # routing problems show up (📏 2026-08-27: a JSON rewrite came back as
        # "想太久了" -- the fallback -- without the analysis budget being hit).
        _log.info("gpt-oss raw decode (%d chars): %s", len(decoded), decoded[:400].replace("\n", "\\n"))
        text = _final_channel(decoded)
        return _outer_json(text) if opts.json_only else text


def _outer_json(text: str) -> str:
    """The first balanced {...} object in a reply, or the reply unchanged if
    there is none -- the caller's JSON parser then fails loudly on it.

    Balanced-brace scan rather than first-`{`/last-`}`: a reply that trails
    off mid-object, or that puts prose with a brace after the JSON, would
    otherwise be sliced into something that is neither."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _final_channel(text: str) -> str:
    """Pull only harmony's "final" channel out of a raw gpt-oss-20b decode.

    📏 Verified against a real generation on the RTX 4090 (2026-08-27): the
    raw decode is `<|channel|>analysis<|message|>…<|end|><|start|>assistant
    <|channel|>final<|message|>…<|return|>` -- the final channel closes with
    `<|return|>` (end of turn), not `<|end|>`, so both are cut on. With the
    special tokens stripped the same text reads "analysis…assistantfinal…",
    which the second branch handles so a tokenizer change cannot leak the
    thinking trace. Falls back to the whole text only when neither shape is
    present -- returning nothing at all would be worse still.
    """
    marker = "<|channel|>final<|message|>"
    if marker in text:
        tail = text.split(marker, 1)[1]
        for stop in ("<|return|>", "<|end|>", "<|call|>"):
            tail = tail.split(stop, 1)[0]
        return tail.strip()
    stripped = "assistantfinal"
    if stripped in text:
        return text.rsplit(stripped, 1)[1].strip()
    if "<|channel|>analysis" in text or text.startswith("analysis"):
        # The budget ran out inside the thinking channel. Leaking that trace
        # as the reply is the failure this function exists to prevent; the
        # caller decides what to tell its user.
        raise ReasoningExhausted("the reply ended inside the analysis channel")
    return text.strip()


class ReasoningExhausted(Exception):
    """gpt-oss spent its whole token budget thinking and wrote no answer."""


_BACKENDS: dict[str, type[ModelBackend]] = {
    "image": MoondreamBackend,
    "audio": Qwen2AudioBackend,
    "video": Qwen25VLVideoBackend,
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
    options: GenerationOptions = field(default_factory=lambda: GenerationOptions())
    audio_prompt: str | None = None
    """Video only: the question for the audio model on the extracted track.
    The frame model gets `prompt`. Both come from the caller."""
    state: str = "queued"  # queued | running | completed | failed
    result_text: str | None = None
    """image / audio / chat: the one answer. None for video, which returns
    `result` instead."""
    result: dict[str, Any] | None = None
    """video: {"visual": str, "audio": str | None, "has_audio_track": bool}.
    Two models, two answers; how they are presented is the caller's."""
    truncated: bool = False
    """An answer hit MAX_OUTPUT_CHARS / MAX_JSON_OUTPUT_CHARS and was cut."""
    reasoning_exhausted: bool = False
    """chat: the model thought until its budget ran out and wrote nothing;
    `result_text` is "" and the caller words the apology."""
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
    the chat modality introduces, and `_wait_for_idle()` for the wait this field
    backs.
    """

    backend: ModelBackend | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    in_flight: int = 0
    load_s: float = 0.0
    """Seconds the resident backend took to load -- reported on its jobs."""


_slot = ModelSlot()
_jobs: dict[str, Job] = {}
_running_tasks: set[asyncio.Task[None]] = set()
"""Keeps a strong reference to every in-flight `_run_job` task -- asyncio
only holds a weak reference to a task created with `create_task`, so an
unreferenced one can be garbage-collected mid-run, silently killing the job.
Discarded via the task's own done-callback once it finishes."""

UNLOAD_WAIT_TIMEOUT_S = 180.0
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


RELEASE_CEILING_GIB = 1.0
"""What this process may still hold on the card after an unload and be
believed. Above it the weights are still referenced from somewhere, and the
only release that has ever worked is the process ending: 📏 2026-08-29,
twice in a row, `POST /unload` after a gpt-oss-20b chat session left this
process holding 12.7 GiB (nvidia-smi) with `_release_vram` run -- H3 then
had ~10 GiB and OOMed on the fourth clip, and on the second clip. Job 86 on
2026-08-27 was the same thing wearing moondream3's face."""


def _held_vram_gib() -> float:
    """What this process still has allocated on the card, in GiB. 0 without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 2**30
    except Exception:  # pragma: no cover - a measurement must never raise
        return 0.0


def _reexec() -> None:
    """Replace this process with a fresh copy of itself: same argv, same
    stdout (the nohup'd inference.log), same port once the old socket closes.
    Models are lazy-loaded, so nothing but the leak is lost."""
    _log.warning("restarting the inference server process to return its VRAM")
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _release_vram() -> None:
    """Best-effort VRAM release after a failed `backend.load()`.

    A partial load can claim CUDA memory before raising, and nothing else
    releases it on that path -- the next job of *any* kind can then fail to
    allocate VRAM it should have had, with a symptom ("random OOM on an
    ordinary job") that looks nothing like its actual cause. See this file's
    module docstring.
    """
    try:
        import gc

        import torch

        # gc first: `empty_cache()` only returns blocks no tensor references,
        # and a just-dropped HF model is held alive by reference cycles until
        # the collector runs. 📏 2026-08-27 (job 86): "evicting chat to load
        # image" set gpt-oss's refs to None, moondream3 then OOMed twice at
        # 22.7GiB in this process, and loaded on the third try once the
        # cycles had been collected in between.
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive; must never mask the real error
        _log.warning("could not release VRAM", exc_info=True)


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
            await asyncio.to_thread(_slot.backend.unload)
            _slot.backend = None
            # The evicted model must actually be gone before the next one
            # claims the card -- see _release_vram.
            await asyncio.to_thread(_release_vram)
        backend_cls = _BACKENDS[modality]
        backend = backend_cls()
        _log.info("loading %s (%s)", backend.model_id, modality)
        started = time.monotonic()
        # Off the event loop, like infer(): a cold load is 📏 ~60s for
        # moondream3 and minutes for Qwen3-Omni-Captioner (quantised on
        # load), and run inline it froze every route -- /poll returned
        # nothing at all for the whole load, so the client's 30s poll
        # timeout gave up on a job that was in fact fine (observed live
        # 2026-08-27, first request against real weights). The slot lock
        # is an asyncio.Lock, so holding it across the await is what
        # serialises loads without stalling /healthz and /poll.
        #
        # One retry, after a sweep: a load that OOMs because the previous
        # tenant's memory had not been collected yet succeeds on the next
        # attempt (📏 job 86 above), and paying that inside the server beats
        # failing the user's request and making them ask again.
        for attempt in (1, 2):
            try:
                await asyncio.to_thread(backend.load)
                break
            except Exception as exc:
                _log.exception(
                    "failed to load %s (%s), attempt %d -- releasing any partial VRAM claim",
                    backend.model_id, modality, attempt,
                )
                backend = backend_cls()
                await asyncio.to_thread(_release_vram)
                if attempt == 2 or "out of memory" not in str(exc).lower():
                    raise
        _slot.load_s = time.monotonic() - started
        _log.info("loaded %s in %.0fs", backend.model_id, _slot.load_s)
        _slot.backend = backend
        return backend


def _has_audio_track(path: Path) -> bool:
    """ffprobe: does the container carry an audio stream? A silent clip must
    skip the audio pass and say so, not run the model on nothing."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode == 0 and "audio" in proc.stdout


def _extract_track(path: Path) -> Path:
    wav = path.with_suffix(".track.wav")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", str(wav)],
        check=True, capture_output=True,
    )
    return wav


async def _infer_with(modality: str, media_path: Path | None, prompt: str | None, job: Job) -> str:
    """Load `modality`'s backend (evicting whatever is resident) and run one
    inference off the event loop, bracketed by `in_flight` -- see ModelSlot.
    Logs one line per inference: modality, load and infer seconds, VRAM after."""
    was_resident = _slot.backend is not None and _slot.backend.modality == modality
    backend = await _ensure_loaded(modality)
    _slot.in_flight += 1
    started = time.monotonic()
    try:
        return await asyncio.to_thread(
            backend.infer, media_path, prompt, history=job.history, options=job.options
        )
    finally:
        _slot.in_flight -= 1
        infer_s = round(time.monotonic() - started, 1)
        try:
            import torch

            vram_gb = round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
        except Exception:
            vram_gb = None
        _log.info(
            "job done: %s load=%.0fs infer=%.1fs vram=%sGB", modality,
            0.0 if was_resident else _slot.load_s, infer_s, vram_gb,
            extra={"ctx": f"job={job.job_id}"},
        )


def _cap(job: Job, text: str) -> str:
    cap = MAX_JSON_OUTPUT_CHARS if job.options.json_only else MAX_OUTPUT_CHARS
    if len(text) > cap:
        job.truncated = True
    return text[:cap]


async def _run_job(job: Job) -> None:
    job.state = "running"
    try:
        if job.modality == "video" and job.media_path is not None:
            # The video modality sees AND hears: Qwen2.5-VL only samples
            # frames (process_vision_info), so on its own it describes a
            # silent film. Run it on the frames with `prompt`, then -- if the
            # clip has a track -- evict to Qwen2-Audio and run that on the
            # ffmpeg-extracted audio with `audio_prompt` (one swap, 📏 ~27 s).
            # Two answers go back separately; the caller presents them.
            visual = await _infer_with("video", job.media_path, job.prompt, job)
            has_track = _has_audio_track(job.media_path)
            heard: str | None = None
            if has_track:
                track = await asyncio.to_thread(_extract_track, job.media_path)
                heard = await _infer_with("audio", track, job.audio_prompt, job)
            job.result = {
                "visual": _cap(job, visual.strip()),
                "audio": _cap(job, heard.strip()) if heard is not None else None,
                "has_audio_track": has_track,
            }
        else:
            try:
                job.result_text = _cap(job, await _infer_with(job.modality, job.media_path, job.prompt, job))
            except ReasoningExhausted:
                job.result_text = ""
                job.reasoning_exhausted = True
        job.state = "completed"
    except Exception as exc:  # a job's own failure must not crash the server
        _log.exception("job %s failed", job.job_id)
        job.error = str(exc)
        job.state = "failed"
    if job.state == "failed":
        # Outside the `except` block on purpose. Observed live 2026-08-27:
        # Qwen3-Omni OOMed half-way through load() and 23.9GB stayed
        # allocated until the process was restarted, even though
        # _ensure_loaded had called _release_vram() -- inside an `except`,
        # the in-flight exception's traceback still references load()'s
        # frames, and those hold the half-built model, so gc cannot free
        # it. Here the exception has been dropped and the sweep can work.
        _release_vram()


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
    audio_prompt: str | None = Form(None),
    history: str | None = Form(None),
    system: str | None = Form(None),
    max_new_tokens: int | None = Form(None),
    reasoning_effort: str | None = Form(None),
    json_only: bool = Form(False),
) -> JSONResponse:
    if modality not in _BACKENDS:
        raise HTTPException(400, f"unknown modality {modality!r}, expected one of {list(_BACKENDS)}")
    # The question is the caller's. This server holds no default wording:
    # image without a prompt is moondream's caption path (a model skill, not
    # copy); audio and video need theirs, and video needs one per model.
    if modality in ("audio", "video", "chat") and not prompt:
        raise HTTPException(400, f"modality {modality!r} requires a prompt")
    if modality == "video" and not audio_prompt:
        raise HTTPException(400, "modality 'video' requires an audio_prompt for the audio model")
    if modality != "video" and audio_prompt:
        raise HTTPException(400, f"audio_prompt is only meaningful for modality 'video', not {modality!r}")

    # Chat has no media to upload -- every other modality still requires one.
    media_path: Path | None = None
    if modality != "chat":
        if media is None:
            raise HTTPException(400, f"modality {modality!r} requires a media file")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        media_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{media.filename or 'upload'}"
        media_path.write_bytes(await media.read())

    job_id = uuid.uuid4().hex
    if reasoning_effort not in (None, "low", "medium", "high"):
        raise HTTPException(400, f"reasoning_effort must be low|medium|high, got {reasoning_effort!r}")
    options = GenerationOptions(
        system=system or None,
        max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS_CEILING) if max_new_tokens else None,
        reasoning_effort=reasoning_effort,
        json_only=json_only,
    )
    job = Job(
        job_id=job_id, modality=modality, media_path=media_path, prompt=prompt,
        audio_prompt=audio_prompt, history=history, options=options,
    )
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
    return {
        "state": job.state,
        "result_text": job.result_text,
        "result": job.result,
        "truncated": job.truncated,
        "reasoning_exhausted": job.reasoning_exhausted,
        "error": job.error,
    }


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
    card. Called by `ai_studio.pipeline.residency.make_room_for` before every generation
    submit -- see that function's docstring for the pull-based hand-off.

    Waits for any in-flight `infer()` call to clear first -- see
    `ModelSlot`'s docstring and this file's module docstring on why unloading
    out from under a still-running generation is the single highest-severity
    risk the chat modality introduces.
    """
    async with _slot.lock:
        current = _slot.backend
    if current is not None:
        await _wait_for_idle(current)
    async with _slot.lock:
        if _slot.backend is not None:
            modality = _slot.backend.modality
            _log.info("unloading %s (GPU hand-off to ComfyUI)", modality)
            await asyncio.to_thread(_slot.backend.unload)
            _slot.backend = None
            await asyncio.to_thread(_release_vram)
            held = await asyncio.to_thread(_held_vram_gib)
            _log.info("unloaded %s; %.2f GiB still allocated in this process", modality, held)
            if held > RELEASE_CEILING_GIB:
                # Answer first, then go: the caller only needs the card, and
                # it needs it now. See RELEASE_CEILING_GIB.
                _log.warning(
                    "%.2f GiB still held after unloading %s (ceiling %.1f); the weights are "
                    "still referenced and the card is not free -- restarting this process",
                    held, modality, RELEASE_CEILING_GIB,
                )
                asyncio.get_running_loop().call_later(0.5, _reexec)
                return {"ok": True, "held_gib": round(held, 2), "restarting": True}
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    # log_config=None: uvicorn's dictConfig would layer its own untimestamped
    # handlers on top of the one above and interleave two formats in
    # inference.log (observed 2026-08-27). access_log off: every job is one
    # line already, and the worker polls every few seconds.
    uvicorn.run(app, host="0.0.0.0", port=8189, log_config=None, access_log=False)
