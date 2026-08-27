# Qwen3-Omni-30B-A3B-Captioner

The audio-understanding model behind the LINE bot's `/說音` trigger — an
audio clip in, a text description/transcript out. Run as a lazily-loaded
backend inside `deploy/inference_server.py` on the same RunPod pod as
MiniMax H3, Flux.1-dev, and the other two understanding models. See
[line-bot.md](line-bot.md) for the trigger and [architecture.md](architecture.md)
for the `UnderstandingProvider` quartet.

> ⚠️ **Licence: not yet verified.** This project has not confirmed this
> specific model's licence terms. Qwen models have historically shipped
> under a mix of Apache-2.0 and the Qwen-specific "Tongyi Qianwen" licence
> depending on size/variant, and a 30B MoE ("A3B") captioner variant has not
> been checked against either. **Confirm the actual licence on HuggingFace
> before any real deployment.** Until checked, do not assume
> `runtime.pod.LICENCE_SAFE_DATACENTERS` (MiniMax H3-specific) says anything
> about this model's restrictions, if it has any.

All figures below are `[reported]` or `[speculative]` — nothing here has run
on this project's own hardware yet.

## Two hard limits, enforced before the model ever sees the input

These are not settings to tune; they are stated constraints from the
model's own card, and both are enforced on the ai-studio side before a
request reaches this model at all:

1. **No text prompt.** `UnderstandingCapabilities.accepts_prompt = False`
   for this modality (`providers/understanding.py`), checked by
   `check_prompt()` and raised loudly rather than silently dropped if a
   caller ever supplies one. `deploy/inference_server.py`'s
   `Qwen3OmniCaptionerBackend.infer()` also never passes `prompt` into the
   model call, as a second line of defence.
2. **≤30 seconds of audio per call.** `Settings.max_audio_understand_s`
   (default 30.0) is checked in `bots/line/webhook.py`'s `_remember_audio()`
   against LINE's own reported `duration` field, **before** the clip is even
   downloaded — not merely before it reaches the GPU.

## Weights

`[reported]`, and the one place this doc's numbers diverge from the other
two model-cards' pattern:

| component | size | notes |
|---|---|---|
| Qwen3-Omni-30B-A3B-Captioner, full precision | `[reported]` ~60GB | downloaded in full by `deploy/pod_setup.sh` |
| resident in VRAM after 4-bit quantization | `[reported]` ~17-22GB | quantized **at load time**, not pre-quantized on disk |

**Deliberately not pre-quantized to a separate GGUF/AWQ repo.** This
project has not confirmed a separately-published Q4 build of this specific
captioner variant exists. `deploy/inference_server.py`'s
`Qwen3OmniCaptionerBackend.load()` instead downloads the full-precision
repo and applies `transformers.BitsAndBytesConfig(load_in_4bit=True)` on
load — trading a larger one-time download (~60GB vs. a hypothetical Q4
artifact's smaller size) for not depending on an unverified third party's
requantization. **If a genuine GGUF build turns out to exist and is
preferred**, `deploy/pod_setup.sh`'s `dl_repo` call and
`inference_server.py`'s loader both need to switch to `llama-cpp-python`
bindings instead — a real, not-yet-done piece of follow-up work if the
60GB download proves too costly in practice.

## Host requirements

| requirement | value | why |
|---|---|---|
| GPU | shared with H3/Flux/moondream3/Tarsier2 — RTX 4090 24GB minimum | one pod, one card; see the GPU hand-off in [schedule.md](schedule.md) |
| VRAM headroom | `[speculative]` ~17-22GB resident, the heaviest of the three understanding models | leaves the least headroom of the three against a 24GB ceiling — the first candidate to investigate if VRAM pressure shows up during the hand-off |
| CUDA | same pod, same template as H3 | `deploy/inference_server.py` runs inside ComfyUI's own `.venv-cu128` |
| Disk | ~60GB, full repo, into the standard HF cache | see `deploy/pod_setup.sh`'s disk-headroom check (~142GB total across all weight sets) |

## The two technical risks, and how they get checked

1. **The exact `transformers` model class is not pinned.**
   `Qwen3OmniCaptionerBackend.load()` uses the generic
   `AutoModelForCausalLM` as a fallback; if the model's own README specifies
   a dedicated class (as several recent Qwen multimodal releases do), that
   needs to replace the generic loader. This fails loudly (an exception on
   load) rather than silently producing a wrong result if the generic class
   cannot actually run this model's forward pass.
2. **Whether `AutoProcessor` on this repo actually accepts a raw waveform +
   sample rate** (the call shape `infer()` assumes) is unverified. Check
   both against a live pod before trusting `/說音` in production:

   ```bash
   uv run ai-studio understand <sample.m4a> --kind audio --provider understand-audio
   ```

## Prompting

There is none, by design (see the hard limits above) — the model's own
model card is explicit that it does not accept a steering prompt, which is
also why `/說音` is the one of the three describe triggers where "no
trailing text" is not merely a consistency choice but a hard requirement.

## What we have measured ourselves, and what we have not

📏 **Measured on our own account**

*(nothing yet — pending a real run)*

⛔ **Still unmeasured — do not treat any number in this file as ours**

- Actual VRAM peak during 4-bit-quantized inference on this hardware.
- Generation latency per audio clip.
- Whether 4-bit quantization degrades caption quality enough to matter for
  a personal-use bot, versus 8-bit or full precision (VRAM permitting).
- Whether the exact wire shape `infer()` assumes (raw waveform in,
  `batch_decode` out) matches this model's real processor/generate contract.
- The licence terms flagged above.
