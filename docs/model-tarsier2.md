# Tarsier2-7b-0115

The video-understanding model behind the LINE bot's `/說影` trigger — a
video clip in, a text description out. Run as a lazily-loaded backend
inside `deploy/inference_server.py`, alongside MiniMax H3, Flux.1-dev,
moondream3-preview, and Qwen3-Omni-Captioner on the same pod. See
[line-bot.md](line-bot.md) for the trigger and [architecture.md](architecture.md)
for the `UnderstandingProvider` quartet.

> ⚠️ **Licence: not yet verified.** This project has not confirmed
> `omni-research/Tarsier2-7b-0115`'s licence terms (commercial-use
> restrictions, redistribution terms, or any geographic exclusion).
> **Confirm the actual licence on HuggingFace before any real deployment.**
> Until checked, do not assume `runtime.pod.LICENCE_SAFE_DATACENTERS`
> (MiniMax H3-specific) says anything about this model's restrictions, if it
> has any.

All figures below are `[reported]` or `[speculative]` — nothing here has
run on this project's own hardware yet.

## Weights

`[reported]`, FP16 (no quantization — the lightest of the three
understanding models, so none is needed to fit a 24GB card):

| component | size | notes |
|---|---|---|
| Tarsier2-7b-0115, FP16 | `[reported]` ~14-16GB | dense 7B; the smallest resident footprint of the three understanding models |

## Host requirements

| requirement | value | why |
|---|---|---|
| GPU | shared with H3/Flux/moondream3/Qwen3-Omni-Captioner — RTX 4090 24GB minimum | one pod, one card; see the GPU hand-off in [schedule.md](schedule.md) |
| VRAM headroom | `[speculative]` ~14-16GB resident — the most headroom of the three against a 24GB ceiling | least likely of the three to be the cause if VRAM pressure shows up during the hand-off, but that is an expectation, not a measurement |
| CUDA | same pod, same template as H3 | `deploy/inference_server.py` runs inside ComfyUI's own `.venv-cu128` |
| Disk | full repo, into the standard HF cache | see `deploy/pod_setup.sh`'s disk-headroom check |

## The one unresolved number: input length

**No source consulted while building this feature states a length cap for
Tarsier2.** `Settings.max_video_understand_s` (default 120.0) exists as a
generous, deliberately `[speculative]` placeholder in
`bots/line/webhook.py`'s `_remember_video()` — enforced the same way the
audio cap is (checked against LINE's own reported `duration` field before
the clip is downloaded), but the number itself has not been benchmarked
against what Tarsier2 actually tolerates or costs per second of dense video
understanding on this hardware. **Do not raise it without measuring first**:
dense video-LM inference cost typically scales with frame count, and an
unbounded video accepted here could be expensive in both wall-clock and
VRAM in a way none of the other two understanding models are exposed to
(image is a single frame; audio is capped at 30s by the model's own limit).

## The one technical risk, and how it gets checked

Tarsier ships custom `trust_remote_code=True` modeling code with a
chat-style video-QA interface; the exact `AutoProcessor(text=..., videos=...)`
call shape `Tarsier2Backend.infer()` assumes is `[speculative]` and has not
been exercised against the real repo. Check before trusting `/說影` in
production:

```bash
uv run ai-studio understand <sample.mp4> --kind video --provider understand-video
```

If the real processor signature differs (a different keyword for the video
path, a required frame-sampling parameter, etc.), this fails loudly with
the model's own exception rather than a plausible-looking wrong description.

## Prompting

`/說影` never sends a steering prompt (see [line-bot.md](line-bot.md)'s
uniform "none of the three take trailing text" rule), even though Tarsier2's
own chat interface could technically accept one. `Tarsier2Backend.infer()`
falls back to `"Describe what happens in this video in detail."` when
`prompt` is `None`. A future CLI caller could supply one via
`UnderstandingRequest.prompt`, which the LINE path deliberately never does.

## 📏 2026-08-27: not loadable by transformers alone

The checkpoint's `architectures` is `TarsierForConditionalGeneration` with
`model_type: llava` over a `qwen2_vl` text/vision config — a custom class
that lives in ByteDance's `tarsier` repo (branch `tarsier2`), not in
transformers; `AutoModelForCausalLM` refuses it ("Unrecognized configuration
class LlavaConfig"). That repo pins `transformers==4.47.0` (plus `decord`,
`triton==2.2.0`); the pod runs 5.16.1 for gpt-oss-20b's MXFP4. Running it
means a second venv and a second server process, or a different video
model that transformers 5 loads natively. Also confirmed: the repo is gated
(`gated: auto`) — accept its terms once and set `HF_TOKEN`, which
`deploy/pod_setup.sh` now requires before downloading.

## What we have measured ourselves, and what we have not

📏 **Measured on our own account**

*(nothing yet — pending a real run)*

⛔ **Still unmeasured — do not treat any number in this file as ours**

- Actual VRAM peak during inference on this hardware, and whether it holds
  at longer input lengths than briefly tested.
- Generation latency per video clip, and how it scales with clip length —
  the input this project needs to determine `MAX_VIDEO_UNDERSTAND_S`.
- Whether the exact wire shape `infer()` assumes matches this model's real
  processor/generate contract.
- The licence terms flagged above.
