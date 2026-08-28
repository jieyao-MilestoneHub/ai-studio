# moondream3-preview

The image-understanding model behind the LINE bot's `/說圖` trigger — a
photo in, a text description out. Run as a lazily-loaded backend inside
`deploy/inference_server.py` on the same RunPod pod as MiniMax H3 and
Flux.1-dev, **not** through ComfyUI: this is a `transformers`-loaded model
with its own custom `trust_remote_code=True` modeling code, not a ComfyUI
graph. See [line-bot.md](line-bot.md) for the trigger and
[architecture.md](architecture.md) for the third `UnderstandingProvider`
quartet this backs.

> ⚠️ **Licence: not yet verified.** This project has not confirmed
> moondream3-preview's licence terms (commercial-use restrictions,
> redistribution terms, or any geographic exclusion). Moondream's earlier
> releases have shipped under Apache-2.0, which would impose no restriction
> at all — but "preview" releases from a model family sometimes carry
> different terms than the GA release, and that has not been checked for
> this specific model. **Confirm the actual licence on HuggingFace before
> any real deployment**, the same discipline `model-h3.md` and
> `model-flux.md` already apply to their models. Until checked, do not
> assume `runtime.pod.LICENCE_SAFE_DATACENTERS` (which encodes MiniMax H3's
> US/EU/UK/KR exclusion specifically) says anything about this model's
> geographic restrictions, if it has any.

All figures below are `[reported]` (from the model card / community
benchmarks) or `[speculative]` (inferred, not measured) — nothing in this
project has run moondream3 on any hardware yet. See
[attribution.md](attribution.md) on number honesty.

## Weights

`[reported]`, official precision:

| component | size | notes |
|---|---|---|
| moondream3-preview, full weights | `[reported]` >16GB | heavier than the "tiny" branding suggests — see the fallback note below |

**If VRAM-constrained, fall back to `microsoft/Florence-2-large`** (<2GB
`[reported]`, near-zero marginal cost on a card already running H3/Flux).
Florence-2 uses a different call shape (`AutoProcessor` + a task prompt like
`"<MORE_DETAILED_CAPTION>"` rather than moondream's `.query(image,
question)` method) — swapping `MoondreamBackend`'s `MODEL_ID` alone is not
enough; `deploy/inference_server.py`'s `infer()` body needs to change too if
this fallback is taken.

## Host requirements

| requirement | value | why |
|---|---|---|
| GPU | shared with H3/Flux — RTX 4090 24GB minimum, per the existing ladder | one pod, one card; understanding and generation never run concurrently (see below) |
| VRAM headroom | `[speculative]` — moondream3 alone is `[reported]` >16GB, which is most of a 24GB card on its own | unmeasured: whether the eviction hand-off with ComfyUI (`ComfyClient.free_memory()` / `InferenceClient.unload()`, see `docs/schedule.md`'s "GPU hand-off" section) leaves any transient double-residency that could OOM |
| CUDA | same pod, same template as H3 — see [model-h3.md](model-h3.md) | `deploy/inference_server.py` runs inside ComfyUI's own `.venv-cu128`, reusing its torch+CUDA build rather than a fresh venv |
| Disk | 📏 47.6GB — the whole repo, into the standard HF cache (`HF_HOME=/workspace/.hf`) | its `model.safetensors.index.json` maps every weight to the four `modelv2-*` shards (18.5GB); `model-0000*` (18.5GB, an older shard set) and `model_fp8.pt` (10.5GB) are kept anyway by decision |

**Unverified dependency risk**: `deploy/inference_server.py` pip-installs
`transformers`/`accelerate`/`bitsandbytes` into ComfyUI's own venv rather
than a separate one. Whether ComfyUI's own pinned dependency versions (if
any) conflict with what moondream3's `trust_remote_code=True` modeling code
needs has not been checked — run `pip check` in that venv on the first real
deployment before trusting it.

## Settings

- **Prompt**: two skills, per the model docs. A bare `/說圖` sends no
  question and `MoondreamBackend.infer()` runs `caption(image,
  length="long")`; `/說圖 <question>` is rewritten by gpt-oss on the pod
  into one specific English question (`fun_workflow/prompts/understanding.py`) and runs
  `query(image, question, reasoning=True)`. **English only** — 📏 asked
  three ways on 2026-08-27 it never wrote Chinese, so the LINE delivery is
  prefixed 「(moondream3 只能用英文描述)」 rather than paying a second model
  swap to translate.
- **Output length**: capped at `UnderstandingCapabilities.max_output_chars`
  (1000, `[speculative]`) on the ai-studio side, and again at
  `MAX_OUTPUT_CHARS` inside `deploy/inference_server.py` — a runaway
  generation must not return an unbounded response either side of the wire.

## The one technical risk, and how it gets checked

Whether `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
actually resolves to a `.query(image, question)`-shaped model object is
`[speculative]` — moondream's exact custom-code API surface has not been
exercised on this project's own hardware. The check is cheap and should be
the first thing run against a live pod, before any real `/說圖` request:

```bash
uv run ai-studio understand <sample.jpg> --kind image --provider understand-image
```

If the model's actual API differs from what `MoondreamBackend.infer()`
assumes, this fails loudly with the model's own exception rather than
producing a plausible-looking but wrong description — there is no silent
degrade path here by design (see `core/errors.py`).

## What we have measured ourselves, and what we have not

📏 **Measured on our own account**

*(nothing yet — this section is intentionally empty pending a real run,
exactly the state `docs/model-h3.md` was in before its own first measured
session)*

⛔ **Still unmeasured — do not treat any number in this file as ours**

- Actual on-disk / in-VRAM footprint once loaded.
- Inference latency per description on any of the ladder's GPUs.
- Whether the fallback to Florence-2-large is ever actually needed in
  practice, or whether moondream3's real footprint fits comfortably enough
  alongside the GPU hand-off's eviction discipline.
- The licence terms flagged above.
