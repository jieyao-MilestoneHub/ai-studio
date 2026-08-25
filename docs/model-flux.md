# Flux.1-dev

The image-generation model behind the LINE bot's `畫圖`/`/img` trigger. Run
through ComfyUI on the same RunPod pod as MiniMax H3 — see
[line-bot.md](line-bot.md) for the dual-trigger design and [runpod.md](runpod.md)
for what sharing a pod costs in download time.

> ⚠️ **Licence: Flux.1-dev is non-commercial.** Black Forest Labs ships `dev`
> under the FLUX.1 [dev] Non-Commercial License — a separate commercial licence
> is required for commercial use. `Flux.1-schnell` is Apache-2.0 with no such
> restriction. This project uses `dev` (chosen for image quality over schnell's
> speed); **confirm the bot's actual use is personal/internal before shipping
> this to any commercial context**, and switch to schnell if it is not. Unlike
> MiniMax H3's geographic exclusion (US/EU/UK/KR, enforced by
> `runtime.pod.LICENCE_SAFE_DATACENTERS`), nothing in this codebase currently
> checks or enforces the *commercial-use* boundary — it is a human decision,
> not a code path, and belongs here rather than being silently assumed away.

All figures below are `[speculative]` — nothing in this project has run
Flux on any hardware yet, unlike `docs/model-h3.md`'s `[reported]`/📏 figures.
See [attribution.md](attribution.md) on number honesty. Re-measure on the
actual pod and promote to 📏 once it has actually run.

## Weights

`[speculative]`, depends on which fp8 quantisation is used for the transformer:

| component | size | notes |
|---|---|---|
| Flux.1-dev transformer, fp8 | ~12 GB | a full fp16 `flux1-dev.safetensors` alone is ~23.8GB; a properly scaled fp8 checkpoint is roughly half |
| T5-XXL text encoder, fp8 | ~5 GB | shared shape with Flux.1-schnell |
| CLIP-L | ~0.25 GB | |
| VAE | ~0.16 GB | |
| **Total** | **~17–23 GB** | use the low end for planning; verify on first real run |

Combined with H3's measured **54.7 GB** int8 working set (`docs/model-h3.md`),
a window open now downloads **~72–78 GB** total, not ~54.7 GB — see
[runpod.md](runpod.md) for the updated no-network-volume math this implies.

## Host requirements

| requirement | value | why |
|---|---|---|
| GPU | shared with H3 — RTX 4090 24GB minimum, per the existing ladder | one pod, one ComfyUI instance serves both models |
| VRAM headroom | `[speculative]` — Flux.1-dev fp8 needs meaningfully less than H3's measured 43.3GB peak, but the two models are never loaded simultaneously (ComfyUI swaps between jobs), so peak VRAM is whichever model is active, not the sum | unmeasured: whether ComfyUI's own model-swap overhead adds meaningful VRAM pressure during the swap itself is untested |
| CUDA | 13.0, same template as H3 | no separate template; same "Official Runpod ComfyUI, CUDA 13" pod |
| Disk | shares H3's 200GB container disk, no network volume | see [runpod.md](runpod.md) |

## Settings

- **Steps**: `DEFAULT_STEPS = 28` in `providers/flux.py` is a `[speculative]`
  placeholder for `dev` — schnell needs 1–4 steps (distilled for few-step
  generation), `dev` needs meaningfully more. Re-measure and tune once this
  has actually run.
- **Sampler**: standard `KSamplerSelect` + `BasicScheduler` +
  `SamplerCustomAdvanced` graph (`workflows/flux_dev.json`) — **not** a
  turbo/lightning-distilled path, so none of H3's turbo-trap caveats
  (`docs/model-h3.md`, `comfy/validate.py`) apply to this workflow at all.
  `uses_turbo_lora()` only fires on `MiniMaxH3TurboLoRA` or a stock LoRA loader
  carrying a turbo/lightning hint string, neither of which this graph has.
- **Resolution**: 1024×1024 default (`flux_capabilities()`'s default), a
  natural fit for Flux's native training resolution — unlike H3, where
  resolution was measured to have no effect on quality (`docs/model-h3.md`),
  no equivalent measurement exists yet for Flux at other resolutions.

## Prompting

Deliberately **not** a structured schema like H3's (`prompts/h3.py`). H3's
schema exists because it is *measured* to matter — 26.0 → 367.6 on the same
scene, changing only the prompt (`docs/model-h3.md`). No equivalent published
or measured schema advantage exists for Flux, which takes plain
natural-language T5/CLIP prose. `prompts/flux.py` is accordingly thin: strip,
collapse whitespace, truncate to `max_chars`, refuse empty input — no LLM call,
which is also why the image path adds $0 to the LLM serverless cost line in
[line-bot.md](line-bot.md#cost).

## What we have measured ourselves, and what we have not

📏 **Measured on our own account**

*(nothing yet — this section is intentionally empty pending a real run,
exactly the state `docs/model-h3.md` was in before its own first measured
session)*

⛔ **Still unmeasured — do not treat any number in this file as ours**

- Actual on-disk footprint of the fp8 transformer + text encoder set.
- Generation time per image at 1024×1024 on any of the ladder's GPUs.
- Peak VRAM during a Flux job specifically, and whether swapping between H3 and
  Flux mid-window costs meaningful extra time or has any failure mode of its
  own (a stuck model swap, an OOM from transient double-residency).
- Whether 28 steps is the right default for `dev`'s quality/latency trade-off
  on this hardware, or whether it should move toward schnell's few-step regime
  if turnaround matters more than per-image quality in practice.
