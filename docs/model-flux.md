# Flux.1-dev

The image-generation model behind the LINE bot's `/圖片` trigger. Run
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
| NSFW LoRA | **0.69 GB** | 687,476,088 bytes, HF blobs API 📏 — the one measured row in this table |
| **Total** | **~17.7–23.7 GB** | use the low end for planning; verify on first real run |

Combined with H3's measured **54.7 GB** int8 working set (`docs/model-h3.md`),
a window open now downloads **~72.7–78.7 GB** total, not ~54.7 GB — see
[runpod.md](runpod.md) for the updated no-network-volume math this implies.
Only the LoRA's 0.69 GB in that range is measured; the rest is still
`[speculative]` and must not be promoted along with it.

## Host requirements

| requirement | value | why |
|---|---|---|
| GPU | shared with H3 — RTX 4090 24GB minimum, per the existing ladder | one pod, one ComfyUI instance serves both models |
| VRAM headroom | `[speculative]` — Flux.1-dev fp8 needs meaningfully less than H3's measured 43.3GB peak, but the two models are never loaded simultaneously (ComfyUI swaps between jobs), so peak VRAM is whichever model is active, not the sum | unmeasured: whether ComfyUI's own model-swap overhead adds meaningful VRAM pressure during the swap itself is untested |
| CUDA | same template as H3 — see [model-h3.md](model-h3.md)'s host requirements for the current template id and the open CUDA-version question | no separate template; same pod |
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

## The NSFW LoRA

`Heartsync/Flux-NSFW-uncensored` / `lora.safetensors`, downloaded by
`deploy/pod_setup.sh` and **renamed on the pod** to
`flux_nsfw_uncensored_v1.safetensors` — the name `workflows/flux_dev.json`
loads. `lora.safetensors` says nothing in a directory that also holds the H3
turbo LoRA, and two strings that have to match are safer when they are the same
string.

| item | value | basis |
|---|---|---|
| base model | `black-forest-labs/FLUX.1-dev` | HF metadata 📏 |
| licence | `creativeml-openrail-m` | HF metadata 📏 |
| file size | **687,476,088 bytes (~0.69 GB)** | HF blobs API 📏 |
| likes | 474, tagged `not-for-all-audiences` | HF metadata 📏 |
| last updated | 2025-05-05 | HF metadata 📏 |
| **trigger word** | **none needed** | neither candidate's model card declares an `instance_prompt` 📏 — this is an "unrestrain" adapter, not a concept LoRA |
| card's suggested settings | guidance **7.0**, steps **28**, 1024×1024, fp16 | model card `[reported]` |
| tensor key format | `transformer.single_transformer_blocks.N.attn.to_k.lora_A/lora_B.weight` = **diffusers/PEFT** | read from the safetensors header 📏 |

`enhanceaiteam/Flux-uncensored`'s `lora.safetensors` is **byte-identical** in
its first 8KB 📏 — the same weight uploaded twice. Heartsync is the one used
because its metadata is complete (licence, base model and content tags all
declared), it has 474 likes, and it has not moved since 2025-05.

### The one technical risk, and how it gets checked

Whether ComfyUI loads a **diffusers-format** Flux LoRA depends on its key
mapping table. For Flux.1-dev it is supported in general, but **nobody has
verified this particular file on this project** `[speculative]`.

The check is cheap and it is the first thing Phase 7.1 does, before any real
request is sent:

```bash
grep -i "lora key not loaded" <comfyui log>     # must print nothing
```

**One grep decides it. Do not look at a picture and guess** — an adapter that
half-loads produces a plausible image, and so does one that does not load at
all.

### Why nodes "8" *and* "10" both point at the LoRA

`UNETLoader` ("1") feeds **two** consumers in this graph: `BasicGuider` ("8")
and `BasicScheduler` ("10"). The LoRA node ("14") is inserted between, and
**both** consumers were rewired to it.

Rewiring only "10" and forgetting "8" is the failure this repo cares about
most: the LoRA would have **no effect at all and raise no error**. `grep` finds
the file, ComfyUI loads it, the graph is valid, the picture is fine — and the
adapter is not in the path that guides sampling. `tests/unit/test_graph_validate.py`
asserts the structure directly rather than trusting this paragraph, and Phase
7.1's second check (same seed, `lora_strength` 1.0 and 0.0, two images that
must differ) catches it live.

`lora_strength` is a workflow binding set explicitly on every submission by
`providers/flux.py` rather than left to the JSON's own default, for the same
reason: it is what makes that A/B a one-line change instead of a graph edit.

### Guidance 3.5 or 7.0 — unresolved, deliberately

The graph uses `FluxGuidance.guidance = 3.5`, the common value for Flux.1-dev.
The LoRA's model card suggests **7.0** `[reported]`. Steps agree at 28; the
guidance figure is off by a factor of two.

This is **not** being changed on paper. Both are defensible, the difference is
visible rather than arguable, and two images at the same seed cost seconds of
GPU time — so it is Phase 7.1's third check, and whichever wins gets written
back into the workflow with a date.

## Image-to-image (`/圖圖`)

Flux.1-dev does img2img with no extra weights: the photo is VAE-encoded into
the starting latent and sampling begins part-way down the schedule instead of
from pure noise. `workflows/flux_dev_i2i.json` is that graph; `denoise` is the
one new knob — 1.0 would ignore the photo (text-to-image), 0.0 would return it
untouched, and `providers.flux.DEFAULT_I2I_DENOISE` starts at 0.7
`[speculative]`. The photo is centre-cropped and scaled to the native canvas
first so output size and VRAM match text-to-image.

What this is **not**: instruction editing ("swap the cat for a dog, keep the
background"). Classic img2img keeps composition and colour and repaints
content in proportion to `denoise`; it does not understand the picture. That
would be **Flux Kontext** (another ~12 GB of weights, same non-commercial
licence, fp8 on a 24 GB card) or **Flux Redux** (SigLIP vision encoder plus a
Redux module, for style/content variations). Neither is wired; if img2img
turns out too blunt on real requests, Kontext is the next step, not more
denoise tuning.

## Three licence and policy facts, in one place

Independent of each other, and all three easy to lose track of:

1. **Flux.1-dev itself is non-commercial** (the box at the top of this file).
   Geographically unrestricted, unlike MiniMax H3 — the two models' licences
   constrain completely different things, and neither implies the other.
2. **The LoRA is `creativeml-openrail-m`**, which is a different licence from
   the base model's with its own use-based restrictions. Both apply at once.
3. **Distributing adult content through LINE carries a real risk of account
   suspension.** That is not a technical question and this document does not
   decide it — but the actual use is a **private, single group, personal
   use**, that is the basis on which the risk was accepted, and it is written
   down here so that in three months nobody has to wonder whether anyone
   thought about it.

The only thing in the code that bounds any of this is
**`LINE_ALLOWED_GROUP_ID`** — a hard allowlist, with an empty value meaning
*serve nobody* rather than *serve everyone* (`bots/line/webhook.py`). That
allowlist is the technical boundary of point 3, and it is sufficient because
of what the actual use is. **No content filtering.** That is a different
project, and half of one is worse than none.

## Prompting

**Not** a structured schema like H3's (`prompts/h3.py`). H3's schema exists
because it is *measured* to matter — 26.0 → 367.6 on the same scene, changing
only the prompt (`docs/model-h3.md`). No equivalent published or measured
schema advantage exists for Flux, which takes plain natural-language T5/CLIP
prose, and inventing one would be cargo cult.

There is now an LLM step all the same, and it does exactly one thing that
matters: **Chinese → English.**

- The trigger message is always Chinese, and Flux's T5/CLIP encoders are very
  weak at it. That makes translation the difference between the picture
  somebody asked for and a picture of something else — not a refinement.
- The LLM returns **JSON** (`{"prompt": "..."}`), validated into `FluxPrompt`.
  It never produces the final string; `render()` does, the same rule the video
  side follows.
- Failure falls back to submitting the original text with
  `built_by="template (llm failed: …)"` recorded on the job. A worse picture
  beats no picture, and the record is what explains it afterwards.
- This does add the image path to the LLM serverless cost line in
  [line-bot.md](line-bot.md#cost), which it previously did not touch. One
  ~400-token completion per image.

`QUALITY_SUFFIX` is appended by `render()` and is deliberately generic
("highly detailed, sharp focus, natural lighting, photorealistic"). Anything
more opinionated is a style choice belonging to the person asking, not to the
plumbing.

### Why the subject goes first

`DEFAULT_MAX_CHARS = 2000` counts **characters, not tokens**, and Flux's two
encoders have very different appetites: **CLIP-L reads only the first 77
tokens** and silently ignores the rest, while T5-XXL reads 512 in the usual
ComfyUI configuration.

So the ceiling protects the *request* (a pathological 50KB LINE message must
not be submitted), not the *quality* — nothing can stop CLIP-L ignoring the
tail. What follows is the ordering: subject first, where both encoders see it;
quality tail last, where only T5 does. Truncation is applied to the subject
*before* the tail is appended, so the tail can never be the part that is cut.

Splitting these into two bindings — one string per encoder — is the real fix,
and is deliberately not this change.

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
