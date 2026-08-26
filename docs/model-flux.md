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

---

## Corrections from the 2026-08-26 research pass

### The LoRA format risk is closed — it loads

This file previously carried the diffusers-vs-ComfyUI key mapping as the one
`[speculative]` technical risk, on the grounds that nobody had verified this
adapter. It is now `[verified]` from ComfyUI's source, and the answer is that
**ComfyUI maps diffusers-format Flux LoRAs natively**.

`comfy/lora.py::model_lora_keys_unet` has an explicit Flux branch that registers
four naming conventions on top of the diffusers namespace, the first of which is
literally `transformer.{key}` — comfyanonymous's own comment reads
*"simpletrainer and probably regular diffusers flux lora format"*. The
`lora_A`/`lora_B` suffixes are consumed as `diffusers2_lora`, which is the PEFT
convention.

`comfy/utils.py::flux_to_diffusers` then patches split `to_q`/`to_k`/`to_v` into
**slices of** BFL's fused `linear1`/`qkv`, which is precisely the mechanism that
makes a diffusers-style LoRA work against the fused checkpoint.

The keys that genuinely fail in the wild carry an extra PEFT wrapper infix —
`transformer.base_model.model.single_transformer_blocks.…` — from SimpleTuner.
Ours has no such infix.

**What this does not remove is the need to check.** ComfyUI never raises on a
key mismatch: it logs `lora key not loaded: <key>` at WARNING and returns a
completely valid, completely un-LoRA'd image. That log line is the detector, and
PLAN.md Phase 7.1 greps for it before anything else.

`LoraLoaderModelOnly` stays. With `clip=None` the text-encoder half of the key
map is never built, so any text-encoder keys would be *loudly* dropped rather
than silently — and a byte-budget calculation puts this file at 100% transformer
weights anyway (all-linear-target, rank 32/fp32 or 64/bf16 `[speculative]`,
987 tensors ≈ the 687,476,088 bytes observed).

### Alpha scaling, and why 1.0 is the right strength

`[verified]` ComfyUI reads a `<key>.alpha` **tensor from inside the safetensors
file** to compute the LoRA scale. PEFT does not write one — it puts `lora_alpha`
in a separate `adapter_config.json`, and the Heartsync repo ships **only**
`lora.safetensors`. So ComfyUI applies `alpha = 1.0`. Diffusers, given a bare
safetensors with no config, infers the same. The two agree at
`strength_model=1.0`, which is what the graph uses.

`[not found]` — what `lora_alpha` the author actually trained with. If it was
not equal to the rank and they relied on a config they never published, both
runtimes are equally wrong and the model card's own example is the reference.

### Negative prompts: structurally unavailable, and that is fine

`[verified]` `BasicGuider` takes one conditioning input and has nowhere to put a
negative. `CFGGuider` could, but ComfyUI **skips the uncond pass entirely at
cfg 1.0** as an explicit optimisation, and above 1.0 costs a second forward pass
per step on a model that is guidance-distilled. ComfyUI's own Flux docs say to
keep CFG at 1.0.

So `FluxPrompt` deliberately has no `negative` field. Encode avoidance
positively in the LLM translation step instead.

### T5 and CLIP-L limits — the widely-repeated numbers are wrong for ComfyUI

This is worth restating carefully because the previous note here described
*diffusers* behaviour, not ComfyUI's.

`[verified]` from `comfy/text_encoders/flux.py`, the T5-XXL tokenizer is
constructed with **`max_length=99999999`, `min_length=256`,
`pad_to_max_length=False`**. So in ComfyUI:

- **T5 has no maximum and does not truncate.** The famous 512 is the *diffusers*
  `FluxPipeline` default (`max_sequence_length=512`), which truncates with a
  warning. ComfyUI simply does not enforce it. Past ~512 tokens you are outside
  the training distribution, but nothing errors.
- The **256 is a minimum** — short prompts are padded up to it. That is where
  the "256 limit" in circulation comes from, and it is a floor, not a ceiling.

`[verified]` For CLIP-L the mechanism is different again from "reads the first
77 tokens". `FluxClipModel.encode_token_weights` returns `(t5_out, l_pooled)` —
the per-token CLIP-L sequence `l_out` is computed and **thrown away**. Only a
single pooled 768-d vector survives, derived from the *first* 77-token chunk.

So: everything past roughly the first 75 content tokens contributes nothing to
the CLIP-L path, and all the semantic work on a long prompt is done by T5. The
ordering in `prompts/flux.py` (subject first, generic quality tail last) is
right, and the reason is this rather than truncation.

`CLIPTextEncodeFlux` accepts separate `clip_l` and `t5xxl` strings and would let
the LLM emit a short subject line for one and the full description for the
other. Not adopted here — it changes output, and generation parameters are
frozen until the first measured run.

### Where the base weights actually come from

`deploy/pod_setup.sh` downloaded **none** of these until now — the graph loaded
four files the pod never fetched. All four verified against the HF file-tree API:

| role | repo | path | bytes |
|---|---|---|---|
| UNET | `black-forest-labs/FLUX.1-dev` **(gated)** | `flux1-dev.safetensors` | 23,802,932,552 |
| T5-XXL | `comfyanonymous/flux_text_encoders` | `t5xxl_fp8_e4m3fn.safetensors` | 4,893,934,904 |
| CLIP-L | `comfyanonymous/flux_text_encoders` | `clip_l.safetensors` | 246,144,152 |
| VAE | `Comfy-Org/Lumina_Image_2.0_Repackaged` | `split_files/vae/ae.safetensors` | 335,304,388 |

Two things that bite a setup script:

- **FLUX.1-dev is gated.** `hf download` returns 401 without a token whose
  account has accepted the FLUX.1 [dev] Non-Commercial License once on the model
  page. The script now checks `HF_TOKEN` *before* starting the 52 GB H3 pull, so
  the failure costs seconds rather than twenty billed minutes.
- **The VAE comes via the Lumina repackage**, which is ungated and
  byte-identical to the gated copy, and is what ComfyUI's own Flux example page
  points at. `hf download` preserves the repo's layout, so it lands at
  `vae/split_files/vae/` and has to be flattened — `VAELoader` looks in `vae/`
  and nowhere else.

`[reported]` `t5xxl_fp8_e4m3fn_scaled.safetensors` (5,157,348,688 — 264 MB more)
carries per-tensor scales and is what ComfyUI's example page now lists. A cheap,
low-risk upgrade, deliberately not taken in this pass because it changes output.

### fp8 on this ladder

`[verified]` `comfy/model_management.py::supports_fp8_compute` requires
`major >= 9`, or `major == 8 and minor >= 9`. **RTX 4090 and L40S are both
sm_89**, so fp8 compute is available on every rung. `fp8_e4m3fn` and
`fp8_e4m3fn_fast` store identical weights — the difference is that `_fast` runs
the *matmul* in fp8 (and clamps activations to ±448, a silent saturation),
while plain `fp8_e4m3fn` upcasts to bf16 to compute.

`[verified]` There is **no** `fp8_scaled` variant of plain `flux1-dev` published
by Comfy-Org, unlike Kontext and Krea. The options are bf16 plus a runtime cast
(what the graph does), or a pre-quantised UNET from `Kijai/flux-fp8`.

On the L40S rung there is headroom for `weight_dtype=default` (bf16, 23.8 GB),
which would remove the whole fp8-plus-LoRA risk class. Not changed here —
generation parameter.

### Black images are not errors

`[reported]` Flux fp8 has produced pure black output with
`RuntimeWarning: invalid value encountered in cast` — NaNs reaching the uint8
conversion. `[verified]` ComfyUI's own `--fp16-vae` help says it "might cause
black images".

Nothing in ComfyUI treats this as a failure: `SaveImage` writes it, `/history`
reports success, and our provider would fetch and push it. That is what
`gates/output_gate.py` and `media.luma_stats()` now exist for — a flat-luma
render is rejected and the requester is told it failed, rather than receiving a
black square.
