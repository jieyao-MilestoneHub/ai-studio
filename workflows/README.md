# ComfyUI workflows

A workflow here is a ComfyUI **API-format** graph plus a `_ai_studio` block that
says where run parameters land.

## Producing one

1. Build the graph in the ComfyUI UI on the pod.
2. **Save (API Format)** — not the ordinary Save. The ordinary format carries UI
   layout and cannot be submitted to `/prompt`.
3. Drop the JSON here and add a `_ai_studio` block (below).
4. `ai-studio generate --provider comfyui --workflow workflows/yours.json`

The `_ai_studio` block is stripped before submission, so ComfyUI never sees it.

```json
{
  "_ai_studio": {
    "expect_turbo": true,
    "bindings": {
      "prompt":   ["6",  "text"],
      "width":    ["27", "width"],
      "height":   ["27", "height"],
      "length":   ["27", "length"],
      "seed":     ["31", "noise_seed"],
      "steps":    ["31", "steps"],
      "fps":      ["40", "fps"],
      "filename": ["40", "filename_prefix"]
    }
  }
}
```

`prompt`, `width`, `height`, and `length` are required; the rest are optional
and are only injected when present. `length` is in **frames**, not seconds —
the provider multiplies duration by fps.

**`flux_dev.json` is the exception**: a still image has no frame count, so it
loads with `required_bindings=IMAGE_REQUIRED_BINDINGS` (`prompt`/`width`/
`height` only, no `length`) via `providers.flux.FluxComfyUIProvider`.
`flux_dev_i2i_face.json` is a third sibling for `/短劇` keyframes: the i2i
graph plus an Impact-Pack `FaceDetailer` (bbox from Impact-Subpack's
`UltralyticsDetectorProvider`) between `VAEDecode` and `SaveImage`.
`providers.flux` submits it only when `/object_info` registers both nodes;
every value on the detailer node is `[speculative]`. The turbo trap below does
not apply to it either — it has no H3 nodes. Stills only, never video
(`docs/drama.md`).

`flux_dev_i2i.json` is its image-to-image sibling, found by name the same way
`h3_i2va_*.json` is found next to `h3_fl2va_*.json` (`Workflow.sibling`): the
empty latent is replaced by `LoadImage → ImageScale → VAEEncode`, and
`BasicScheduler.denoise` is bound so the provider decides how much of the
photo survives (`DEFAULT_I2I_DENOISE`, `[speculative]` 0.7). Everything else —
weights, LoRA wiring, steps — is identical, and a test asserts the LoRA still
sits in front of every model consumer in the copy. It is a
standard non-turbo Flux.1-dev graph (`UNETLoader` → `DualCLIPLoader` →
`CLIPTextEncode` → `FluxGuidance` → `BasicGuider`/`BasicScheduler`/
`SamplerCustomAdvanced` → `VAEDecode` → `SaveImage`) — the turbo trap below
does not apply to it at all, since `uses_turbo_lora()` only fires on
`MiniMaxH3TurboLoRA` or a stock LoRA loader carrying a turbo/lightning hint,
neither of which this graph has.

Bindings are explicit rather than inferred by node class because a graph with
two text encoders makes inference ambiguous, and because a re-export that
renumbers nodes then fails loudly at load time instead of silently generating
from an empty prompt.

## ⚠️ The turbo trap

`validate_graph()` runs on every load and every submission, and this is what it
is guarding:

> The MiniMax H3 turbo LoRA **cannot** be driven by ComfyUI's stock LoRA loader.
> The pruned model replaces its AdaLN branch with a lookup table, so the LoRA
> must go through **`MiniMaxH3TurboLoRA`**, and sampling must go through
> **`MiniMaxH3TurboSampler`** rather than `KSamplerSelect`.
>
> Wire it the stock way and you get vertical comb artifacts and banded
> gradients — **while running about four times faster** (2.53 s/iteration
> against 9.8 s/iteration). Benchmark that and you will conclude you found a
> 6.3× free speedup. You did not: the model is skipping work and producing
> garbage. Turbo's real advantage over 20-step base is about **1.7×**.
> [reported]

So:

| do | do not |
|---|---|
| `MiniMaxH3TurboLoRA` | `LoraLoaderModelOnly`, `LoraLoader` |
| `MiniMaxH3TurboSampler` | `KSamplerSelect`, `KSampler`, `KSamplerAdvanced` |

Setting `"expect_turbo": true` also asserts the turbo path is *present*, so a
workflow that silently lost its LoRA fails instead of quietly rendering at full
base cost.

## Settings that matter

| setting | value | why |
|---|---|---|
| Native canvas | 864×480 or 1344×768 | 864×480 is ~2.3× faster; 1344×768 upscales to 1080p far more gently (1.43× vs 2.25×) |
| Steps | 12, turbo | ≈80% of base-20 quality at 61% of the time [reported] |
| fps | 24 | H3's native rate. Delivering at native rate avoids frame interpolation, which is blacklisted |
| ComfyUI launch | `--fast-disk --use-sage-attention --reserve-vram 0.7` `[reported]` | `--fast-disk` is not optional on a 64 GB host `[reported]`: ComfyUI does not release the 32B text encoder after loading, and without it the second model load starts hitting swap. **The flag names are unverified against a real ComfyUI** — `deploy/pod_setup.sh` probes `main.py --help` and drops what this build does not advertise, because an unknown flag means argparse exits and there is no ComfyUI at all |

## Resolution is not the quality lever

Same seed, same scene, changing only the prompt: free prose **26.0**, more
specific prose **205.9**, official structured schema **367.6**. Same prose at
five times the pixels: 608×352 **29.2**, 864×480 **30.3**, 1344×768 **26.0**.
[reported]

A blurry result is a prompt problem. Use `ai_studio.prompts.h3` — it builds the
official schema from typed fields, so the structure is the output rather than
something you have to remember.
