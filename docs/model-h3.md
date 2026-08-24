# MiniMax H3

The generation model. Run through ComfyUI on a RunPod pod.

> **Licence:** MiniMax H3's licence **excludes the United States, the EU, the
> UK, and South Korea.** That makes datacenter placement a licensing decision,
> not just a latency one. Iceland (`EUR-IS-*`) and Norway (`EUR-NO-*`) are EEA
> but not EU; Romania (`EU-RO-1`), Czechia, France, the Netherlands and Sweden
> are EU. The allowed list is `runtime.pod.LICENCE_SAFE_DATACENTERS`. Testing on
> your own hardware is one thing; this is the kind of clause that surfaces once
> a client deliverable exists.

All performance figures below are `[reported]` — quoted from third-party
measurements, not measured by us. See [attribution.md](attribution.md) on number
honesty.

## Weights

Roughly 63 GB in total:

| component | size | notes |
|---|---|---|
| `fl2va` DiT | 21 GB | text-to-video and keyframe modes |
| `ref2va` DiT | 21 GB | reference mode, for character consistency |
| Qwen3-VL-32B text encoder | 15.7 GB | the reason host RAM matters |
| 2 × VAE | — | |
| 4-step turbo LoRA | — | see the trap below |

**Download only `fl2va` first (~42 GB).** Add `ref2va` when you actually need
character consistency. On a datacenter link 42 GB takes about 9 minutes at
~75 MB/s, which on a 4090 costs roughly $0.11.

## Host requirements

| requirement | value | why |
|---|---|---|
| GPU | RTX 4090 24 GB | VRAM peaks at 15.7–21.9 GB, so 24 GB is ample |
| **Host RAM** | **≥ 60 GB** | ComfyUI does not release the 32B text encoder after loading. A 31 GB host crashed part-way through a second consecutive generation. Not selectable via API — deploy, inspect, terminate if short. |
| CUDA | **13.0** | H3's quantised fast paths assume the CUDA 13 generation; on 12.8 strong cards fall back to a slow path |
| Template | Official RunPod ComfyUI, CUDA 13 | the community templates crowd it out in search — look for the "Official Runpod ComfyUI templates" card |
| Disk | 200 GB container disk (template default) | no network volume — see [runpod.md](runpod.md) |

Launch ComfyUI as:

```bash
python main.py --fast-disk --use-sage-attention --reserve-vram 0.7
```

`--fast-disk` is not optional on a 64 GB host. It keeps weights in the page
cache rather than anonymous memory; without it RSS climbs and the second model
load starts hitting swap. This is the same problem as the 31 GB crash seen from
the other side.

## Timings, RTX 4090, 5s at 24fps

| canvas | RTX 4090 | RTX 3050 (local, for contrast) |
|---|---|---|
| 608×352 | 182s | 416s |
| **864×480** | **133s** | 886s |
| 1280×736 | 361s | 3082s |
| 1344×768 | ~300s warm, ~330s cold | — |

15s at 1280×736 took 2111s (~35 min).

> ⚠️ **One figure looks wrong.** On the 4090, 608×352 (182s) is *slower* than
> the larger 864×480 (133s), while the RTX 3050 column scales normally with
> resolution. Most likely a first-run warm-up artifact in the source
> measurement. Re-measure before relying on it — `providers/comfyui.py` carries
> the same caveat next to the table it uses for cost estimation.

## Settings

- **Sweet spot: 1344×768 + 12-step turbo** ≈ 80% of base-20 quality at 61% of
  the time, about 4–5 minutes per 5s clip.
- **We currently default to 864×480**, which is ~2.3× faster and scored
  marginally *higher* on the prompt-quality benchmark. The trade is upscaling:
  864×480 → 1920×1080 is 2.25×, while 1344×768 → 1920×1080 is only 1.43×. Both
  are preset in `editing/format_policy.py`; compare them on real footage before
  settling.

## ⚠️ The turbo trap

The turbo LoRA **cannot** go through ComfyUI's stock LoRA loader. The pruned
model replaces its AdaLN branch with a lookup table, so it needs
`MiniMaxH3TurboLoRA`, and sampling needs `MiniMaxH3TurboSampler` rather than
`KSamplerSelect`.

Wired the stock way you get vertical comb artifacts and banded gradients —
**while running about four times faster** (2.53 vs 9.8 s/iteration). Benchmark
that and you conclude you found a 6.3× free speedup; you found the model
skipping work. Turbo's real gain over 20-step base is about 1.7×.

Enforced in `comfy/validate.py` and tested in
`tests/unit/test_graph_validate.py`. If anyone reports suspiciously fast H3
numbers, ask which sampler they used.

## Prompting is the quality lever, not resolution

Same seed, same 1344×768, same scene, changing only the prompt:

| prompt style | score |
|---|---|
| free prose | 26.0 |
| more specific prose | 205.9 |
| **official structured schema** | **367.6** |

Same prose at five times the pixels: 608×352 **29.2**, 864×480 **30.3**,
1344×768 **26.0** — i.e. no effect.

**A blurry result is a prompt problem.** `videogen.prompts.h3` implements the
official schema from
[VIDEO_PROMPT_WRITING_GUIDE_base_en.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
as typed fields:

- a mode-specific instruction line (T2VA has none), then one blank line
- `integrated_multimodal_description:` — `[Shot 1]` with style and composition,
  later shots with strictly increasing cut times inside the duration
- `overall_soundscape:` — 1–4 sentences of ambience and action sound
- `non_diegetic_music:` — 1–3 sentences on instrumentation and dynamics, or `N/A`

Camera motion comes from a closed vocabulary (`pushes in`, `trucks right`,
`arcs around the subject`, …) written as prose, with amplitude and speed only
when they are not the default. Dialogue keeps identity outside `<d>` and the
verbatim words inside.
