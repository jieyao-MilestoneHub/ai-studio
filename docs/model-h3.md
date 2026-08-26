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
| CUDA | **13.0** `[reported]` (⚠️ unverified against the current template, see below) | `[reported]`, and inherited rather than tested here: H3's quantised fast paths are said to assume the CUDA 13 generation, with 12.8 falling back to a slow path on strong cards. Nothing in this project has checked either half. `runtime/session.py` passes `--min-cuda-version 13.0` on every create, so if the standard template is really 12.8 that flag is either a no-op or a refusal on every rung — see PLAN.md Phase 7.0 |
| Template | RunPod's official **"ComfyUI"** template (`cw3nka7d08`) for standard GPUs — `runtime.session.TEMPLATE_COMFYUI_STANDARD` | ⚠️ 📏 checked 2026-08-25: the template id this repo previously hardcoded (`2lv7ev3wfp`) has been renamed/rescoped to **"ComfyUI Blackwell Edition"** and is now specifically for RTX 5090/B200, not the RTX 4090/L40S this project targets. RunPod's current docs only mention `runpod/comfyui:latest` (standard) and `runpod/comfyui:cuda12.8` (Blackwell) image tags — no "cuda13" tag is documented anywhere today, which is in tension with the CUDA 13.0 requirement above. **Verify the standard template's actual CUDA version live before trusting the turbo path on it** — this has not been done yet. |
| Disk | 200 GB container disk (template default) | no network volume — see [runpod.md](runpod.md) |

Launch ComfyUI as:

```bash
python main.py --fast-disk --use-sage-attention --reserve-vram 0.7
```

`--fast-disk` is not optional on a 64 GB host `[reported]`. It keeps weights in
the page cache rather than anonymous memory; without it RSS climbs and the
second model load starts hitting swap. This is the same problem as the 31 GB
crash seen from the other side.

⚠️ **The flag names themselves have not been verified against a real ComfyUI.**
Both `--fast-disk` and `--use-sage-attention` are `[reported]`, and an
unrecognised flag is not a warning — it is argparse exiting and no ComfyUI at
all, discovered after the weights have been paid for. `deploy/pod_setup.sh`
therefore probes `main.py --help` and drops any flag this build does not
advertise, logging that it did; `tests/unit/test_deploy_scripts.py` runs that
probe block against synthetic help output so the fallback is tested rather than
assumed. Promote these to 📏 only after reading them out of a live
`main.py --help`.

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

## What we have measured ourselves, and what we have not

📏 **Measured on our own account**

| | |
|---|---|
| Peak VRAM, 864×480 / 124 frames, int8 + bypass, A40 | **43.3 GB** |
| Sampling, 6-step turbo, same settings, A40 | **125 s** (87 → 40 → 25 → 18 s/iter; the first step carries the warm-up) |
| Submit → mp4 on disk, same run | **300 s** |
| Output | 864×480, 24fps, 124 frames, 5.17 s, h264 High + AAC stereo 32kHz, 0.99 MB |
| Pod disk overhead, 80GB volume + 20GB container | **$0.014/hr** (a $0.44/hr pod bills 0.454; a $0.74/hr pod bills 0.754) |
| Weight set actually needed, int8 path | **54.7 GB** across five files |

⛔ **Still unmeasured — do not treat the numbers elsewhere in this file as ours**

- **Generation time on a 48GB Ada card with fp8.** The `1.38×` fp8 advantage is
  `[reported]` from someone else's 4090. Our only timing is A40 + int8. Note
  that until 2026-08-25 this was not merely unmeasured but *unreachable*: the
  48GB rungs pointed at a datacenter with no L40S in it.
- **Whether 43.3GB fits in 24GB with `low_vram=True`.** Rungs 3 and 4 of the
  ladder assume it does, on the node pack's word. An attempt on a 4090 was cut
  short before generation (see below), so this is still an assumption.
- **Whether the second clip in a window is faster.** This decides whether a
  window amortises its setup at all. ComfyUI logs `Using RAM pressure cache`,
  which suggests it may evict the model between jobs.

### Why the 4090 attempt did not finish

Worth recording, because two of the three causes were bugs in our own tooling
and are now fixed in `deploy/pod_setup.sh`:

1. **`hf_transfer` is not preinstalled on the image**, and on its version of
   `huggingface_hub` the `HF_HUB_ENABLE_HF_TRANSFER` switch is a no-op that
   warns *"not used anymore, please use HF_XET_HIGH_PERFORMANCE instead"*.
   Downloads ran at **~5 MB/s instead of ~75 MB/s** — 51GB goes from 11 minutes
   to nearly three hours of billed pod time. Installing it also needs
   `--break-system-packages`, since the image's python is externally managed.
2. **Starting the download before the ComfyUI upgrade saturates the link**, and
   `git fetch` then hung behind it for over eight minutes. The upgrade is small;
   it now runs first.
3. Placement itself: the ladder fell through L40S Secure and L40S Community
   and landed on **RTX 4090 Secure**, a 24GB rung — so the run would have
   measured the `low_vram` path rather than the fp8 one anyway.

   **That third cause was misdiagnosed here, and the correction matters more
   than the original note.** Both L40S rungs were recorded as "refused", i.e.
   out of stock. They were not: 📏 checked against `/catalog/gpus` on
   2026-08-25, **L40S is not offered in Iceland at all**. Its secure stock sits
   in EU-NL-1, OC-AU-1, US-NC-1, US-TX-3 and US-TX-4, and every ladder rung
   pinned `EUR-IS-2`. Those two rungs could never have been filled, on any day,
   at any price.

   A deploy refusal looks identical whether the datacenter is empty or has
   never had that card, which is why this survived a live run. The ladder now
   places L40S in **OC-AU-1** — the only one of those five that H3's licence
   permits, the Netherlands being EU and the rest US — and `ai-studio pod
   placement` checks every rung against the catalog so a dead rung is loud
   instead of looking like bad luck.

The run was terminated rather than pushed through, at a cost of $0.44 for the
whole session. The measurement is worth redoing; it is not worth paying for a
run that was not going to finish inside its window.

## Clip length: 10.1 s by default, measured

📏 RTX 4090, 864×480, int8 + merged LoRA, 6-step turbo, 2026-08-26, one run
each (`ai-studio generate --seconds N` against a warm pod):

| frames (17k+5) | seconds | ComfyUI "Prompt executed" | VRAM peak (nvidia-smi, 500 ms) |
|---|---|---|---|
| 124 | 5.17 | 79–100 s (seven production jobs) | — |
| 192 | 8.0 | 170 s, of which ~60 s was a model reload (ComfyUI's RAM-pressure cache had evicted it) | 23.5 GB, during the reload |
| 243 | 10.1 | **79 s** | **22.1 GB** |
| 294 | 12.25 | 215 s (a reload is likely inside that, like the 192 run; not separated) | 23.0 GB |

Length barely moves the clock: the turbo path's time is dominated by the
32B text encoder and the VAE, not by frame count, and the VRAM peak comes
from loading, not from activations. At 10.1 s the subject, clothing and
street were the same in the first and last frame. 294 frames (12.25 s) also
rendered, slower and closer to the 24 GB ceiling; the default stops at 243
because the community guide puts drift risk above ~10 s, one run is not a
measurement of drift, and 23.0 GB leaves no room for a bigger text encoder.
`pipeline.convert_worker.DEFAULT_FRAMES` is the single place this number
lives; `snap_frames` keeps any other request on the grid.

## Prompting

**The request is the spec.** The conversion (`prompts/convert.py`) translates
the user's words into the schema below; it does not improve them. Every
concrete word the user typed survives as its plain English equivalent, proper
nouns and on-screen text stay verbatim, and a thin request produces a short
description rather than one padded with invented detail. The reason is
accountability: when a clip is wrong, the user has to be able to see their own
words in the prompt to know whether the model or the translation lost them.

The shot discipline follows the H3 community guide `[reported]`
(<https://www.aiposthub.com/minimax-h3-tutorial-prompt-multimodal-video/>):
first shot opens with the one-sentence core idea (who, where, doing what,
style); every shot names its timing, restates the subject, its framing and
one action; at most two cuts under ten seconds; a person who is not speaking
is said to be not speaking with lips closed; the user's 「不要…」 list becomes
explicit "no ..." phrases; music is tied to events, not only to seconds. For
image-to-video the picture already fixes appearance and framing, so the
prompt states that role up front and describes only what happens next.

 is the quality lever, not resolution

Same seed, same 1344×768, same scene, changing only the prompt:

| prompt style | score |
|---|---|
| free prose | 26.0 |
| more specific prose | 205.9 |
| **official structured schema** | **367.6** |

Same prose at five times the pixels: 608×352 **29.2**, 864×480 **30.3**,
1344×768 **26.0** — i.e. no effect.

**A blurry result is a prompt problem.** `ai_studio.prompts.h3` implements the
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
