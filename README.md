# ai-studio

Open-weight video, image and understanding models on rented GPUs, measured
honestly — and a LINE group that uses them.

## Monorepo

| package | what | start here |
|---|---|---|
| **`ai-studio`** (root) | GPU deployment and measurement: a RunPod pod running **MiniMax H3**, **Flux.1-dev**, **moondream3**, **Qwen2-Audio**, **Qwen2.5-VL** and **gpt-oss-20b**; the money guards; the numbers. Knows nothing about who asked. | this README · [`docs/`](#docs) · [`CLAUDE.md`](CLAUDE.md) |
| [**`fun_workflow/`**](fun_workflow/) | The LINE bot built on it: webhook, queue, on-demand worker, `/短劇` dramas, `/himonkey` chat, the status page each result links to. Installs ai-studio editable from `..`. | [`fun_workflow/README.md`](fun_workflow/README.md) |
| [**`twin/`**](twin/) | A personal digital-twin agent framework. Own stack, spec-driven, unrelated to the two above. | [`twin/README.md`](twin/README.md) |

Three `pyproject.toml`s, three `uv.lock`s, three test suites, three sets of
layering contracts. Status: generation, understanding, the pod lifecycle,
the bot and the measurement pipeline run in production on an RTX 4090; the
editing grammar is [specified, not implemented](docs/editing-grammar.md).

## Try it without a GPU

```bash
uv sync --group dev
uv run ai-studio doctor
uv run ai-studio generate "a baker opening the shutters before sunrise" --provider stub
uv run ai-studio understand photo.jpg --kind image

cd fun_workflow && uv sync --group dev
uv run funapp drama-dryrun            # the whole /短劇 machine: stub Flux/H3, real ffmpeg
```

Needs `ffmpeg` ≥ 8.0 on `PATH`; `doctor` says what is missing. The stubs
honour the real provider protocol (`submit` / `poll` / `fetch` / `cancel`).

## Measured on the GPU

📏 measured by us · `[reported]` quoted · `[speculative]` inferred — only 📏
appears here. Generated from the timestamped snapshots in
[`assets/metrics/`](assets/metrics/) by `ai-studio metrics export` +
`ai-studio metrics readme`; the status page shows the same GPU tier and
rate to whoever asked.

<!-- metrics:start -->
_Snapshot 2026-08-28T16:49:09+00:00 — RunPod Secure Cloud, our own runs only. Sources and notes per row are in the JSON._

**Measured on our own hardware**

| model | GPU | metric | value | on |
|---|---|---|---|---|
| MiniMax H3 | RTX 4090 24 GB | clip 5.17 s (124 frames): submit to mp4 | **79 s** | 2026-08-26 |
| MiniMax H3 | RTX 4090 24 GB | clip 10.1 s (243 frames): submit to mp4 | **79 s** | 2026-08-26 |
| MiniMax H3 | RTX 4090 24 GB | clip 10.1 s (243 frames): VRAM peak | **22.1 GB** | 2026-08-26 |
| MiniMax H3 | RTX 4090 24 GB | clip 8.0 s (192 frames): submit to mp4 | **170 s** | 2026-08-26 |
| MiniMax H3 | RTX 4090 24 GB | clip 8.0 s (192 frames): VRAM peak | **23.5 GB** | 2026-08-26 |
| MiniMax H3 | RTX 4090 24 GB | clip 12.25 s (294 frames): submit to mp4 | **215 s** | 2026-08-26 |
| MiniMax H3 | RTX 4090 24 GB | clip 12.25 s (294 frames): VRAM peak | **23 GB** | 2026-08-26 |
| MiniMax H3 | RTX 4090 24 GB | text-encoder reload after a swap | **60 s** | 2026-08 |
| MiniMax H3 | A40 48 GB | clip 5.17 s (124 frames): VRAM peak | **43.3 GB** | 2026-08 |
| MiniMax H3 | A40 48 GB | clip 5.17 s (124 frames): sampling, 6-step turbo | **125 s** | 2026-08 |
| MiniMax H3 | A40 48 GB | clip 5.17 s (124 frames): submit to mp4 | **300 s** | 2026-08 |
| MiniMax H3 | any | weight set on disk, int8 path | **54.7 GB** | 2026-08 |
| Flux.1-dev | RTX 4090 24 GB | checkpoint reload after a swap | **15 s** | 2026-08 |
| Flux.1-dev NSFW LoRA | any | adapter on disk | **0.69 GB** | 2026-08 |
| Qwen2-Audio-7B-Instruct | RTX 4090 24 GB | VRAM resident, fp16 | **17.2 GB** | 2026-08-27 |
| Qwen2-Audio-7B-Instruct | any | repo on disk | **16.8 GB** | 2026-08-27 |
| Qwen2-Audio-7B-Instruct | RTX 4090 24 GB | cold load | **27 s** | 2026-08-27 |
| Qwen2.5-VL-7B-Instruct | RTX 4090 24 GB | VRAM resident, fp16 | **17 GB** | 2026-08-27 |
| Qwen2.5-VL-7B-Instruct | any | repo on disk | **16.6 GB** | 2026-08-27 |
| Qwen2.5-VL-7B-Instruct | RTX 4090 24 GB | cold load | **53 s** | 2026-08-27 |
| Qwen2.5-VL + Qwen2-Audio | RTX 4090 24 GB | video description, both models, cold | **105 s** | 2026-08-27 |
| moondream3-preview | any | repo on disk | **47.6 GB** | 2026-08 |
| gpt-oss-20b | any | repo on disk | **41.3 GB** | 2026-08-27 |
| RunPod pod | any | disk overhead, 80 GB volume + 20 GB container | **0.014 USD/hr** | 2026-08 |

**Pod sessions this month**

| tier | datacenter | $/hr | sessions | GPU-hours | USD | minutes min / median / max |
|---|---|---|---|---|---|---|
| RTX 4090/SECURE | EUR-IS-1 | 0.74 | 18 | 9.85 | $7.29 | 8.2 / 21 / 118.7 |

**Per-render benchmark** (folded daily from real renders)

_Nothing folded yet: the per-render record line landed on 2026-08-28 and no pod has rendered since. The daily `ai-studio archive` fills `runs/benchmark/<month>.json`; the next `metrics export` picks it up._
<!-- metrics:end -->

## What runs where

| provider | model | trigger | doc |
|---|---|---|---|
| `comfyui` | MiniMax H3 (video) | `/影片` `/圖影` `/短劇` | [model-h3](docs/model-h3.md) |
| `flux` | Flux.1-dev (image) | `/圖片` `/圖圖` `/短劇` | [model-flux](docs/model-flux.md) |
| `understand-image` | moondream3 | `/說圖` | [model-moondream3](docs/model-moondream3.md) |
| `understand-audio` | Qwen2-Audio-7B | `/說音`, the sound of `/說影` | [model-qwen2-audio](docs/model-qwen2-audio.md) |
| `understand-video` | Qwen2.5-VL-7B + Qwen2-Audio | `/說影` | [model-qwen2.5-vl](docs/model-qwen2.5-vl.md) |
| `chat` | gpt-oss-20b — also rewrites every prompt | `/himonkey`, `/短劇`'s screenwriter | [model-gpt-oss-20b](docs/model-gpt-oss-20b.md) |

One pod, one card at a time: ComfyUI holds the H3/Flux checkpoint, the
inference server holds one of the other four, and `pipeline.residency`
evicts whichever side the next job does not need. Every provider polls
rather than blocks (RunPod's proxy cuts requests at ~100 s). The pod-side
server holds no wording: every question arrives with the request.

Licences: **H3 excludes the US, EU, UK and South Korea** (placement is a
licence decision), **Flux.1-dev is non-commercial**, the Qwen models are
Apache-2.0, moondream3's is unverified.

## Architecture

```
ai-studio      core → config · benchmark · checks · prompts · editing → media · storage
               → llm · inference · comfy · providers → pipeline → runtime → cli
fun_workflow   core → config → storage → prompts → pipeline → bots → api → cli   (only cli imports ai_studio.runtime)
```

Enforced by `import-linter` ([`pyproject.toml`](pyproject.toml)); what
ai-studio deliberately does *not* know, and where each of those things
lives instead, is in [`CLAUDE.md`](CLAUDE.md#architecture). Full picture:
[docs/architecture.md](docs/architecture.md).

## Money

Every expensive mistake here is a quiet one. `pod down` terminates (a
stopped pod still bills its disk); never configure an auto-deploy
reservation; `AI_STUDIO_MAX_COST_USD` per run, `AI_STUDIO_MAX_MONTH_USD`
per month and a daily open cap are all checked *before* a pod exists.
Runbook: [docs/runpod.md](docs/runpod.md) · lifecycle:
[docs/schedule.md](docs/schedule.md).

## Docs

| | |
|---|---|
| [architecture](docs/architecture.md) · [schedule](docs/schedule.md) · [runpod](docs/runpod.md) · [observability](docs/observability.md) | layers; pod lifecycle and money; deployment runbook; tracing, archive, benchmark fold |
| [editing-grammar](docs/editing-grammar.md) · [attribution](docs/attribution.md) | the rules to edit by (not yet implemented); what is inherited from `Hao0321/video-autopilot-kit` |
| `docs/model-*.md` | one per model: weights, licence, what we measured and what we have not; retired models kept as decision records |
| [fun_workflow/docs/line-bot.md](fun_workflow/docs/line-bot.md) · [drama.md](fun_workflow/docs/drama.md) | every trigger, the two-second webhook budget, caps, costs; the `/短劇` pipeline |
| [twin/README.md](twin/README.md) · [twin/reference/SPEC.md](twin/reference/SPEC.md) | the digital twin |

## Licence

MIT — [LICENSE](LICENSE). Derived-work attribution in [NOTICE](NOTICE).
