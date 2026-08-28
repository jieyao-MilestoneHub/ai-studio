# assets/metrics

Everything this project has measured on its own GPUs, as timestamped
snapshots. No generated media lives in this repository; the numbers are the
record. Only figures graded 📏 (measured by us) are exported — never
`[reported]` or `[speculative]` ones — and nothing in a snapshot identifies a
request, a user, a group or a pod.

| file | what | source |
|---|---|---|
| `measured-<stamp>.json` | every figure measured on our own hardware, with model, GPU, date and the doc that records the run | `ai_studio.benchmark.measured.MEASURED` |
| `sessions-<stamp>.json` | every pod session this calendar month, per GPU tier and per session: minutes, USD, $/hr, datacenter, VRAM, quantisation, why it closed | `runs/.spend_ledger.json` + `logs/sessions/*.json` |
| `benchmark-<stamp>.json` | per-`(kind, gpu_tier)` means over real renders (seconds, cost, VRAM, frames/s), one entry per month; **only written once at least one day has been folded** | `runs/benchmark/<month>.json`, folded daily by `ai-studio archive` |

`<stamp>` is UTC, `YYYYMMDDTHHMMSSZ`, so files sort by time and nothing is
overwritten. Each carries `generated_at`, `kind`, `note`, `data`.

```bash
uv run ai-studio metrics export          # writes new snapshots here
uv run ai-studio metrics readme          # re-renders the tables below from the latest of each kind
```

## Latest

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
