# ai-studio

**What can one rented GPU actually do, and what does it cost?**

Open-weight video, image, vision-language and chat models on a single
24 GB card, deployed with hard money guards and measured on our own runs.

## Why

- **One GPU, six models.** MiniMax H3 (video), Flux.1-dev (image),
  moondream3, Qwen2-Audio, Qwen2.5-VL (understanding) and gpt-oss-20b (chat
  and prompt rewriting) share one RTX 4090, one model resident at a time.
  The question is what that card actually delivers — seconds, VRAM, dollars —
  not what a model card says.
- **The money is guarded before it is spent, not reported after.** A
  calendar-month ceiling, a derived daily allowance and a per-day cap on pod
  creations are all checked before a pod exists — see
  [pod lifecycle & money](docs/schedule.md).
- **Numbers are graded.** 📏 measured by us, `[reported]` quoted,
  `[speculative]` inferred. Only the first kind is exported, with a
  timestamp, to [`assets/metrics/`](assets/metrics/README.md).

## What is inside

| package | what it does |
|---|---|
| **`ai-studio`** (root) | The GPU side. Opens a RunPod pod on demand, provisions ComfyUI and a small inference server over SSH, serves the six models behind one `submit / poll / fetch / cancel` protocol, guards spend before a pod exists, records every render. Knows nothing about who asked. |

One Python package, with its own lockfile, tests and layering contracts.

### What is deliberately not published

Two other things live on the working machine and are **not** in this
repository:

- **The request side** — a chat webhook, a queue, a worker and a delivery
  path that let a group trigger any model this repo serves. An application
  built on top of the GPU work is a different project, with concerns this one
  does not want: service credentials, member identifiers, an operator on call.
- **A digital-twin side quest** — a separate framework with its own stack, its
  own spec, and a real person's history as training data.

This repository publishes one thing: what a rented GPU costs and delivers.
Commits before September 2026 still contain both, and that history is not
rewritten — removing them from the published tree stops them going further,
and pretending they were never there would be the dishonest version. The seam
is real either way: ai-studio exposes plain functions and never imports back,
so the GPU side stands alone and is tested that way.

## Quick start

No GPU, no account, no cost:

```bash
uv sync --group dev
uv run ai-studio doctor                                   # python, ffmpeg + filters, credentials, disk
uv run ai-studio generate "a baker opening the shutters" --provider stub
uv run ai-studio understand photo.jpg --kind image
```

With a RunPod key in `.env` (`.env.example` lists every name):

```bash
uv run ai-studio pod capacity        # what the licence-safe ladder can get right now — no spend
uv run ai-studio session open        # one pod, self-terminating at its lease end
uv run ai-studio session close       # terminates; a stopped pod would still bill its disk
uv run ai-studio bench               # this month's measurements per GPU tier
```

## How it stays honest and cheap

- Every expensive mistake on a GPU cloud is a quiet one, so per-run,
  per-month and per-day ceilings are checked *before* a pod is created, and
  a quiet pod is reaped minutes after its last render.
- Placement is a licence decision: MiniMax H3 excludes several
  jurisdictions, so the capacity ladder names only datacenters where it may
  run. Flux.1-dev is non-commercial.
- The pod-side server holds no wording: every question travels with the
  request, so the same GPU code serves any client.
- Layering is enforced by `import-linter`, not described; what ai-studio
  deliberately does not know is listed in [`CLAUDE.md`](CLAUDE.md).

## Docs

[architecture](docs/architecture.md) · [pod lifecycle & money](docs/schedule.md) ·
[RunPod runbook](docs/runpod.md) · [observability](docs/observability.md) ·
[measurements](assets/metrics/README.md) · one doc per model under
[`docs/`](docs/)

## Licence

MIT — [LICENSE](LICENSE); model licences differ, see each model's doc.
Derived-work attribution in [NOTICE](NOTICE).
