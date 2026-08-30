# ai-studio

**What can one rented GPU do, and what can free compute do after that?**

Open-weight video, image, vision-language and chat models on a single
24 GB card, deployed with hard money guards and measured on our own runs —
plus a side quest: can a person, in August 2026, fine-tune a usable digital
twin of themselves on free-tier compute alone?

## Why

- **One GPU, six models.** MiniMax H3 (video), Flux.1-dev (image),
  moondream3, Qwen2-Audio, Qwen2.5-VL (understanding) and gpt-oss-20b (chat
  and prompt rewriting) share one RTX 4090, one model resident at a time.
  The question is what that card actually delivers — seconds, VRAM, dollars —
  not what a model card says.
- **Free compute next.** `twin/` takes the same discipline to the other end
  of the budget: Qwen3-8B + LoRA on the free tiers of Modal, Kaggle and
  Lightning, judged by a spec with acceptance criteria written before the
  first training run.
- **Numbers are graded.** 📏 measured by us, `[reported]` quoted,
  `[speculative]` inferred. Only the first kind is exported, with a
  timestamp, to [`assets/metrics/`](assets/metrics/README.md).

## What is inside

| package | what it does |
|---|---|
| **`ai-studio`** (root) | The GPU side. Opens a RunPod pod on demand, provisions ComfyUI and a small inference server over SSH, serves the six models behind one `submit / poll / fetch / cancel` protocol, guards spend before a pod exists, records every render. Knows nothing about who asked. |
| [**`fun_workflow/`**](fun_workflow/README.md) | The request side, built on it: a LINE group's webhook, queue and worker; `/himonkey` chat; a status page per result showing which GPU rendered it and what it rents for. |
| [**`twin/`**](twin/README.md) | The side quest: a personal digital-twin agent framework — one person's history and interviews in, an agent with their judgment and restraint out. Spec-first, with hard guardrails around the personal data it ingests. |

Three independent Python packages in one repository, each with its own
lockfile, tests and layering contracts. `fun_workflow` depends on
`ai-studio`; nothing depends on `twin`.

## Quick start

No GPU, no account, no cost:

```bash
uv sync --group dev
uv run ai-studio doctor                                   # python, ffmpeg + filters, credentials, disk
uv run ai-studio generate "a baker opening the shutters" --provider stub
uv run ai-studio understand photo.jpg --kind image

cd fun_workflow && uv sync --group dev
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
[`docs/`](docs/) · [the bot](fun_workflow/docs/line-bot.md) ·
[the twin spec](twin/reference/SPEC.md)

## Licence

MIT — [LICENSE](LICENSE); model licences differ, see each model's doc.
Derived-work attribution in [NOTICE](NOTICE).
