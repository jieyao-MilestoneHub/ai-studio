# ai-studio

Run open-weight video, image, and understanding models on rented GPUs. Measure them honestly, and experiment through a practical client — from LINE groups to real-world workloads.

Digital twins are just a side quest. We explore how far free resources can take you toward training your own, before you have the compute to fine-tune a large LLM.

## Where it came from

The project started on 2026-08-24 as a single question: can one person run
**MiniMax H3**, an open video model, on a rented GPU for pocket money and
turn what it produces into something edited rather than merely generated?
The first commit was the generation path on RunPod plus an editing grammar
borrowed from [`Hao0321/video-autopilot-kit`](docs/attribution.md) — rules a
gate can fail a build on, instead of taste applied by hand.

The next day a LINE group became the reason to keep the pod busy: a message
in the chat, a clip back a few minutes later. That side grew fast —
image generation with **Flux.1-dev**, photo/audio/video understanding with
**moondream3**, **Qwen2-Audio** and **Qwen2.5-VL**, chat and prompt
rewriting with **gpt-oss-20b**, a one-minute drama pipeline — all on one
24 GB card, one model resident at a time.

Every step cost real money, and every claim about a model's speed or memory
was either something we had measured or something we had read. Keeping
those two apart became a rule of the repository, and then its purpose.

## What it is for

1. **Deploy open models on a rented GPU safely.** One pod, opened on demand
   and closed minutes after the last render; a licence-aware placement
   ladder; per-run, per-month and per-day money guards that run *before* a
   pod exists. Every expensive mistake on a GPU cloud is a quiet one, and
   this codebase is built to make them loud.
2. **Measure what each GPU actually does.** Every real render logs its
   seconds, cost and peak VRAM; a daily job folds them into a per-GPU-tier
   report; snapshots of the results are committed under
   [`assets/metrics/`](assets/metrics/) with a timestamp in every file name.
   Figures are graded — 📏 measured by us, `[reported]` quoted,
   `[speculative]` inferred — and only the first kind is ever exported.
3. **Give a group chat something to play with**, and show them, on the page
   each result links to, which GPU rendered it and what that GPU rents for.

## The repository

A monorepo of three independent Python packages, each with its own
`pyproject.toml`, lockfile, tests and layering contracts:

| package | role | start here |
|---|---|---|
| **`ai-studio`** (root) | The GPU side: the pod, the models behind one provider protocol, the money guards, the measurements. It knows nothing about who asked for a render. | this file · [`docs/`](docs/) · [`CLAUDE.md`](CLAUDE.md) |
| [**`fun_workflow/`**](fun_workflow/) | The request side: the LINE webhook, the queue, the worker that opens the pod, `/短劇` dramas, `/himonkey` chat, the status pages. Installs ai-studio from `..`. | [`fun_workflow/README.md`](fun_workflow/README.md) |
| [**`twin/`**](twin/) | The side quest: a personal digital-twin agent framework, built on free compute (Modal, Kaggle, Lightning) as far as that goes before fine-tuning a large model needs more. Its own stack and spec. | [`twin/README.md`](twin/README.md) |

The boundary between the first two is enforced, not described: only
`fun_workflow`'s command line may touch the pod runtime, and nothing in
ai-studio names a chat trigger, a delivery limit or a feature. What ai-studio
deliberately does not know, and where each of those things lives instead, is
listed in [`CLAUDE.md`](CLAUDE.md#architecture).

## Using it

**Without a GPU** — both packages run offline against synthetic providers
that honour the real protocol:

```bash
uv sync --group dev
uv run ai-studio doctor                                 # python, ffmpeg + filters, credentials, disk
uv run ai-studio generate "a baker opening the shutters" --provider stub
uv run ai-studio understand photo.jpg --kind image

cd fun_workflow && uv sync --group dev
uv run funapp drama-dryrun                              # the whole /短劇 pipeline with stub models and real ffmpeg
```

**With a GPU** — put a RunPod key in `.env` (see `.env.example`), then:

```bash
uv run ai-studio pod capacity        # what the licence-safe ladder can get right now, no spend
uv run ai-studio session open        # one pod, provisioned over SSH, terminates itself at the lease end
uv run ai-studio session status      # tier, $/hr, elapsed, spent
uv run ai-studio session close       # terminates; a stopped pod would still bill its disk
```

**As a bot** — `fun_workflow/README.md` covers the eleven triggers, the
LINE credentials, and the one-command installers for an always-on box.

**The numbers** — `uv run ai-studio bench` prints this month's measurements;
`uv run ai-studio metrics export` writes a timestamped snapshot under
`assets/metrics/`, and `metrics readme` renders the tables in
[`assets/metrics/README.md`](assets/metrics/README.md).

## Reading on

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | the layers and the one invariant everything hangs off |
| [docs/schedule.md](docs/schedule.md) · [docs/runpod.md](docs/runpod.md) | the pod's lifecycle and the money; the deployment runbook |
| [docs/observability.md](docs/observability.md) | tracing a request across services; the archive; the benchmark fold |
| `docs/model-*.md` | one per model: weights, licence, what we measured and what we have not |
| [docs/editing-grammar.md](docs/editing-grammar.md) | the editing rules — specified, not yet implemented |
| [fun_workflow/docs/](fun_workflow/docs/) | the bot in full; the drama pipeline |

## Licence

MIT — [LICENSE](LICENSE). MiniMax H3's licence excludes several
jurisdictions and Flux.1-dev's is non-commercial; see the model docs before
you deploy. Derived-work attribution in [NOTICE](NOTICE).
