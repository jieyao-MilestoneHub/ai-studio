# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This repository is a monorepo for AI model development. This file covers the
root package, **ai-studio**. A second, independent subsystem lives in
[`twin/`](twin/CLAUDE.md) — a personal digital-twin agent framework, currently
spec-only and unrelated to ai-studio's code or stack. Read `twin/CLAUDE.md`
when working under `twin/`; nothing below applies there.

### ai-studio

AI video generation on RunPod: **MiniMax H3** clips and **Flux.1-dev** images
generated through ComfyUI on a shared GPU pod, then assembled with an editing
grammar derived from
[`Hao0321/video-autopilot-kit`](https://github.com/Hao0321/video-autopilot-kit)
(MIT). A FastAPI service + LINE bot lets a group chat trigger either
(`/影片` for video, `/圖片` for image, photo then `/圖影` / `/圖圖` for
image-to-video / image-to-image; one spelling each, no aliases) — see
`docs/line-bot.md`.

Currently built: core model, format policy, H3 prompt builder, Flux prompt
builder, ComfyUI client, stub provider, pod lifecycle, CLI, FastAPI + LINE bot
(dual trigger, queue, status pages, monthly budget guard). **Not built:** the
editing grammar implementation, gate rules, planner, render, and any
image/video *understanding* path (generation-only today). The grammar is
specified in `docs/editing-grammar.md` and waiting.

## Commands

```bash
export PATH="/c/ffmpeg/ffmpeg-master-latest-win64-gpl/bin:$PATH"   # see Platform traps

uv sync --group dev

uv run pytest tests -q
uv run pytest tests/unit/test_format_policy.py -q                  # one file
uv run pytest tests -q -k turbo                                    # one rule
uv run pytest tests -q -m "not ffmpeg"                             # skip ffmpeg-dependent

uv run ruff check --no-cache src tests examples
uv run lint-imports                                                # the layering contracts
uv run mypy

uv run ai-studio doctor                                             # python, ffmpeg + filters, credentials
uv run ai-studio generate "a test clip" --provider stub             # offline end-to-end
uv run ai-studio format yt_longform_1080p                           # inspect the delivery transform
uv run ai-studio pod capacity | up | status | down
```

Python is pinned to **3.13** (`.python-version`); runpod-flash and several deps
refuse 3.14+.

## Skills and agents

| use | when |
|---|---|
| `runpod-session` skill | any pod work — spin up, generate, shut down, "the pod is stuck" |
| `editing-rule` skill | touching `editing/`, `gates/`, or `docs/editing-grammar.md` |
| `verify` skill | before a commit or handing work back |
| `cost-guard` agent | reviewing anything that can spend money |
| `architecture-reviewer` agent | reviewing changes against the invariants below |
| `twin/` work | any task under `twin/` — read `twin/CLAUDE.md` first; it has its own skills/agents table |

RunPod's official plugin is installed; route through the `runpod` skill rather
than guessing at the API. When an MCP tool does not expose a field you need,
read `https://api.runpod.io/v2/openapi.json` — that is where the pod schema in
`runtime/pod.py` came from.

## Architecture

Layers, enforced by `import-linter` contracts in `pyproject.toml`:

```
L0 core → L1 config·prompts·editing → L2 media·storage
   → L3 gates·providers·comfy·planner·render → L4 pipeline → L5 runtime·cli → L6 api·bots
```

Four contracts, each protecting something specific:

- `editing` ⊬ `providers`/`render`/`runtime`/`storage`/`media` — the editing
  rules must be readable and testable with zero infrastructure
- `gates` ⊬ `providers`/`render`/`runtime` — **gates are pure functions of
  on-disk JSON**, so they run against fixtures with no GPU
- `render` ⊬ `providers` — swapping the model cannot touch editing
- nothing ⊬ `api`/`bots` — phase 2 stays a leaf

### Invariants a linter cannot catch

**`ProviderCapabilities` lives in `core`, not `providers`.** That inversion is
what lets `editing.format_policy` and `planner` reason about the model's native
size and clip-length quantum without importing a backend; they read the
capabilities snapshot from `provider_manifest.json`. Do not move it.

**Absolute time exists in one file, produced by one function.** Authoring models
carry `segment_id`; `CaptionCue` has no `start`/`end` at all. Time is computed
only in `render.timeline.resolve_timeline` and written only to `offsets.json`.
Upstream shipped captions 2–3s out of sync from hand-split segments — binding to
an index makes that impossible rather than merely detectable.

**Fail loudly, never silently degrade.** Unknown colour key, platform, caption
kind, provider name, workflow binding — all raise. Upstream shipped three videos
where "gold" silently rendered white.

**Semantic names, not effect names.** Authors write `TransitionReason`;
renderers pick `TransitionKind`. One table maps between them.

**Gate ordering is architectural.** PRE gates (plan, format, prompt) are pure
functions of `plan.json` and run before any GPU-second is spent. At 2–6 minutes
of GPU per clip, a post-generation check is a receipt, not a check.

### The provider protocol

`submit` / `poll` / `fetch` / `cancel`, not one blocking `generate()`. It maps
one-to-one onto ComfyUI's `/prompt`, `/history`, `/view`, `/interrupt` — and it
has to, because **RunPod's pod proxy is severed by Cloudflare at ~100 seconds**
while an H3 clip takes 2–6 minutes. `ClipJob` serialises into `clips.json` so a
crashed run reattaches to in-flight jobs instead of paying twice; `fetch` is
separate so copying output off the pod is never the forgotten step.

Run artifacts live in `runs/<run_id>/`; see `docs/architecture.md`.

## Money

Every expensive mistake here is a quiet one.

- **`pod down` terminates. Stopping a pod still bills for its disk.** There is
  deliberately no `stop()` on `PodManager`.
- **Never configure an auto-deploy reservation** — it bills whenever capacity
  appears, including overnight. `pod capacity` raises rather than queueing.
- RTX 4090 stock is thin, and **MiniMax H3's licence excludes the US, EU, UK,
  and South Korea** — placement is a licensing decision. See
  `runtime.pod.LICENCE_SAFE_DATACENTERS` (Iceland and Norway are EEA, not EU;
  Romania and Sweden are EU).
- Host RAM must be ≥60 GB and is not selectable via API; `pod up` verifies and
  terminates a short host.
- `AI_STUDIO_MAX_COST_USD` is checked before submission — a **per-run** ceiling.
- `AI_STUDIO_MAX_MONTH_USD` (default $50) is a **calendar-month** ceiling,
  enforced by `runtime.budget.MonthlyBudgetGuard` inside `runtime.session.
  ensure_pod` — the single path that ever creates a pod, called by the worker
  loop the moment anything is queued, at any hour (there are no business
  hours since 2026-08-27; the reaper closes a quiet pod after 5/10 min) — necessary
  because the capacity ladder's own worst case already exceeds $50/month on
  GPU alone. It refuses the window outright if the month's remaining budget
  can't cover even a minimal session at the ladder's priciest rung, or shrinks
  the window if it can only partly cover the full one. There is no more
  "skip if the queue is empty" gate on `session open` — nothing opens a pod
  except a request, so that gate has nothing left to guard; see
  `docs/schedule.md`.
- **Flux.1-dev's licence is non-commercial**, separate from its geographic
  restriction (there is none) — see `docs/model-flux.md`.

## Two traps that look like wins

**The H3 turbo trap.** Driving the turbo LoRA through `LoraLoaderModelOnly`, or
sampling with `KSamplerSelect`, produces comb artifacts and banding **while
running ~4× faster** (2.53 vs 9.8 s/iter). Benchmark it and you conclude you
found a 6.3× speedup; you found the model skipping work. Enforced by
`comfy/validate.py`. If anyone reports suspiciously fast H3 numbers, ask which
sampler they used.

**Resolution is not the quality lever.** Same seed and scene, changing only the
prompt: free prose 26.0 → structured schema 367.6. Same prose at 5× the pixels:
no change. A blurry result is a prompt problem — use `ai_studio.prompts.h3`.

## Platform traps

- **On Windows, ffmpeg is installed but not on PATH** at
  `C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin`. Prepend it yourself in a bare
  shell (see the `export PATH=...` line under Commands), and if you want
  Claude-Code-invoked commands to see it too, put the prepend in
  `.claude/settings.local.json` (gitignored, per-machine) — **not** in the
  checked-in `.claude/settings.json`. Claude Code's `env` block sets values
  verbatim with no shell expansion, so a `;$PATH`/`:$PATH` suffix does not
  merge with the real PATH — it clobbers it. A Windows-only value there once
  broke every `PreToolUse:Bash` hook on Linux/macOS (`bash: not found`)
  because it replaced the entire PATH instead of extending it.
- **Always pass `encoding="utf-8"`** to `open()` and `subprocess`. The Windows
  default is cp950 and throws on any non-ASCII byte. Keep CLI output ASCII —
  em-dashes and bullets render as mojibake in the console.
- **`ruff --fix` and `ruff format` cannot write in this sandbox**; run
  `ruff check --no-cache` and apply fixes with the edit tools. `--no-cache` is
  required.
- The Bash working directory **persists between calls** — `cd` back to the repo
  root or use absolute paths.
- `zoompan` and `minterpolate` are permanently banned
  (`editing/format_policy.BANNED_FILTERS`).

## Number honesty

Grade every figure: 📏 measured by us, `[reported]` quoted from someone else,
`[speculative]` inferred. **Most numbers in `docs/` are `[reported]`** — the H3
performance figures have not been verified on our own hardware. Do not promote
one to 📏 without measuring it.

## Attribution boundary

We inherit the upstream kit's **craft** — rhythm, transitions, captions, audio,
gate discipline. We do **not** inherit its **evidence epistemics**
(`proof_stage.py`, the risky-claim regex gate, "real screenshots only"). Those
keep claims honest about *real footage*; our material is generative by design,
so they are a category error here rather than a standard we are failing. See
`docs/attribution.md`.
