# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI video generation on RunPod: **MiniMax H3** clips generated through ComfyUI on
a GPU pod, then assembled with an editing grammar derived from
[`Hao0321/video-autopilot-kit`](https://github.com/Hao0321/video-autopilot-kit)
(MIT). Phase 2 adds FastAPI + a LINE bot so a group chat can trigger generation
in natural language.

Currently built: core model, format policy, H3 prompt builder, ComfyUI client,
stub provider, pod lifecycle, CLI. **Not built:** the editing grammar
implementation, gate rules, planner, render, FastAPI, LINE. The grammar is
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

uv run videogen doctor                                             # python, ffmpeg + filters, credentials
uv run videogen generate "a test clip" --provider stub             # offline end-to-end
uv run videogen format yt_longform_1080p                           # inspect the delivery transform
uv run videogen pod capacity | up | status | down
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
- `VIDEOGEN_MAX_COST_USD` is checked before submission.

## Two traps that look like wins

**The H3 turbo trap.** Driving the turbo LoRA through `LoraLoaderModelOnly`, or
sampling with `KSamplerSelect`, produces comb artifacts and banding **while
running ~4× faster** (2.53 vs 9.8 s/iter). Benchmark it and you conclude you
found a 6.3× speedup; you found the model skipping work. Enforced by
`comfy/validate.py`. If anyone reports suspiciously fast H3 numbers, ask which
sampler they used.

**Resolution is not the quality lever.** Same seed and scene, changing only the
prompt: free prose 26.0 → structured schema 367.6. Same prose at 5× the pixels:
no change. A blurry result is a prompt problem — use `videogen.prompts.h3`.

## Platform traps

- **ffmpeg is installed but not on PATH** at
  `C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin`. `.claude/settings.json`
  prepends it; export it manually in a bare shell.
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
