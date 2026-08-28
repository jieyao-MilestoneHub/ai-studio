# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

This repository is a monorepo of three independent packages. This file covers
the root package, **ai-studio**. The other two live in their own directories
with their own `pyproject.toml`, `uv.lock`, tests and `CLAUDE.md`:

| directory | what | read |
|---|---|---|
| `fun_workflow/` | the LINE group's playground built *on* ai-studio: webhook, request queue, GPU worker, `/短劇`, `/himonkey`, the status page users open. Installs ai-studio editable from `..`. | [`fun_workflow/CLAUDE.md`](fun_workflow/CLAUDE.md) |
| `twin/` | a personal digital-twin agent framework, unrelated to ai-studio's code or stack | [`twin/CLAUDE.md`](twin/CLAUDE.md) |

When working under either directory, read its `CLAUDE.md`; nothing below
applies there except the platform traps.

### ai-studio

**GPU test deployment and measurement.** ai-studio owns the pod, the models,
the money, and the numbers — and nothing about who asked for a render:

- **Pod lifecycle** on RunPod (`runtime/pod.py`, `runtime/session.py`): the
  licence-safe capacity ladder, `ensure_pod` (the single path that ever
  creates a pod), provisioning over SSH (`deploy/pod_setup.sh` +
  `deploy/inference_server.py`), readiness, the idle reaper, close.
- **Model access**: **MiniMax H3** clips and **Flux.1-dev** images through
  ComfyUI (`comfy/`, `providers/comfyui.py`, `providers/flux.py`);
  **moondream3**, **Qwen2-Audio-7B-Instruct** and **Qwen2.5-VL-7B-Instruct**
  describing a photo/audio/video back, and **gpt-oss-20b** for chat and as
  the **prompt rewriter**, all through the pod-side inference server
  (`inference/client.py`, `providers/understanding.py`, `providers/chat.py`,
  `pipeline/pod_llm.py`). Every request is rewritten into its model's best
  input shape before it renders (`prompts/convert.py` H3 schema + community
  rules, `prompts/flux.py`, `prompts/understanding.py`); `pipeline/residency.py`
  is the one-card model hand-off.
- **Money**: per-run and calendar-month ceilings (`runtime/budget.py`), the
  daily pod-open cap (`runtime/opens.py`).
- **Measurement**: every real render logs one `benchmark.records` line;
  `ai-studio archive` folds them daily into `runs/benchmark/<YYYY-MM>.json`
  (`benchmark/report.py`); `benchmark/rates.py` reads that and the open
  session back out — the data behind any "what does this GPU cost and do"
  display, including the GPU tier / $/hr rows on fun_workflow's status page.
  `ai-studio bench` prints it.
- An editing grammar derived from
  [`Hao0321/video-autopilot-kit`](https://github.com/Hao0321/video-autopilot-kit)
  (MIT), **specified in `docs/editing-grammar.md` and not implemented**
  (`editing/format_policy.py` is the only built piece; `gates/`, `planner/`,
  `render/` are shells).

Not here, by design: anything that knows what a LINE group is. The queue,
worker loop, webhook, `/短劇` pipeline and screenwriter, the `/himonkey`
persona, the delivery index and the always-on host's installers are all in
`fun_workflow/`. ai-studio exposes what they need as plain functions
(`runtime.session.{ensure_pod,load_state,provision,wait_ready,
wait_understanding_ready,close_if_idle,touch_activity,log_reap}`,
`providers.registry.get_provider`, `pipeline.{pod_llm,residency}`,
`benchmark.*`, `media`, `paths`, `storage.archive.run_archive`) and never
imports back.

The three understanding models' licences: moondream3 unverified
(`docs/model-moondream3.md`); the two Qwen models are Apache-2.0 per their
cards (`docs/model-qwen2-audio.md`, `docs/model-qwen2.5-vl.md`). The models
they replaced, and why, are in `docs/model-qwen3-omni-captioner.md` and
`docs/model-tarsier2.md`. No serverless endpoint is involved (retired
2026-08-27; its client is deleted).

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

uv run ai-studio doctor                                             # python, ffmpeg + filters, credentials, logs/archive
uv run ai-studio bench                                              # this month's per-tier measurements + the open pod's rate
uv run ai-studio archive --dry-run                                  # what tonight's 03:00 archive would tar and prune
uv run ai-studio preflight --skip-suite                             # GPU-side checks: poster, every graph, placement ladder
uv run ai-studio generate "a test clip" --provider stub             # offline end-to-end
uv run ai-studio understand photo.jpg --kind image                  # offline understanding path
uv run ai-studio format yt_longform_1080p                           # inspect the delivery transform
uv run ai-studio pod capacity | up | status | down
uv run ai-studio session open | close | status | reap --hold
```

The request side is `funapp` (`cd fun_workflow && uv run funapp ...`); see
its `CLAUDE.md`. Both packages' sweeps are what `verify` runs.

Python is pinned to **3.13** (`.python-version`); runpod-flash and several deps
refuse 3.14+.

## Skills and agents

| use | when |
|---|---|
| `runpod-session` skill | any pod work — spin up, generate, shut down, "the pod is stuck" |
| `editing-rule` skill | touching `editing/`, `gates/`, or `docs/editing-grammar.md` |
| `verify` skill | before a commit or handing work back — runs both packages |
| `docs/observability.md` | tracing one request across services (`grep '"token":…' logs/*/*.jsonl`), what each record timestamps, the daily archive and its retention |
| `cost-guard` agent | reviewing anything that can spend money |
| `architecture-reviewer` agent | reviewing changes against the invariants below |
| `fun_workflow/` work | any task under `fun_workflow/` — read `fun_workflow/CLAUDE.md` first |
| `twin/` work | any task under `twin/` — read `twin/CLAUDE.md` first; it has its own skills/agents table |

RunPod's official plugin is installed; route through the `runpod` skill rather
than guessing at the API. When an MCP tool does not expose a field you need,
read `https://api.runpod.io/v2/openapi.json` — that is where the pod schema in
`runtime/pod.py` came from.

## Architecture

Layers, enforced by `import-linter` contracts in `pyproject.toml`:

```
L0 core → L1 config·benchmark → L2 media·storage → L3 llm·inference·comfy·providers
   → L4 pipeline (residency, pod_llm) → L5 runtime → L6 cli
prompts · editing · gates · planner · render sit beside the spine with their own contracts
```

Five contracts, each protecting something specific:

- the layered spine above
- `prompts` ⊬ `llm`/`providers`/`comfy`/`pipeline`/`runtime`/`storage` —
  prompt building does no I/O; it takes an `LlmClient` by protocol
- `editing` ⊬ `providers`/`render`/`runtime`/`storage`/`media` — the editing
  rules must be readable and testable with zero infrastructure
- `gates` ⊬ `providers`/`render`/`runtime` — **gates are pure functions of
  on-disk JSON**, so they run against fixtures with no GPU
- `render` ⊬ `providers` — swapping the model cannot touch editing

There is no longer a "bots/api are leaves" contract: those packages left the
tree. The equivalent rule now lives on the other side — only `fun_workflow.cli`
may import `ai_studio.runtime` or `ai_studio.cli`
(`fun_workflow/tests/unit/test_layering.py`).

### Invariants a linter cannot catch

**`ProviderCapabilities` lives in `core`, not `providers`.** That inversion is
what lets `editing.format_policy` and `planner` reason about the model's native
size and clip-length quantum without importing a backend; they read the
capabilities snapshot from `provider_manifest.json`. Do not move it.

**`runtime` depends on nothing that takes requests.** `ensure_pod` counts the
day's opens in its own ledger (`runs/.pod_opens.json`) and the reaper's
"is work waiting" is a `hold` bool the caller passes. The queue is the
request side's; the pod is ours.

**`benchmark` sits below `storage` and never imports `runtime`.**
`benchmark.rates.live_rate` takes a session-shaped object; the CLI and the
request side pass `runtime.session.load_state()`. That is what keeps the
report readable from the archive without a layer cycle.

**Assets resolve from the checkout, not the cwd.** `ai_studio.paths` finds
`workflows/` and `deploy/` from `__file__`; both packages are editable
installs. Nothing should ever say "run from the repo root" again.

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
  enforced by `runtime.budget.MonthlyBudgetGuard` inside
  `runtime.session.ensure_pod` — the single path that ever creates a pod,
  called by `funapp worker` the moment anything is queued, at any hour (there
  are no business hours since 2026-08-27; `funapp reap` closes a quiet pod
  after 5/10 min). It refuses outright if the month's remaining budget can't
  cover even a minimal session at the ladder's priciest rung, or shrinks the
  window if it can only partly cover the full one. `session open` is the
  manual path and goes through the same guard and the same daily cap.
- **`AI_STUDIO_MAX_POD_OPENS_PER_DAY`** (15) is counted in
  `runs/.pod_opens.json` — the backstop for a crash-looping worker, which
  the monthly guard cannot see. See `docs/schedule.md`.
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
- **`ruff format` cannot write in this sandbox**; `ruff check --fix --no-cache`
  can and is how import blocks get sorted. `--no-cache` is required.
- The Bash working directory **persists between calls** — `cd` back to the repo
  root or use absolute paths. `fun_workflow/` has its own venv: run its tools
  as `uv run --project fun_workflow ...` from the root or `cd` in first.
- `zoompan` and `minterpolate` are permanently banned
  (`editing/format_policy.BANNED_FILTERS`).

## Number honesty

Grade every figure: 📏 measured by us, `[reported]` quoted from someone else,
`[speculative]` inferred. **Most numbers in `docs/` are `[reported]`** — the H3
performance figures have not been verified on our own hardware. Do not promote
one to 📏 without measuring it. `runs/benchmark/<month>.json` (`ai-studio
bench`) is where 📏 numbers come from now; a person reads it and promotes.

## Attribution boundary

We inherit the upstream kit's **craft** — rhythm, transitions, captions, audio,
gate discipline. We do **not** inherit its **evidence epistemics**
(`proof_stage.py`, the risky-claim regex gate, "real screenshots only").
Those keep claims honest about *real footage*; our material is generative by
design, so they are a category error here rather than a standard we are failing.
See `docs/attribution.md`.
