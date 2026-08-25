# ai-studio

Generate video clips with **MiniMax H3** and images with **Flux.1-dev** on
RunPod GPUs, then assemble the video output into a finished piece using a
rigorous, mechanically-enforced editing grammar. A FastAPI service + LINE bot
lets a group chat trigger either in natural language — `生成`/`/gen` for
video, `畫圖`/`/img` for image.

The premise: an AI video model gives you *shots*. It does not give you an
*edit*. Everything interesting in this repo about the video path is that
second part — pacing, cuts, captions, loudness — encoded as rules that a gate
can fail a build on, rather than as taste applied by hand. Images are a
simpler path: one Flux.1-dev generation, one delivered file, no assembly step.

> **Status: early.** The RunPod/ComfyUI generation paths, the pod lifecycle,
> and the LINE bot are built and running. The editing grammar is currently
> **documented but not implemented** — see
> [docs/editing-grammar.md](docs/editing-grammar.md) and the roadmap below.
> Nothing here claims to work that hasn't been run.

---

## Try it without a GPU

The clip pipeline runs offline against a synthetic clip provider. No RunPod
account, no API key, no GPU, no money.

```bash
uv sync --group dev
uv run videogen doctor          # checks python, ffmpeg + required filters, credentials
uv run videogen generate "a baker opening the shutters before sunrise" --provider stub
uv run videogen format yt_longform_1080p   # how 864x480 maps onto the delivery canvas
uv run python examples/build_prompt.py     # the MiniMax H3 structured prompt builder
```

You need `ffmpeg` on `PATH` with `libass`. `videogen doctor` tells you exactly
which filters are missing if any are.

This only exercises the video (H3) path — `generate` always builds a clip
request and writes an mp4. Flux.1-dev images have no offline stub; they run
through the LINE bot's queue against a live pod — see
[docs/line-bot.md](docs/line-bot.md) and
[docs/model-flux.md](docs/model-flux.md).

---

## Architecture

```
L0  core                          pure data model — imports nothing internal
L1  config · prompts · editing     policy tables, H3 + Flux prompt schemas, editing rules
L2  media · storage                artifact stores, ffmpeg invocation
L3  gates · providers · comfy       rule checks, clip/image backends, ComfyUI + LLM protocols
    · llm · planner · render
L4  pipeline                       request queue, drain loop, stage graph
L5  runtime · cli                  pod lifecycle, command line
L6  api · bots                     FastAPI service + LINE bot
```

Four import rules are enforced in CI by `import-linter`, and they are the
architecture rather than a style preference:

| Rule | Why |
|---|---|
| `editing` may not import `providers`, `render`, `runtime`, `storage`, `media` | The editing rules must be readable and testable with no infrastructure. |
| `gates` may not import `providers`, `render`, `runtime` | Gates are pure functions of on-disk JSON, so they run against fixtures. |
| `render` may not import `providers` | Swapping the model cannot touch the editing layer. |
| nothing may import `api` or `bots` | Phase 2 is a leaf — additive by construction, not just by intent. |

`ProviderCapabilities` lives in `core`, not `providers`. That is what lets the
format policy and planner reason about a model's native size and clip-length
quantum without importing a backend. See
[docs/architecture.md](docs/architecture.md).

---

## The editing grammar

This is the substance of the video path.
**→ [docs/editing-grammar.md](docs/editing-grammar.md)**

Every rule carries four fields — *rule + parameters*, *landing module*,
*mechanism*, *source* — and every rule that lands gets a gate and a fixture
that must fail without it.

The grammar is derived from
[Hao0321/video-autopilot-kit](https://github.com/Hao0321/video-autopilot-kit)
(MIT). We inherit its craft — rhythm, transitions, captions, audio, gate
discipline — and deliberately **not** its evidence epistemics, which exist to
keep claims honest about *real footage* and are a category error for
generative material. The boundary is documented in
[docs/attribution.md](docs/attribution.md).

---

## Providers

| Provider | Protocol | Model | GPU | Use |
|---|---|---|---|---|
| `stub` | clip | — | none | Offline development and CI. Synthetic clips at the real native spec. |
| `comfyui` | clip | MiniMax H3 | yours | Video, driven through ComfyUI's HTTP API on a RunPod pod. |
| `flux` | image | Flux.1-dev | yours | Images, same ComfyUI HTTP API and pod, driven by the LINE bot's queue rather than the local CLI. |

A clip provider exposes `submit` / `poll` / `fetch` / `cancel` rather than one
blocking `generate()`. An H3 clip takes 2–6 minutes and RunPod's pod proxy is
cut off by Cloudflare at ~100 seconds, so holding a request open is not an
option — and splitting `fetch` out is what stops a clip being left on a pod
that is about to be terminated. `flux` implements the equivalent shape for a
still image.

⚠️ **Flux.1-dev's licence is non-commercial** — a separate restriction from
MiniMax H3's geographic exclusion below. See
[docs/model-flux.md](docs/model-flux.md).

---

## The LINE bot

**→ [docs/line-bot.md](docs/line-bot.md)** for the full design.

A FastAPI service runs 24/7 on a small VPS while the GPU pod exists only for a
scheduled daily window (see [docs/schedule.md](docs/schedule.md)). A LINE
group triggers either generator in natural language — `生成`/`/gen` queues a
video, `畫圖`/`/img` queues an image — and gets a link back once the window
renders it. Requests land in a SQLite queue immediately, because LINE needs
its `200` in under two seconds; an LLM converts each one into a structured
H3 or Flux prompt in the background, so a bad prompt never occupies GPU time.

---

## Running on RunPod

**→ [docs/runpod.md](docs/runpod.md)** for the deployment runbook,
[docs/model-h3.md](docs/model-h3.md) for MiniMax H3 specifics, and
[docs/model-flux.md](docs/model-flux.md) for Flux.1-dev.

Things to know before you spend anything:

- **RTX 4090 stock is thin.** Check availability before deploying, and never
  configure an auto-deploy reservation — it will start billing when capacity
  appears, including overnight.
- **`pod down` terminates.** Stopping a pod does not stop disk billing.
- **MiniMax H3's licence excludes the US, EU, UK, and South Korea.** Placement
  is a licence question, not just a latency one.
- **Flux.1-dev's licence is non-commercial**, independent of geography —
  confirm the bot's actual use before shipping this to any commercial context.
- `VIDEOGEN_MAX_MONTH_USD` (default $50) is a calendar-month budget ceiling,
  checked before a session is allowed to open a pod at all.

---

## Roadmap

Built:

- [x] Project skeleton, layering contracts, core data model
- [x] Format policy (delivery-canvas mapping)
- [x] Structured prompt builders — MiniMax H3 schema and Flux.1-dev
- [x] ComfyUI client + workflow graph validation
- [x] Pod lifecycle with RAM verification and terminate-by-default
- [x] `stub` and `comfyui` clip providers, `flux` image provider
- [x] FastAPI service + LINE bot — dual trigger, request queue, status pages,
      monthly budget guard

Not built:

- [ ] Editing grammar implementation
- [ ] Gate rules — the shell exists in `gates/core.py`; no rules run yet
- [ ] Planner
- [ ] Render

---

## Licence

MIT — see [LICENSE](LICENSE). Derived-work attribution in [NOTICE](NOTICE).
