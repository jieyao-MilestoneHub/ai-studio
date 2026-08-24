# video-gen

Generate video clips with **MiniMax H3** on RunPod GPUs, then assemble them into
a finished piece using a rigorous, mechanically-enforced editing grammar.

The premise: an AI video model gives you *shots*. It does not give you an *edit*.
Everything interesting in this repo is about the second part — pacing, cuts,
captions, loudness — encoded as rules that a gate can fail a build on, rather
than as taste applied by hand.

> **Status: early.** The RunPod/ComfyUI generation path and the project skeleton
> are being built first. The editing grammar is currently **documented but not
> implemented** — see [docs/editing-grammar.md](docs/editing-grammar.md) and the
> roadmap below. Nothing here claims to work that hasn't been run.

---

## Try it without a GPU

The whole pipeline runs offline against a synthetic clip provider. No RunPod
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

---

## Architecture

```
L0  core                          pure data model — imports nothing internal
L1  config · prompts · editing     policy tables, prompt schema, editing rules
L2  storage                        artifact stores
L3  gates · providers · comfy      rule checks, clip backends, ComfyUI protocol
    planner · render
L4  pipeline                       stage graph, run/resume
L5  runtime · cli                  pod lifecycle, command line
L6  api · bots                     phase 2 — nothing imports these
```

Four import rules are enforced in CI by `import-linter`, and they are the
architecture rather than a style preference:

| Rule | Why |
|---|---|
| `editing` may not import `providers`, `render`, `runtime`, `storage` | The editing rules must be readable and testable with no infrastructure. |
| `gates` may not import `providers`, `render`, `runtime` | Gates are pure functions of on-disk JSON, so they run against fixtures. |
| `render` may not import `providers` | Swapping the model cannot touch the editing layer. |
| nothing may import `api` or `bots` | Phase 2 stays a leaf; adding it is additive. |

`ProviderCapabilities` lives in `core`, not `providers`. That is what lets the
format policy and planner reason about the model's native size and clip-length
quantum without importing a backend. See
[docs/architecture.md](docs/architecture.md).

---

## The editing grammar

This is the substance of the project.
**→ [docs/editing-grammar.md](docs/editing-grammar.md)**

Every rule carries four fields — *rule + parameters*, *landing module*,
*mechanism*, *source* — and every rule that lands gets a gate and a fixture that
must fail without it.

The grammar is derived from
[Hao0321/video-autopilot-kit](https://github.com/Hao0321/video-autopilot-kit)
(MIT). We inherit its craft — rhythm, transitions, captions, audio, gate
discipline — and deliberately **not** its evidence epistemics, which exist to
keep claims honest about *real footage* and are a category error for generative
material. The boundary is documented in
[docs/attribution.md](docs/attribution.md).

---

## Providers

| Provider | GPU | Use |
|---|---|---|
| `stub` | none | Offline development and CI. Synthetic clips at the real native spec. |
| `comfyui` | yours | MiniMax H3 driven through ComfyUI's HTTP API on a RunPod pod. |

A provider exposes `submit` / `poll` / `fetch` / `cancel` rather than one
blocking `generate()`. An H3 clip takes 2–6 minutes and RunPod's pod proxy is
cut off by Cloudflare at ~100 seconds, so holding a request open is not an
option — and splitting `fetch` out is what stops a clip being left on a pod
that is about to be terminated.

---

## Running on RunPod

**→ [docs/runpod.md](docs/runpod.md)** for the deployment runbook and
[docs/model-h3.md](docs/model-h3.md) for the model specifics.

Three things to know before you spend anything:

- **RTX 4090 stock is thin.** Check availability before deploying, and never
  configure an auto-deploy reservation — it will start billing when capacity
  appears, including overnight.
- **`pod down` terminates.** Stopping a pod does not stop disk billing.
- **MiniMax H3's licence excludes the US, EU, UK, and South Korea.** Placement
  is a licence question, not just a latency one.

---

## Roadmap

- [x] Project skeleton, layering contracts, core data model
- [ ] Structured prompt builder (MiniMax H3 schema)
- [ ] ComfyUI client + workflow graph validation
- [ ] Pod lifecycle with RAM verification and terminate-by-default
- [ ] **Milestone: real H3 inference producing a playable clip**
- [ ] Editing grammar implementation
- [ ] Gate layer
- [ ] FastAPI service
- [ ] LINE group integration

---

## Licence

MIT — see [LICENSE](LICENSE). Derived-work attribution in [NOTICE](NOTICE).
