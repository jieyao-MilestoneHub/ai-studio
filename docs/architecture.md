# Architecture

## The one invariant everything hangs off

> **Every gate is a pure function of JSON artifacts on disk. Gates never import
> providers, render, or runtime.**

That single rule buys four things at once:

- gates run against fixture directories with no GPU, no ffmpeg, no account
- a contributor can add an editing rule without understanding RunPod
- any stage can be re-run or re-checked in isolation
- a failed run stays fully diagnosable from `runs/<id>/` afterwards

Everything below is arranged to preserve it.

## Layers

```
L0  core                     data model only; imports nothing internal
L1  config · benchmark ·     settings; render records, the monthly report and live
    prompts · editing        rates; H3 + Flux prompt schemas; editing rules
L2  media · storage          ffmpeg invocation, artifact stores, the daily archive
L3  gates · providers · comfy    rule checks, clip/image/understanding/chat backends,
    inference · llm · planner ·   ComfyUI + the pod-side inference-server client +
    render                        the offline scripted LLM
L4  pipeline                 model residency hand-off, the pod-side prompt rewriter
L5  runtime · cli            pod lifecycle, budget, open ledger, command line

fun_workflow/ (own package)  api · bots · pipeline(queue, worker, drain, drama)
                             · prompts(drama, chat) · storage(index, gc) · cli —
                             installs ai-studio editable; only its `cli` may
                             import `ai_studio.runtime`
```

Enforced by `import-linter` in `pyproject.toml`, run in CI:

| contract | why |
|---|---|
| `editing` ⊬ `providers`, `render`, `runtime`, `storage`, `media` | the editing rules must be readable and testable with zero infrastructure |
| `gates` ⊬ `providers`, `render`, `runtime` | preserves the invariant above |
| `render` ⊬ `providers` | swapping the model cannot touch editing |
| `prompts` ⊬ `llm`, `providers`, `comfy`, `pipeline`, `runtime`, `storage` | prompt building does no I/O |

The former "nothing imports `api`/`bots`" contract is now a package boundary:
those modules live in `fun_workflow/`, which depends on ai-studio and never
the reverse. Its own guardrail (`fun_workflow/tests/unit/test_layering.py`)
keeps `ai_studio.runtime` reachable only from its composition root.

### The dependency inversion that makes it work

`ProviderCapabilities` lives in **`core`**, not in `providers`.

`editing.format_policy` needs the model's native size to derive the delivery
transform. `planner` needs its clip-length quantum to refuse an unrepresentable
shot. If either asked a provider directly, swapping the model would ripple
through the editing layer — exactly what the layering is meant to prevent.

Instead a provider publishes a capabilities snapshot into
`provider_manifest.json`, and everything downstream reads the snapshot. Nothing
above `core` needs to know a provider exists.

## Two timing disciplines

**1. Absolute time exists in one file.**

Authoring models carry `segment_id` and no timestamps — `CaptionCue` has no
`start` or `end` fields at all. Absolute time is computed by one function
(`render.timeline.resolve_timeline`) and written to one file (`offsets.json`).

Upstream shipped captions 2–3 seconds out of sync because segments were split by
hand and the timings drifted. Binding to an index makes that class of bug
*structurally impossible* rather than merely detectable, and collapses "a caption
may never straddle a cut" into an assertion inside a single function.

**2. Semantic names, not effect names.**

A `Scene` declares a `TransitionReason`; a renderer picks a `TransitionKind`.
One table maps between them. You cannot ask for a wipe without first saying what
the wipe means.

## Run artifacts

```
runs/<run_id>/
  spec.json                 the submitted VideoSpec, verbatim
  plan.json                 scenes + shots + semantics. NO timings.
  provider_manifest.json    capabilities snapshot + FormatPlan + cost estimate
  clips.json                per shot: job id, state, key, sha256, cost   <- resume point
  offsets.json              the EDL: absolute times, cuts, transitions
  captions.json / .ass
  render_manifest.json      the literal argv of every ffmpeg invocation
  probe.json                ffprobe + loudness measurement of the output
  gates/*.json              one GateReport each
  result.json               final URI, poster, cost, waivers
  state.json                completed stages + input hashes, for resume
```

Two of these earn their place specifically:

- **`clips.json`** is the resume point that matters. One H3 clip is 2–6 minutes
  of GPU time; regenerating because assembly crashed is the expensive failure.
- **`render_manifest.json`** records literal argv, which is what lets
  `grammar_gate` assert over what a run *actually did*. It turns "we banned
  `zoompan`" from a code-review norm into an executable check.

## Provider protocol

```python
submit(request) -> ClipJob      # POST /prompt      -> prompt_id
poll(job)       -> ClipJob      # GET  /history/<id>
fetch(job, dst) -> ClipAsset    # GET  /view?filename=...
cancel(job)                     # POST /interrupt
```

Four methods rather than one blocking `generate()`, for reasons that are all
about money:

1. RunPod's pod proxy is severed by Cloudflare at ~100 seconds; an H3 clip runs
   for 2–6 minutes. There is no version of "just await the render".
2. `ClipJob` serialises into `clips.json`, so a crashed orchestrator reattaches
   to in-flight GPU jobs instead of paying twice.
3. Outputs die with the pod, and RunPod public-endpoint URLs expire after seven
   days. Making the copy-into-our-storage step its own method is what stops it
   being the step everyone forgets.

### A third quartet: `UnderstandingProvider`

Alongside `ClipProvider`/`ProviderCapabilities`/`ClipRequest`/`ClipJob`/
`ClipAsset` and their `Image*` siblings, `core/understanding_spec.py` and
`providers/base.py`'s `UnderstandingProvider` give the LINE bot's `/說圖`
`/說音` `/說影` (media in, a text description out) the same submit/poll/
fetch/cancel shape — same Cloudflare-proxy reasoning (a cold model load for
a >16GB understanding model is not guaranteed to fit the ~100s window even
though a warm caption call would). It is a *sibling* type, not a merge into
`ClipRequest`/`ImageRequest`: a description has no width, height, fps, or
duration to produce, so forcing it through either would mean inventing
values that mean nothing — the same argument `image_provider_spec.py` makes
for a still image having no frame count.

Two things make this quartet different from the other two:

- **No output file.** `fetch()` takes no `dest` — the result is text,
  already captured in `job.raw` at poll time, not a file to download.
- **A second, separate server process on the pod.** moondream3-preview,
  Qwen3-Omni-Captioner, and Tarsier2 are not ComfyUI nodes, so they are not
  driven through `comfy/client.py` at all. `deploy/inference_server.py`
  (port 8189) is a small always-resident process with its own lazy
  load/unload discipline — see `docs/schedule.md`'s "the GPU hand-off"
  section for how it shares the one 24GB card with ComfyUI's H3/Flux
  checkpoint without ever holding both at once.

### A fourth quartet: `ChatProvider`

`core/chat_spec.py` (`ChatCapabilities`/`ChatRequest`/`ChatJob`/`ChatAsset`,
`MediaKind.CHAT`) and `providers/base.py`'s `ChatProvider` give `/himonkey`
(text in, text out, gpt-oss-20b) the same shape again, and for the same
reason: a cold load of a 16GB checkpoint does not fit the ~100s proxy
window. Like understanding it has no output file and is served by the same
`deploy/inference_server.py` process — it is the fourth backend behind the
one `ModelSlot`, so `pipeline.drain.make_room_for()` treats it as the same
evictable side as the three understanding kinds. What is *different*: it
carries a rolling history (`JobQueue.recent_chat_turns()`, last 10 turns)
into each request, and its spend is capped by its own sub-budget
(`AI_STUDIO_MAX_CHAT_MONTH_USD`, checked in `pipeline.drain.render_chat`
before submit) and per-user daily cap, separate from the video/image ones.

## Gate ordering is architectural

Gates split PRE / POST. PRE (plan, format, prompt) are pure functions of
`plan.json` and run **before any GPU-second is spent**; POST (pace, caption,
audio, grammar, delivery) run after assembly.

This is not tidiness. At 2–6 minutes of GPU per clip, a gate that runs after
generation is a receipt, not a check.

## Current state

| layer | status |
|---|---|
| `core` | built — models, capabilities, ids, timecode, errors |
| `config` | built |
| `prompts` | built — MiniMax H3 structured schema, the Flux.1-dev prompt builder, the understanding models' questions (the `/短劇` screenwriter is `fun_workflow/prompts/drama.py`) |
| `benchmark` | built — `records` (the log line a render emits), `report` (`runs/benchmark/<month>.json`, folded daily by `archive`), `rates` (live GPU rate + per-tier means) |
| `editing` | `format_policy` built; the grammar is **specified, not implemented** |
| `media` | built — ffmpeg/ffprobe invocation, still-image probing for Flux, two-pass `loudnorm` and concat for `/短劇` |
| `storage` | `local` built; `s3` pending |
| `comfy` | built — client, graph binding, turbo validation |
| `inference` | built — `InferenceClient`, the HTTP surface for `deploy/inference_server.py`'s three understanding models |
| `llm` | `scripted.py` only — the offline `LlmClient` for tests and dry runs; the production rewriter is `pipeline.pod_llm` |
| `providers` | `stub`/`stub-understanding` and `comfyui` (clip, MiniMax H3) built; `flux` (image, Flux.1-dev) built; `understand-{image,audio,video}` (moondream3/Qwen3-Omni-Captioner/Tarsier2) built; `chat` (gpt-oss-20b) built |
| `gates` | shell built; no rules yet |
| `planner`, `render` | not started |
| `pipeline` | `residency` (one-card model hand-off) and `pod_llm` (gpt-oss-20b on the pod as an `LlmClient`). The request queue, worker with its prepare phase, drain loop and the resumable `/短劇` stage machine are `fun_workflow/pipeline/` |
| `checks` | built — the pre-launch checklist machinery both packages' `preflight` commands build on |
| `runtime` | `pod`, `session` (idle grace recorded per activity; `provision(extras=)` ships `pod_setup.d/` extensions), `budget` and `opens` (the daily pod-open ledger) built against the live REST v2 schema |
| `cli` | `doctor`, `bench`, `archive`, `preflight`, `format`, `generate`, `understand`, `rewrite`, `pod {capacity,placement,up,status,down}`, `session {open,close,status,reap}` |
| `fun_workflow/` | built — FastAPI webhook/status/file service, the LINE bot, queue, worker, drain, `/短劇`; console script `funapp` |
