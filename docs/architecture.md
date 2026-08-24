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
L1  config · prompts · editing   settings, H3 prompt schema, editing rules
L2  media · storage          ffmpeg invocation, artifact stores
L3  gates · providers · comfy    rule checks, clip backends, ComfyUI protocol
    planner · render
L4  pipeline                 stage graph, run/resume
L5  runtime · cli            pod lifecycle, command line
L6  api · bots               phase 2 — nothing imports these
```

Enforced by `import-linter` in `pyproject.toml`, run in CI:

| contract | why |
|---|---|
| `editing` ⊬ `providers`, `render`, `runtime`, `storage`, `media` | the editing rules must be readable and testable with zero infrastructure |
| `gates` ⊬ `providers`, `render`, `runtime` | preserves the invariant above |
| `render` ⊬ `providers` | swapping the model cannot touch editing |
| nothing ⊬ `api`, `bots` | phase 2 stays a leaf, so adding it is additive |

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
| `prompts` | built — full MiniMax H3 schema, 15 tests |
| `editing` | `format_policy` built (12 tests); the grammar is **specified, not implemented** |
| `media` | built — ffmpeg/ffprobe invocation |
| `storage` | `local` built; `s3` pending |
| `comfy` | built — client, graph binding, turbo validation (17 tests) |
| `providers` | `stub` and `comfyui` built |
| `gates` | shell built; no rules yet |
| `planner`, `render`, `pipeline` | not started |
| `runtime` | `pod` built against the live REST v2 schema |
| `cli` | `doctor`, `format`, `generate`, `pod {capacity,up,status,down}` |
| `api`, `bots` | reserved, empty |
