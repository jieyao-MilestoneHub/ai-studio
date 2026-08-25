# PLAN — the models: what they actually require, and what now enforces it

**Companion to [`PLAN.md`](PLAN.md), not a replacement.** That file is the goal
chain (LINE → API → prompt → RunPod → media → group) and its phases are tracked
by their own PRs. This one is about the two models underneath it: what the
research says, which of this repo's beliefs were wrong, and what has been built
so the wrong ones cannot come back.

Grading throughout follows [`docs/attribution.md`](docs/attribution.md):
📏 measured by us · `[verified]` read from primary source (a model card,
MiniMax's repo, ComfyUI's own code) · `[reported]` someone else measured it ·
`[speculative]` inferred.

---

## Why this exists

The chain is implemented and **not one GPU-second has ever been spent.** Every
performance figure in `docs/` is `[reported]` or `[speculative]`; neither
workflow has been submitted to a real ComfyUI; the Flux LoRA has never been
loaded.

The budget makes that expensive to get wrong: **$1.556 of approved spend against
an L40S at $1.004/hr** — one 40-minute window. A pitfall found *during* that
window is not a bug report, it is the whole budget.

So the work is not "add features". It is: find the pitfalls on paper, then make
each one **structurally impossible** before the money is spent — the repo's own
stated preference, *structural prohibition beats a lint rule*.

## Scope, agreed before starting

| question | decision |
|---|---|
| Model choice | **Locked.** MiniMax H3 + Flux.1-dev + NSFW LoRA. Not a selection exercise |
| Generation parameters (steps, resolution, guidance) | **Frozen** until the first live run measures them. Differences get recorded as pending A/Bs, never changed blind |
| What may be fixed | Objective errors only — missing weights, filename mismatches, a wrong billing rate, an illegal frame count |
| Post-generation verification | Minimal viable, built on the `ffprobe` wrappers already in `media.py` |

---

## Part 1 — Five things this repo stated as fact that are wrong

Recorded rather than silently edited, because "we used to believe X" is what
stops the same wrong thing being re-derived. Full detail with sources lives in
[`docs/model-h3.md`](docs/model-h3.md) and
[`docs/model-flux.md`](docs/model-flux.md); this is the index.

| # | the claim | verdict |
|---|---|---|
| 1 | Host RAM **must be ≥ 60 GB** | **Contradicted.** ComfyUI [#15488](https://github.com/Comfy-Org/ComfyUI/issues/15488) documents H3 killing the GPU *because* 64 GB was visible; capping to 32 GB gave 29 clean runs. Our 31 GB crash is real 📏 but the inference does not follow. **The guard is unchanged** — relaxing a money guard on two issue threads is worse than carrying a stale one |
| 2 | **864×480 is a native canvas** | **Wrong.** Native is a 768 px short edge → 1344×768 `[verified]`. 864×480 is legal (multiples of 32) and ~2.3× faster `[reported]`, but off-native. Max area is ~1.03 MP, and ComfyUI's own "1.0 Megapixel" preset **exceeds it** |
| 3 | **124 frames is the validated floor** | **Wrong as a model constraint.** ComfyUI's minimum is 5 `[verified]`. 124 is the turbo LoRA's trained lower bound `[reported]`. The `17k+5` grid itself is confirmed `[verified]` |
| 4 | Comb artifacts = wrong wiring; 4× speed = "skipping work" | **Single-source, and the cause is disputed.** One blog. Every other source describes *audio* degradation, and the real sampler bug was fixed upstream 2026-08-06 (PR [#15243](https://github.com/Comfy-Org/ComfyUI/pull/15243)). A likelier mechanism `[speculative]`: a LoRA on an int8 checkpoint forces dequantisation to bf16 — **and the bf16 path has a black-frames bug** ([#15563](https://github.com/Comfy-Org/ComfyUI/issues/15563)) |
| 5 | The Flux LoRA **may not load** (`[speculative]` risk) | **Resolved — it loads.** `comfy/lora.py` has an explicit Flux branch registering `transformer.{key}`, and `flux_to_diffusers` patches split q/k/v into slices of the fused `linear1`. The keys that genuinely fail carry a `base_model.model.` infix; ours does not |

### Two blockers answered for free

- **#8 / Phase 7.0 — CUDA.** `runpod/comfyui:cuda13.0` **exists** `[verified]`
  on Docker Hub; RunPod's *docs* list only `latest` and `cuda12.8`, which is
  what we were reading. `--min-cuda-version 13.0` is not a dead flag. CUDA 13
  is not *required* — it is what makes `int8_convrot` fast.
- **#5 — launch flags.** `--fast-disk`, `--use-sage-attention` and
  `--reserve-vram` are all real `[verified]` in `comfy/cli_args.py`. Our hedge
  was over-cautious.

### One suspicion checked and cleared

Research flagged that H3's official cut syntax is an **inline timestamp in the
prose**, not a field — so a serialiser emitting a `cut_at_s` key would produce
cuts the model never sees. **Ours is already correct**, verified by running it:

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot of a bakery.
[Shot 2] At 00:03.500, the camera cuts to a close-up of the bread.
```

That is the guide's exact phrasing. It is the "`render()` is the only thing that
produces the final string" invariant paying for itself.

### New risks now on record

- **ComfyUI v0.32.0 is a trap** `[verified]`: ~4× slowdown at full resolution,
  unfixed, resolution-dependent so it would hit us at exactly our canvases.
  `deploy/pod_setup.sh` upgrades **unpinned** — the pod runs whatever shipped
  that morning.
- **H3 licence terms we never recorded**: excluded territories are about where
  the *product and its users* are, not only the GPU; >$20M revenue needs
  authorisation; commercial UI must display "MiniMax H3"; outputs need
  AI-generation identifiers. Only **H3-Base** is open — local ComfyUI tops out
  at 768p, the "2K" headline is the hosted product.
- **`black-forest-labs/FLUX.1-dev` is gated.** 401 without `HF_TOKEN` and a
  one-time licence acceptance.
- **Neither a black image nor a no-op LoRA is an error.** ComfyUI saves the
  first and warns-but-returns on the second.

---

## Part 2 — What was built

### The core finding: nothing validated a single numeric value

`validate_graph` enforces nine rules and **every one is about wiring or class
names**. `test_ui_workflow.py` pinned only the two values that differ between
the H3 pair. Resolution, `length`, fps, steps, LoRA strength, the scheduler, the
weight filenames — none were pinned by any test, and all could change silently.

Meanwhile one constant lived in up to four unlinked places:

| where | what |
|---|---|
| both H3 workflow JSONs | `length: 124` |
| `pipeline/convert_worker.py` | `DEFAULT_DURATION_S = 124 / 24` |
| `pipeline/drain.py` | `max(frames, 124) / caps.native_fps` — a second, independent 124 |
| `providers/comfyui.py` | `round(duration_s * fps)` — the arithmetic |
| `providers/comfyui.py` | `clip_duration_quantum=None` ← **the field designed for this rule, switched off** |

The rule that produced 124 lived only in a docstring. Consequence, verified:
`ai-studio generate` defaults to `--seconds 5.0` → **120 frames**, which is not
on the grid, and nothing caught it.

### 1. `core/model_profile.py` — one source of truth

`ProviderCapabilities` describes one *configured instance*. A `ModelProfile`
describes the **model**: which canvases exist, which frame counts are legal,
which weight file belongs to which quantisation. In `core` for the same reason
`ProviderCapabilities` is (`CLAUDE.md`: "Do not move it") — it is what lets
`editing`, `planner` and `gates` reason about a model without importing a
backend.

`FrameGrid` carries `17k+5` as data, with `minimum` (5, the model's) and
`recommended_minimum` (124, the adapter's) recorded **separately** so they can
never be conflated again. All three provider capability functions are now
derived from a profile instead of hand-written.

### 2. Guardrails — where each pitfall became impossible

| pitfall | mechanism |
|---|---|
| An off-grid frame count reaches the GPU | `ComfyUIProvider.submit` raises. `ClipRequest` carries `duration_s`, never frames, so this is the boundary where provider-agnostic becomes H3-specific — and it **refuses rather than snaps**, because a program that submitted 120 frames did not mean 124. The CLI snaps *explicitly and out loud*, because a person typing `--seconds` wants the nearest thing that works |
| Unknown canvas → silently wrong economics | `require_canvas` raises. The old `MEASURED_LATENCY_S.get((w,h), 300.0)` handed an off-table canvas 1344×768's timing, so its cost estimate *and* its job timeout were someone else's numbers |
| Workflow drifts from the profile | 12 conformance tests: `length` on-grid, canvas known, fps, steps, LoRA name and strength, DiT filename ↔ quantisation, and that the two H3 graphs differ **only** in those two nodes |
| A graph loads a file the pod never downloads | Generalised from the Flux-LoRA check: **every `*_name` in every workflow must appear in `pod_setup.sh`**. Keyed off the graphs, so the next node added is covered without anyone remembering. This is the check that would have caught #19 |
| A broken render reaches the group | `gates/output_gate.py` — below |

### 3. `provider_manifest.json` — the missing half, built

Named in `core/provider_spec.py`, `providers/base.py`, `docs/architecture.md`
and `CLAUDE.md` ("Do not move it") — and nothing had ever written it. The
inversion was real at the type level; only serialisation was missing.

It matters because gates are, by contract, *pure functions of on-disk artifacts*
and may not import `providers`. Without this file a gate cannot learn what the
model was configured to produce. `write_provider_manifest()` now publishes the
capabilities snapshot plus a workflow digest, so two runs with the same manifest
provably submitted the same graph.

### 4. `gates/output_gate.py` — the first real rule body

`gates/core.py` has been a complete shell with **zero rules** since the layer was
built. This is the first.

It exists because **a broken generation is not an error anywhere in the stack.**
A NaN latent decodes to flat black or flat grey; ComfyUI saves it, `/history`
reports success, `probe()` returns sensible dimensions, and the bot pushes it to
the group. The same is true of a LoRA that failed to load.

Six rules, reading only the manifest and a per-job `output.json`: dimensions,
fps, duration ≈ frames/fps, audio present iff the model generates it, size floor,
and **`OUT-FLAT`** — the luma-spread check that catches the black render.
`media.luma_stats()` does the ffmpeg measurement in the pipeline and writes it to
JSON; the gate stays ffmpeg-free and fixture-testable, per its own contract.

Three fixtures ship with it, two deliberately bad, and `selftest` asserts the
gate catches its own rules on them.

**It is wired**: `worker.verify_output()` runs it after every render. A rejected
render is failed and routed through the existing failure-delivery path, so the
requester is *told* rather than handed a black square — and a test asserts that
wiring, because a gate nobody calls is exactly as useful as no gate.

### 5. Objective errors fixed

1. **Clip cost was billed at the wrong rate.** `ComfyUIProvider` accepted
   `hourly_usd`, forwarded it to capabilities, and **never stored it**; `fetch`
   billed from the module constant. Every clip on the L40S rung ($1.004/hr)
   recorded ~26% low. `providers/flux.py` did it correctly — the two disagreed.
2. **#19 — the four Flux base weights**, with exact repo/filename pairs, an
   `HF_TOKEN` check *before* the 52 GB H3 pull begins, and the Lumina VAE
   flattened into the directory `VAELoader` actually reads.
3. **`clip_duration_quantum=None`** → the real quantum.
4. **Dead parameters**: `flux_capabilities(steps=…)` was accepted and never read.
5. **`MEASURED_LATENCY_S`** was named "measured" while its own comment graded it
   `[reported]`. Gone — the data lives in the profile, graded per canvas, with
   `None` where nobody has measured rather than a plausible-looking default.

---

## Part 3 — Recorded, deliberately not changed

Every one needs the paid run or a decision. Cheap to settle: most are two images
at one seed.

| item | current | alternative | why not now |
|---|---|---|---|
| Canvas | 864×480 | 1344×768 (native) | Generation parameter. 4090 VRAM at native is unknown, and one blog even measured 864×480 peaking *higher* |
| H3 steps | 6 | 12 (our own docs' "sweet spot"), 20 (`[reported]` audio baseline) | Unreconciled today; picking blind is a coin flip. **If the clip has dialogue, 4-6 step turbo is `[reported]` to degrade audio specifically** |
| Flux guidance | 3.5 | 7.0 (LoRA card) | BFL's reference is 3.5; 7.0 is an undocumented author claim |
| Scheduler | `simple` | `beta` | Contradictory guidance across sources |
| Turbo LoRA | larryvrh + custom node | lightx2v `..._comfyui_bf16` | Documented by ComfyUI itself and needs **no custom node** — would remove a version-sniffing third-party dependency. Different shifts (6.0/3.0 vs 12.0/3.0), so it is a real change |
| T5 encoder | `t5xxl_fp8_e4m3fn` | `..._scaled` (+264 MB) | Lower quantisation error; ComfyUI's example page now lists it |
| Prompt encoding | one string to both encoders | `CLIPTextEncodeFlux`, short subject for CLIP-L | Only CLIP-L's *pooled* vector survives for Flux, so everything past ~75 tokens contributes nothing to it. Our LLM already emits structured English, so this is nearly free — but it changes output |
| `weight_dtype` | `fp8_e4m3fn` | `default` (bf16) on the L40S rung | 48 GB has the headroom, and it removes the fp8-plus-LoRA risk class entirely |
| ComfyUI version | unpinned | pin | **Should be pinned** — v0.32.0's 4× regression is real. Left for the deploy owner to choose a version |
| Host RAM ≥ 60 GB | enforced | evidence says it may be harmful | Changing a money guard blind is worse than a stale one |

---

## Part 4 — What the first live run must still answer

Unchanged from `PLAN.md` Phase 7.1, but now each has something behind it:

1. **`grep -i "lora key not loaded"`** — must print nothing. ComfyUI never
   raises on a key mismatch; it warns and returns a perfectly good un-LoRA'd
   image. This one grep decides it. **Do not look at a picture and guess.**
2. **Same seed, `lora_strength` 1.0 vs 0.0.** Two identical images mean the LoRA
   is wired to nothing — which is silent. A structural test already asserts no
   node bypasses the LoRA node; this confirms it live.
3. **guidance 3.5 vs 7.0**, then write the winner back with a date.

And the numbers to promote from `[reported]` to 📏: cold-open time (which
calibrates `close_if_idle`), per-clip generation time, whether the second clip
in a window is faster, peak VRAM, and — new — whether `torch.version.cuda`
inside the pod is really 13.0, because a CUDA-13 *image* does not imply a cu130
*wheel* and only the wheel activates the fast path.

---

## Verification

```bash
export PATH="/c/ffmpeg/ffmpeg-master-latest-win64-gpl/bin:$PATH"

uv run pytest tests -q                              # 442 passed, 1 skipped
uv run pytest tests/unit/test_model_profile.py -q   # 38: grid, canvases, conformance
uv run pytest tests/unit/test_output_gate.py -q     # 11: rules, fixtures, wiring
uv run pytest tests/unit/test_deploy_scripts.py -q  # 35: every weight is downloaded
uv run ruff check --no-cache src tests examples
uv run lint-imports                                 # 6 kept — core stays import-free
uv run mypy

# the gate catches its own rule, which is the shell's own contract
uv run python -c "
from ai_studio.gates.core import selftest
from ai_studio.gates.output_gate import output_gate
selftest(output_gate, 'tests/fixtures/output_gate/black', expect_fail='OUT-FLAT')
print('caught the black render')"
```
