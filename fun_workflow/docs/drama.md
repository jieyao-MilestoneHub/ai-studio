# `/短劇` — a one-minute, six-shot drama with a stable lead

`/短劇 <一句故事前提>` turns a one-line premise into a ~60 s story with one
recurring character: gpt-oss-20b writes the screenplay, Flux.1-dev paints a
character sheet and six keyframes, MiniMax H3 animates each keyframe for ~10 s,
and ffmpeg levels and hard-cuts the six clips together. One LINE video message
comes back.

> **Every number on this page is `[speculative]` until the first real drama has
> run.** The per-step figures come from the single-job measurements in
> [model-h3.md](model-h3.md) and [model-flux.md](model-flux.md); nothing has
> yet measured the whole chain on our own pod. Promote a figure to 📏 only from
> a run's `state.json` and the spend ledger.

## The problem this design answers

Continuing or extending an H3 clip drifts the face: the model's attention to
the face decays over time, and a chain of "continue from the last frame"
generations compounds it. The user's own notes on this (2026-08-27) named the
four countermeasures below, and this pipeline is their implementation.

| countermeasure | where it lives |
|---|---|
| **Cut, don't extend.** Six shots of the measured-stable default length (243 frames = 10.125 s), hard-cut. No clip ever runs past the ~10 s drift threshold. | `core.drama_spec.SHOT_COUNT`, `pipeline.convert_worker.DEFAULT_FRAMES` |
| **先固圖、後生視.** Every shot is image-to-video from a Flux *still*, never from the previous shot's last frame. The still is image-to-image **from the character sheet**, so the face is re-anchored on every shot. | `pipeline.drama` stages 1–3 |
| **Prompt anchor.** One precise appearance string (`25-year-old Asian woman, oval face, small mole under right eye, dark chin-length straight hair`) is written once by the screenwriter and **pasted verbatim** into every keyframe prompt and every H3 prompt. The screenwriter is told the lead's *name*, never the appearance, so it cannot paraphrase the face. A screenplay with the anchor missing from any keyframe prompt does not validate. | `prompts.drama.keyframe_prompt`, `h3_prompt`; `core.drama_spec.Screenplay._check` |
| **FaceDetailer on stills only.** An Impact-Pack detail pass on each keyframe, when the pod has the nodes. Never on video: the whole point is to fix the face before motion, not after. | `workflows/flux_dev_i2i_face.json`, `providers.flux.supports_face_repair` |

**Deferred: H3 `ref2va` (Omni Reference, up to 9 reference images).** The
weights (21 GB) are not downloaded and the native ComfyUI node for it is
unverified. `ClipRequest.extra` is the hook; nothing is wired. Revisit once a
real drama shows whether the keyframe chain alone holds the face.

## The pipeline

```
/短劇 <premise>                LINE webhook: MediaKind.DRAMA, group cap per day
      │
      ▼  worker.prepare()  —  gpt-oss-20b resident, json_only, 3 calls
   Screenplay  { title, logline, style, anchor{name, appearance, wardrobe, voice},
                 6 × shot{scene, framing, action, camera, dialogue, cut_reason} }
      │        stored in jobs.prompt_json; ScreenplayError ⇒ job FAILED + LINE reply
      ▼  render_drama()   —  runs/drama/<token>/state.json after every artifact
   1  character  Flux T2I ×2  front + three-quarter, 864×480     ┐ make_room_for(IMAGE)
   2  keyframes  Flux i2i ×6  from character/front.png, denoise 0.55,  │ once
                              anchor verbatim, +FaceDetailer if present ┘
   3  clips      H3 I2V ×6    first_frame = keyframe_i, 243 frames  ┐ make_room_for(VIDEO)
                              prompt = h3_prompt(shot) with the anchor ┘ once
   4  level      ffmpeg loudnorm, two-pass, linear=true, per clip     (CPU)
   5  assemble   ffmpeg concat (libx264 crf 18, aac, +faststart)      (CPU)
      │
      ▼  files/<token>.mp4 → poster → LINE video message, caption 🎭《title》logline
```

**Stage order is checkpoint order.** All eight Flux stills run before the first
H3 clip, so ComfyUI swaps its resident checkpoint twice per drama, not twelve
times (📏 Flux reload ~15 s, H3 60–90 s).

**Keyframes are rendered at H3's canvas (864×480), not Flux's 1024².** The i2i
graph scales and centre-crops the source to the bound size, so the keyframe
*is* the first frame — no second crop between the still and the clip.

## Resume: `runs/drama/<token>/`

```
state.json            DramaState: every artifact with path + sha256 + cost, face_repair, spent_usd
character/{front,three_quarter}.png
keyframes/shot_{1..6}.png
clips/shot_{1..6}.mp4
leveled/shot_{1..6}.mp4
render_manifest.json  every ffmpeg argv, literally
```

An artifact counts only if the file exists **and still hashes right**. A lease
end, a requeue or a worker restart re-renders exactly what is missing or
corrupt: a drama that dies after clip 4 costs clips 5 and 6 on the next
attempt. A finished drama re-invoked renders nothing. The state file is
`runs/drama/<token>/state.json`; `funapp gc` sweeps these directories on the
same retention clock as `files/`, never one whose job is still pending.

## Money and time

Two gates run before any spend, in `pipeline.drama`:

- **Cost gate** — `(stills left × Flux cost) + (clips left × H3 cost) + already
  spent` must fit `AI_STUDIO_MAX_COST_USD` (the per-run ceiling, $5 by
  default). This is the first LINE path to use it. Refusal is terminal and told
  to the group.
- **Time gate** — before each GPU submit, the pod's lease must have
  `STAGE_RESERVE_S` left (90 s for a still, 360 s for a clip). Otherwise the
  job is *requeued* with its state intact and finishes in the next window,
  rather than starting a clip `--terminate-after` would throw away.

On top of those: `AI_STUDIO_MAX_DRAMAS_PER_DAY` (default 3, **group-wide**,
checked in the webhook before enqueue) — a drama is 15–30 GPU-minutes, so the
per-user job cap alone would let one afternoon spend the month.

`[speculative]` envelope on an RTX 4090 at $0.74/hr: three screenwriter calls
(~2 min including one gpt-oss load) + 8 Flux stills (~8 × 30 s) + 6 H3 clips
(6 × 79–215 s) + CPU concat ≈ **15–30 minutes, $0.20–0.40** per drama.

The reaper: a drama's last GPU job is an H3 clip, so `DRAMA_IDLE_MINUTES`
equals the video grace (10). What makes that safe for a half-hour render is
that `render_drama` calls `touch_activity("drama")` after **every** fetched
still or clip, so the grace only ever measures a real gap.

## Settings

| variable | default | meaning |
|---|---|---|
| `AI_STUDIO_MAX_DRAMAS_PER_DAY` | 3 | group-wide daily cap; 0 off |
| `AI_STUDIO_DRAMA_FACE_REPAIR` | true | FaceDetailer on keyframe stills when the pod has it |
| `AI_STUDIO_DRAMA_KEYFRAME_DENOISE` | 0.55 | i2i denoise for keyframes: lower keeps the face, higher frees the scene `[speculative]` |
| `AI_STUDIO_MAX_COST_USD` | 5.00 | the per-run ceiling the cost gate checks |

## Face repair on the pod

`deploy/pod_setup.sh` installs `ltdrdata/ComfyUI-Impact-Pack` +
`ComfyUI-Impact-Subpack` and downloads `Bingsu/adetailer face_yolov8m.pt`
**best-effort**: nothing in that block may `die`, because H3, Flux, the
understanding models and chat do not depend on it. `providers.flux` checks
`/object_info` once per pod for `FaceDetailer` and
`UltralyticsDetectorProvider`; without both, keyframes render plain i2i and
the state records `face_repair: skipped: pod has no FaceDetailer nodes`. If the
face graph is present but errors, the keyframe is retried once without it and
the state records `failed: …`. The status page shows this line.

Every parameter on the `FaceDetailer` node in `workflows/flux_dev_i2i_face.json`
(`guide_size 512, denoise 0.35, cfg 1.0, steps 12`) is `[speculative]`, authored
from the node's signature and not yet run.

## What to measure on the first real run

Record these in this file, then say 「可以測試了」:

1. The three screenwriter calls: wall time, whether any needed a retry, and the
   token count of the shots replies against the 1536 ceiling.
2. Six keyframes side by side: is it the same person? That is the whole test of
   the denoise default.
3. `face_repair` in `state.json`: applied / skipped / failed.
4. Per-clip H3 seconds from `state.json` timestamps, total `spent_usd`, and the
   ledger's figure for the session.
5. The level jump at each cut with and without stage 4 — grammar §5.5 is ours
   and unmeasured.
