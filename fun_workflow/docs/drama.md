# `/短劇` — a one-minute, six-shot drama with a stable lead

`/短劇 <一句故事前提>` turns a one-line premise into a ~56 s story with one
recurring character: gpt-oss-20b writes the screenplay into a fixed six-beat
template (hook / setup / conflict / turn / payoff / cliffhanger), Flux.1-dev
paints a character sheet and six keyframes, MiniMax H3 animates each keyframe
for 7–12 s -- cutting to a second framing inside the longer clips itself --
and ffmpeg levels the clips and splices them with the timeline's numbers:
hard cuts with a short audio crossfade, a dissolve where the screenplay says
time passed, a fade in and out, the title over the first 1.5 s and every
spoken line burned in as a Mandarin caption. One LINE video message comes
back.

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
/短劇 <premise>                LINE webhook: JobKind.DRAMA, group cap per day
      │
      ▼  worker.prepare()  —  gpt-oss-20b resident, json_only, 3 calls
   Screenplay  { title, logline, style, anchor{name, appearance, wardrobe, voice},
                 world{location, light, signature_prop},
                 6 × shot{beat, frames, scene, cut_reason,
                          1-2 × sub_shot{framing, action, camera, line}} }
      │        stored in jobs.prompt_json; ScreenplayError ⇒ job FAILED + LINE reply
      ▼  render_drama()   —  runs/drama/<token>/state.json after every artifact
   0  plan       plan.json (segments, cut reasons, cues, pacing band) →
                 gates/plan_gate.json: pacing / transitions / captions checked,
                 a FAIL is terminal before any submit → render.timeline →
                 offsets.json: the only place absolute time is computed   (CPU, free)
   1  character  Flux T2I ×2  front + three-quarter, 864×480     ┐ make_room_for(IMAGE)
   2  keyframes  Flux i2i ×6  from character/front.png, denoise 0.55  │ once
                              (0.70 when the shot opens wide), anchor + │
                              world verbatim, +FaceDetailer if present ┘
   3  clips      H3 I2V ×6    first_frame = keyframe_i, 175-277 frames ┐ make_room_for(VIDEO)
                              prompt = h3_prompt(shot): 1-2 PromptShots ┘ once
   4  level      ffmpeg loudnorm, two-pass, linear=true, per clip     (CPU)
   5  assemble   captions.ass from plan.json cues + offsets.json (title card,
                 one event per line, windowed to its segment), then one
                 ffmpeg filter_complex: concat / xfade, acrossfade at every
                 clip boundary, fade in/out, ass burn-in
                 (libx264 crf 18, aac, +faststart)                    (CPU)
      │
      ▼  files/<token>.mp4 → poster → LINE video message, caption 🎭《title》logline
```

**The rhythm is a template.** `core/drama_spec.BEAT_TEMPLATE` fixes six
beats of unequal length on H3's 17k+5 frame grid, and which of them carry a
second sub-shot; the screenwriter fills the slots and cannot change them.
Six equal ten-second shots -- the first dramas -- read as a slideshow:
`[speculative]` numbers, all of them, until a real run is watched.

| shot | beat | frames | seconds | sub-shots | H3 cuts at |
|---|---|---|---|---|---|
| 1 | hook | 158 | 6.6 | 2 | 2.5 s (the first cut is the hook) |
| 2 | setup | 243 | 10.1 | 2 | 5.5 s |
| 3 | conflict | 192 | 8.0 | 1 | — |
| 4 | turn | 243 | 10.1 | 2 | 5.0 s (the one push-in lives here) |
| 5 | payoff | 243 | 10.1 | 2 | 4.5 s |
| 6 | cliffhanger | 209 | 8.7 | 1 | — |

1288 frames = 53.7 s before dissolves; ten segments; shot-length CV 0.149
against the upstream kit's 0.11 metronome floor. Framings must alternate
across all ten sub-shots (a closed vocabulary: wide, medium, medium
close-up, close-up, over-the-shoulder, two-shot), the anchor and the world
bible are pasted verbatim into every keyframe and H3 prompt, and `push_in`
is allowed once, on the turn. All of that is `Screenplay`'s validator, so a
screenplay that breaks it fails at conversion, not on the pod.

**The gate runs before the money.** `ai_studio.gates.plan_gate` reads
`plan.json` and nothing else: the pacing band (`DRAMA_PACING`, or the
looser `DRAMA_PACING_HELD` with sub-shots off), the transition caps and
hard-cut ratio, every caption's read speed and line length, and that every
cue points at a real segment. Warnings are recorded; a failure ends the
job with the rule id in the LINE reply. The fixture drama warns twice --
the held conflict shot, and two dissolves over nine splices -- which is
the gate reading the grammar correctly.

**Cuts mean something.** `cut_reason` on a shot is what the cut *into* it
means; `ai_studio.editing.transitions` maps it to an effect and downgrades
anything it has no evidence for -- with generated clips, only
`time_passing → dissolve (0.5 s)` survives; everything else is a hard cut
with a 0.125 s audio crossfade so the two independently generated
soundtracks do not jump at the splice. `AI_STUDIO_DRAMA_SUBSHOTS=false`
collapses every shot to one held framing: the hedge for a model that
ignores `cut_at_s` under image-to-video, which nobody has measured yet.

**Stage order is checkpoint order.** All eight Flux stills run before the first
H3 clip, so ComfyUI swaps its resident checkpoint twice per drama, not twelve
times (📏 Flux reload ~15 s, H3 60–90 s).

**Keyframes are rendered at H3's canvas (864×480), not Flux's 1024².** The i2i
graph scales and centre-crops the source to the bound size, so the keyframe
*is* the first frame — no second crop between the still and the clip.

## Resume: `runs/drama/<token>/`

```
state.json            DramaState: every artifact with path + sha256 + cost, face_repair, spent_usd
plan.json             segments, clip boundaries with their cut reasons, caption cues, the pacing band -- no times
gates/plan_gate.json  the PRE gate's report; its one-line verdict is state.json's plan_gate and the status page's 剪接檢查
offsets.json          the timeline: every segment's start/end, every boundary, clip offsets, total
captions.ass          the title card and every spoken line, times copied from offsets.json
character/{front,three_quarter}.png
keyframes/shot_{1..6}.png
clips/shot_{1..6}.mp4
leveled/shot_{1..6}.mp4
render_manifest.json  every ffmpeg argv, literally
```

`plan.json`, `gates/plan_gate.json` and `offsets.json` are rewritten on every call -- they are
derived from the screenplay and cost nothing. An artifact counts only if
the file exists **and still hashes right**. A lease
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

The reaper: a drama's last GPU job is an H3 clip, so the drama grace
(`pipeline/idle.py`) equals the video grace (10). What makes that safe for a half-hour render is
that `render_drama` calls `touch_activity("drama")` after **every** fetched
still or clip, so the grace only ever measures a real gap.

## Settings

| variable | default | meaning |
|---|---|---|
| `AI_STUDIO_MAX_DRAMAS_PER_DAY` | 3 | group-wide daily cap; 0 off |
| `AI_STUDIO_DRAMA_FACE_REPAIR` | true | FaceDetailer on keyframe stills when the pod has it |
| `AI_STUDIO_DRAMA_KEYFRAME_DENOISE` | 0.55 | i2i denoise for keyframes: lower keeps the face, higher frees the scene `[speculative]` |
| `AI_STUDIO_DRAMA_KEYFRAME_DENOISE_WIDE` | 0.70 | the same for a shot that opens wide or as a two-shot: the sheet is a portrait and 0.55 keeps its framing `[speculative]` |
| `AI_STUDIO_DRAMA_SUBSHOTS` | true | ask H3 to cut to the second framing inside a clip; off = one held framing per shot |
| `AI_STUDIO_DRAMA_FONT_NAME` | Noto Sans CJK TC | the caption font, resolved by fontconfig; `funapp preflight` check 7 confirms it lands on a CJK face |
| `AI_STUDIO_DRAMA_FONTS_DIR` | — | a directory of font files for libass on a host without fontconfig |
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

## The first real run (📏 2026-08-29, RTX 4090 SECURE, EUR-IS-1)

Job 105, 「最後一個飯糰」. What it measured, and what it changed:

- **H3 honours `cut_at_s` under image-to-video.** Shot 1 (wide → medium at
  2.5 s): frames at 2.4 s and 2.7 s are two different framings of the same
  woman. The multi-shot prompt works from a keyframe. `AI_STUDIO_DRAMA_SUBSHOTS`
  stays on.
- **A clip inside a drama renders in 📏 219–244 s** (209–243 frames), not the
  79 s the single-job benchmark measured: Flux stays staged in VRAM and the
  whole H3 stack reloads. `STAGE_RESERVE_S["video"]` = 360 s still covers it.
- **277 frames OOMs beside the staged Flux weights** (7.6 GiB allocated,
  1.9 GiB requested, 4 MiB free) after three shorter clips succeeded; after
  that first OOM ComfyUI's own unload left 16 MiB free and both requeued
  attempts died in a second. Hence: every template slot ≤ 243 frames
  (`MAX_SHOT_FRAMES`), `image.evict()` before the first clip, and
  `clip.evict()` on an OOM before the attempt is handed back.
- **Wide keyframes came out on the character sheet's grey wall**, no store in
  sight: image-to-image at 0.70 keeps the source's backdrop. The sheet is
  now shot inside the world bible's location.
- **The OOM's real owner is the inference server.** On the second pod
  (job 106) `nvidia-smi` showed it holding 12.7 GiB *after* `POST /unload`
  had answered `ok` — gpt-oss-20b's exact footprint — so H3 had ~10 GiB and
  clip 2 died on "VRAM grow failed". `_release_vram`'s gc + `empty_cache`
  does not reach it; job 86 on 08-27 (moondream3 OOM after chat) was the
  same leak. `/unload` now measures what is still allocated and re-execs
  the server process when it is over 1 GiB — the one release that always
  works. `[speculative]` until the next pod's `inference.log` shows the
  restart and a clean `nvidia-smi` after it.
- **FaceDetailer MISS, twice.** First `No module named 'skimage'` (pip's
  exit code lost in a grep pipe), then `No module named 'piexif'`: the
  `git+…sam2` line in Impact-Pack's requirements wants a torch this venv
  does not have and pip refuses the whole file. The script now installs the
  requirements minus `git+` lines and proves the imports. Unverified until
  the next pod.
- **The cliffhanger came back as a prop shot** ("the phone buzzes") both
  times despite the prompt rule; `SubShot` now refuses an action without
  "the lead" in it -- but on the third run (job 107) the retry sent the
  *same* prompt and the model wrote the *same* violation back, verbatim,
  and the whole drama failed with nothing spent. `_ask`'s retry now states
  the exact `ScreenplayError` it is fixing rather than repeating the
  original prompt; unverified until the model actually corrects on retry.
- The three screenwriter calls took 📏 107 s / 45 s / 49 s on gpt-oss-20b
  (the first includes the 72 s model load); no retry needed. Two clips would
  have been the wrong beat: shot 6 was written as the phone buzzing, not the
  lead — the shots prompt should say the lead is in every shot.
- LINE's monthly push quota was already exhausted, so the group heard
  nothing either way;「讓我看看」is the pull path.

## What to measure on the next real run

Record these in this file, then say 「可以測試了」:

1. The three screenwriter calls: wall time, whether any needed a retry, and the
   token count of the shots replies against the 1536 ceiling.
2. Six keyframes side by side: is it the same person? That is the whole test of
   the denoise default.
3. `face_repair` in `state.json`: applied / skipped / failed.
4. Per-clip H3 seconds from `state.json` timestamps, total `spent_usd`, and the
   ledger's figure for the session.
5. The level jump at each cut with and without stage 4 — grammar §5.5 is ours
   and unmeasured — and whether the 0.125 s audio crossfade hides what is left.
6. **Does H3 honour `cut_at_s` under image-to-video?** Scrub shot 1 at 2.5 s
   and shot 4 at 6.0 s. If the framing does not change there, set
   `AI_STUDIO_DRAMA_SUBSHOTS=false` and note it here; the guide's multi-shot
   examples are text-to-video.
7. The wide keyframes (shots 2, 4, 6 in the dry-run fixture): did 0.70 leave
   the portrait behind while keeping the face?
8. The two shots replies' length in tokens: two sub-shots each is the
   longest reply the screenwriter has been asked for.
9. Captions against speech: a line is shown for its whole segment because
   nobody knows *when* inside the clip H3 places the words. Note how far
   off the spoken line is from the caption's window; if it is consistently
   late, the window can start later.
