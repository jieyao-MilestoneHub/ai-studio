# Editing grammar

The rules this project edits by. Derived from
[Hao0321/video-autopilot-kit](https://github.com/Hao0321/video-autopilot-kit)
(MIT, © 2026 Hao0321 Studio) — see [attribution.md](attribution.md) for the
per-module map and for what we deliberately left behind.

> **Status: specified, not yet implemented.** The generation path is being built
> first. Each rule below names the module that will enforce it; a rule is not
> "done" until it has an implementation, a gate assertion, and a fixture that
> fails without it.

Every rule carries four fields:

| field | meaning |
|---|---|
| **Rule** | the rule and its exact parameters |
| **Lands in** | the module that implements it and the gate that asserts it |
| **Mechanism** | *why* it works — the reason, not the ritual |
| **Source** | where the number came from, and how much to trust it |

Confidence tags follow the upstream convention: 📏 measured by us, `[reported]`
measured by someone else and quoted, `[speculative]` inferred. **Everything on
this page is currently `[reported]`** — nothing has been measured on our own
footage yet.

---

## 0. Three rules that override everything

1. **Retention over spectacle.** Every high-energy effect has a frequency cap,
   and the caps are written in the module docstring next to the function that
   could violate them.
2. **Clarity over effect.** Body captions never animate.
3. **Fail loudly.** Unknown colour key, unknown platform, unknown caption kind —
   all raise. Upstream shipped three videos where "gold" silently rendered white
   because a lookup fell back to a default instead of raising.

---

## 1. Motion

### 1.1 `zoompan` is permanently banned

| | |
|---|---|
| **Rule** | Never use ffmpeg's `zoompan`, in any form. All push-in and Ken Burns motion goes through sub-pixel PIL `Image.AFFINE` transforms with explicit easing. The one-step alternative in pure ffmpeg is `crop=iw/1.12:ih/1.12,scale=W:H:flags=lanczos`. |
| **Lands in** | `editing/motion.py`; asserted by `gates/grammar_gate.py` over the literal argv recorded in `render_manifest.json` |
| **Mechanism** | `zoompan` computes its window on integer pixel coordinates, so a slow zoom snaps between whole pixels instead of sliding. The jitter is not tunable — it is inherent to the filter. An affine transform interpolates at sub-pixel precision. |
| **Source** | Upstream permanent-ban list `[reported]` |

Already enforced: `editing/format_policy.BANNED_FILTERS` and the tests in
`tests/unit/test_format_policy.py`.

### 1.2 ⚠️ Generated clips do not get motion — the conflict that is easy to miss

| | |
|---|---|
| **Rule** | `motion_semantic` must be `None` on any shot with `source_kind=GENERATED`. Only `STILL` shots may take a push-in. Overriding requires a per-shot waiver recorded in `plan.json`. |
| **Lands in** | `core/models.Shot` validator (**already enforced**); waiver recorded by `gates/plan_gate.py` |
| **Mechanism** | Ken Burns exists to make a *still* move. An H3 clip already contains camera movement — the prompt schema has a whole vocabulary for it (`pushes in`, `trucks right`, `arcs around`). Layering a synthetic push-in on top produces two independent motions in one frame, which reads as amateurish. This is the same failure the `zoompan` ban targets, arriving through a different door. |
| **Source** | Our own analysis of the upstream grammar against generative source material |

`editing/motion.py` still gets built — for title cards, reference stills, and
i2v source frames. It is just not the default path for model output.

### 1.3 Easing

| | |
|---|---|
| **Rule** | Enter with ease-out (fast in, slow stop); exit with ease-in (slow start, fast leave); sustained movement uses ease-in-out. Push-in default `z0=1.00 → z1=1.05` over 6s at 30fps with `smootherstep`. Render source art at 2× and downsample. |
| **Lands in** | `editing/easing.py`, `editing/motion.py` |
| **Mechanism** | Asymmetric enter/exit curves are, per upstream, "the invisible source of professional feel". Rendering at 2× means the affine transform samples down rather than up, so the zoom costs no sharpness. |
| **Source** | Upstream `fx_lib.py` / `premium-motion-fx.md` `[reported]` |

---

## 2. Rhythm

### 2.1 Dual-mode wave

| | |
|---|---|
| **Rule** | Every scene is tagged `fast` or `focus`. In `fast`, a visual event every 3–5s and any still frame >4s **fails**. In `focus`, a hold may run to 40s and >40s **fails**. No three consecutive scenes may share a mode. |
| **Lands in** | `editing/rhythm.py`; asserted by `gates/pace_gate.py` |
| **Mechanism** | A single global cut rate is flat regardless of which rate you pick. Forbidding three same-mode scenes in a row forces a waveform mechanically, without anyone having to feel it. The old "≥12 cuts/min" target survives only as the floor for `fast` segments. |
| **Source** | Upstream `pace_gate.py` `[reported]` |

### 2.2 A visual event is not only a cut

| | |
|---|---|
| **Rule** | A *visual event* is a cut, **a caption change**, or a sub-cut inside a single clip. |
| **Lands in** | `editing/rhythm.py`, `planner/shot_planner.py` (`Shot.subcuts`) |
| **Mechanism** | This resolves the clip-quantum conflict. A 5s H3 clip contains no cut, so a strict cuts-per-minute rule would fail every generated scene on its face. It also matches what was actually measured: across seven competitor verticals, **caption-change rate exceeded cut rate in all seven**, by up to 8×. One sample ran 32s with 5 cuts and 40 caption changes. Rhythm is carried by captions, not by the edit. |
| **Source** | Upstream `competitor-vertical-teardown-2026.md`, frame-accurate measurement of 7 shorts `[reported]` |

Measured competitor figures, for calibration:

| clip | duration | cuts/min | median cut gap | caption changes/min | median dwell |
|---|---|---|---|---|---|
| food buffet | 54.3s | 30.9 | 2.00s | 39.7 | 1.43s |
| unboxing | 31.8s | 9.4 | 5.54s | **75.4** | 0.73s |
| hotpot | 24.5s | 51.5 | 1.10s | 71.1 | 1.07s |
| AI tool demo | 33.4s | 25.1 | 2.02s | 52.0 | 0.63s |

---

## 3. Transitions

### 3.1 Semantic table — meaning first, effect second

| | |
|---|---|
| **Rule** | ≥90% of splices are hard cuts. Only a chapter boundary earns a motivated transition, and it is chosen from a table keyed by *meaning*: topic change → wipe/whip/slide; time passing → short dissolve; drilling into detail → zoom-through; default → hard cut. More than one non-hard-cut per chapter **fails**. |
| **Lands in** | `editing/transitions.py`; asserted by `gates/pace_gate.py` |
| **Mechanism** | A scene sheet may only contain a `TransitionReason`, never a `TransitionKind` — so you cannot write "put a wipe here" without first stating what the wipe means. That single indirection is what stops transitions accumulating as decoration. |
| **Source** | Upstream `transitions.py` semantic table `[reported]` |

Already modelled: `core/enums.TransitionReason` vs `TransitionKind`.

### 3.2 Frequency caps and durations

| | |
|---|---|
| **Rule** | Per video: `luma_wipe ≤ 3`, `zoom_punch ≤ 2`. Durations are exact: whip **0.30s**, wipe **0.50s**, zoom **0.33s**. Every transition carries a stinger SFX on the same frame (±1 frame). One forward direction per video; reverse is reserved for callbacks. Never stack two transition effects. |
| **Lands in** | `editing/transitions.py` module docstring (caps live next to the code that could break them); asserted by `gates/pace_gate.py` |
| **Mechanism** | A silent transition reads as cheap; the SFX is what sells it. The durations land on whole frames at both 24 and 30fps, which is not a coincidence. |
| **Source** | Upstream `transitions.py` header `[reported]` |

### 3.3 Evidence contract

| | |
|---|---|
| **Rule** | A transition may only be *named* if the shot-pair evidence exists. A whip-pan cut requires real whip motion in both shots, same direction, motion blur covering the cut. A foreground wipe requires >70% real occlusion. Without evidence it downgrades to `hard_cut`. |
| **Lands in** | `editing/transitions.py` |
| **Mechanism** | Naming a transition you cannot execute produces a labelled cut that does not look like the label. Downgrading is honest; faking is not. |
| **Source** | Upstream `mediastorm-craft-system.md` `[reported]` |

---

## 4. Captions

### 4.1 Bind to segment index, never to a timestamp

| | |
|---|---|
| **Rule** | `CaptionCue` carries a `segment_id` and no time fields. Absolute time is computed in exactly one function (`render/timeline.resolve_timeline`) and written to exactly one file (`offsets.json`). |
| **Lands in** | `core/models.CaptionCue` (**already enforced** — the model has no time fields), `render/timeline.py` |
| **Mechanism** | Upstream shipped captions 2–3 seconds out of sync because segments were split by hand and the timings drifted. Binding to an index makes mis-timing *structurally impossible* rather than merely detectable. It also collapses "a caption may never straddle a cut" into an assertion inside one function. |
| **Source** | Upstream `word_captions.py` / M105 `[reported]` |

### 4.2 White-first

| | |
|---|---|
| **Rule** | Base colour is always white. At most **2** non-white colours per video. Coloured characters ≤ **35%** of all characters. Colour words, never whole sentences — if a coloured word is ≥60% of its sentence, cancel the colouring. Same meaning keeps the same colour throughout. |
| **Lands in** | `editing/captions.py`; asserted by `gates/caption_gate.py` |
| **Mechanism** | Colour carries meaning only while it is scarce. Numbers and prices take the accent; everything else stays white. |
| **Source** | Upstream `caption-art-direction.md` `[reported]` |

### 4.3 Read speed outranks caption density

| | |
|---|---|
| **Rule** | Chinese: **>5 chars/sec warns, >7 chars/sec fails.** Target caption cadence is median dwell ≤1.8s and ≥30 changes/min — but read speed wins the conflict. Sacrifice density for legibility, never the reverse. |
| **Lands in** | `gates/caption_gate.py` |
| **Mechanism** | Upstream's phrasing: 讀不完＝白寫 — a caption nobody can finish reading was not written. Dense captions that cannot be read are worse than sparse ones that can. |
| **Source** | Upstream `shorts_gate.py` S-R/S-O `[reported]` |

### 4.4 Emoji must be stripped

| | |
|---|---|
| **Rule** | `strip_emoji()` runs on every caption before ASS emission. Real emoji require a PNG sticker overlay. |
| **Lands in** | `editing/captions.py`; asserted by `gates/caption_gate.py` |
| **Mechanism** | libass with a CJK font has no emoji glyphs, so an emoji renders as a tofu box — and it renders that way *into the finished video*, discovered after delivery. |
| **Source** | Upstream `shorts_vertical.py` `[reported]` |

### 4.5 Kinetic restraint

| | |
|---|---|
| **Rule** | Kinetic caption kinds ≤ **40%** of cues. Animate one key word, never a sentence. Enter 200–300ms ease-out, then hold still ≥1.5s. Animate position and scale only — never colour. Comprehension must never depend on animation timing. |
| **Lands in** | `editing/captions.py`; asserted by `gates/caption_gate.py` |
| **Mechanism** | A caption that must be seen animating to be understood is unreadable on a rewatch, on a scrub, and in a thumbnail. |
| **Source** | Upstream wave6 W6B-2 `[reported]` |

Already modelled: `core/enums.CaptionKind.is_kinetic`.

---

## 5. Audio

Our source material has **native audio** — H3 generates sound with the picture.
That removes the entire voice chain (highpass, de-esser, compressor) that
upstream needed for recorded narration. What remains is mixing and delivery.

### 5.1 Four-layer dB ladder

| | |
|---|---|
| **Rule** | Voice −6…−12 dB · key SFX −12…−18 · ambience −18…−24 · BGM −18…−25. Within the SFX band: impact ≈ −12, whoosh ≈ −15, UI click ≈ −18, asserting `click ≤ whoosh − 3` and `impact ≥ whoosh + 3`. |
| **Lands in** | `editing/audio.py`; asserted by `gates/audio_gate.py` |
| **Mechanism** | Voice is always loudest; every other layer gets out of its way. The intra-SFX spacing is what makes a click "felt not heard". |
| **Source** | Upstream `editing-craft-fundamentals.md` `[reported]` |

### 5.2 Real sidechain ducking, not a fixed volume

| | |
|---|---|
| **Rule** | `sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400:makeup=1`. Never a static `volume=` on the music bus. Optionally pair with a permanent 2–5 kHz, −3…−4 dB wide-Q dip on the music bus. |
| **Lands in** | `editing/audio.py`; asserted by `gates/audio_gate.py` |
| **Mechanism** | A fixed −18 dB does not duck further when someone speaks, and does not come back up in the gaps. Upstream diagnosed exactly this as their "music fights the voice" bug. The EQ pocket lets you back the ratio off so the music stays present instead of pumping. |
| **Source** | Upstream `audio_chain.py`, wave6 W6A-5 `[reported]` |

### 5.3 Two-pass loudness

| | |
|---|---|
| **Rule** | Pass 1 `loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json` to measure; pass 2 the same with `measured_*` values **and `linear=true`**. Single-pass is forbidden. |
| **Lands in** | `editing/audio.py`; asserted by `gates/grammar_gate.py` (argv) and `gates/delivery_gate.py` (measured result) |
| **Mechanism** | Single-pass `loudnorm` *is* a dynamic compressor — it pumps. `linear=true` applies one fixed gain across the whole file, preserving dynamics. |
| **Source** | Upstream `[reported]` |

### 5.4 SFX budget

| | |
|---|---|
| **Rule** | ≤5 SFX per minute; ≥3 in any sliding 10s window **warns**; **`SFX_events / cut_count > 0.5` fails**. 2–3 rotating variants per class; the same file within 20s must be pitch-shifted ±2–3 semitones. |
| **Lands in** | `editing/audio.py`; asserted by `gates/audio_gate.py` |
| **Mechanism** | The ratio rule catches the specific tell of amateur editing — a whoosh on every cut — which a per-minute cap alone misses on a fast sequence. |
| **Source** | Upstream wave6 W6A-1/2 `[reported]` |

### 5.5 Cross-clip loudness normalisation *(ours, not upstream)*

| | |
|---|---|
| **Rule** | Measure every generated clip's loudness independently and align before concatenation. |
| **Lands in** | `editing/audio.py` |
| **Mechanism** | Upstream had one continuous recorded narration bed, so this problem did not exist. We have N independently generated clips whose native audio levels have no reason to match, and a level jump at a cut is far more noticeable than a picture jump. |
| **Source** | Ours `[speculative]` — needs measuring on real H3 output |

---

## 6. Format and delivery

Implemented — see `editing/format_policy.py` and
`tests/unit/test_format_policy.py`.

| | |
|---|---|
| **Rule** | Scale first, crop second. Resampler always lanczos. `MAX_UPSCALE = 2.5`, `MIN_AREA_RETAINED = 0.85`. `minterpolate` banned. Even dimensions only. |
| **Lands in** | `editing/format_policy.py`; asserted by `gates/format_gate.py` |
| **Mechanism** | Cropping before scaling forces a non-integer, non-uniform scale factor and reintroduces sampling jitter. `minterpolate` smears generative artifacts into something worse than judder — the artifacts are not real motion, so interpolating them invents motion that was never there. |
| **Source** | Upstream ban list + our derivation 📏 (geometry verified in tests) |

| native | target | upscale | area kept |
|---|---|---|---|
| 864×480 | 1920×1080 | 2.25× | 98.8% |
| 1344×768 | 1920×1080 | 1.43× | 98.4% |
| 864×480 | 1280×720 | 1.50× | 98.8% |
| 864×480 | 1080×1920 | ❌ 4.00× | 31% — rejected, falls back to `hybrid_pad` at 1.67× / 75% |

---

## 7. What this grammar does *not* cover

Upstream's evidence layer — the risky-claim regex gate, the "real dashboard
screenshots only" rule, `proof_stage.py` — is **not ported**. Those rules keep
*claims* honest about *real footage*. Our source material is generative by
design, so they are a category error here rather than a standard we are failing.
See [attribution.md](attribution.md) for the full statement of that boundary.
