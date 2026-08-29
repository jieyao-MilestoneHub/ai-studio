# Attribution

The editing grammar in this project is derived from
**[Hao0321/video-autopilot-kit](https://github.com/Hao0321/video-autopilot-kit)**
— MIT License, Copyright (c) 2026 Hao0321 Studio.

MIT permits commercial use, modification, and redistribution. The only
obligation is that the copyright notice and permission text travel with
derivative works, which they do in [NOTICE](../NOTICE).

---

## The boundary: craft inherited, epistemics not

Upstream is built for **proof-driven tutorials assembled from real footage** —
screen recordings, real dashboard screenshots, self-recorded narration. Its own
skip list explicitly rejects generative material for that genre, on the grounds
that a tutorial's b-roll must be a real screenshot or a real recording or nobody
should believe it.

Our source material is generative by design. So:

> **We inherit the kit's craft grammar — rhythm, transitions, captions, audio,
> gate discipline — and we do not inherit its evidence epistemics.**
>
> Upstream's `src/longform_maker/proof_stage.py`, its S-P risky-claim regex
> gate, and its "real screenshots only" rules exist to keep *claims* honest
> about *real* material. Applied to AI-generated footage they are a category
> error, not a standard we are failing to meet. They are excluded deliberately,
> not by omission. Every other gate is inherited and tightened.

The boundary is clean because upstream's evidence layer is one identifiable
file rather than a philosophy spread through the codebase.

---

## Per-module derivation map

| ours | upstream | inherited | changed or dropped |
|---|---|---|---|
| `editing/transitions.py` (built) | `src/longform_maker/transitions.py` | semantic table, frequency caps, exact durations | stinger binding dropped (no SFX library); evidence downgrade always fires for generated clips; a hard cut carries a short audio crossfade of our own |
| `editing/captions.py` | `word_captions.py`, `silent_vlog_maker/shorts_vertical.py` | ASS styles, `strip_emoji`, raise-on-unknown colour key, per-kind char limits | rebound from word timestamps to `segment_id`; no whisper alignment (our audio is model-native, not recorded narration) |
| `editing/audio.py` | `audio_chain.py`, `music_engine.py` | dB ladder, sidechain ducking, two-pass loudnorm, `acrossfade` on splices, SFX budget | **entire voice chain dropped** — highpass, de-esser, compressor, room tone. H3 generates its own audio; there is no recorded narration to condition. Added cross-clip loudness normalisation, which upstream never needed. |
| `editing/motion.py` | `fx_lib.py` | sub-pixel PIL AFFINE push-in, easing curves, the `zoompan` ban | gated on `source_kind` — generated clips already contain camera motion, so motion defaults off for them |
| `editing/rhythm.py` (built: the duration band and CV floor; the dual-mode wave is not) | `pace_gate.py`, `teardown.py` | dual-mode wave, the caption-rate-exceeds-cut-rate finding | "visual event" widened to include sub-cuts, so fixed-length model clips can participate in fast pacing |
| `editing/format_policy.py` | `delivery.py`, `constants.py` | raise-on-unknown platform, banned-filter list, lanczos-only | targets re-derived from H3's native canvases rather than from phone footage |
| `gates/core.py` | `gate_core.py` | `make_assert`, shared report shape, `selftest_runner` | pydantic `GateReport` |
| `gates/pace_gate.py`, `caption_gate.py`, `audio_gate.py` | same names | structural rules | — |
| **— not ported —** | **`proof_stage.py`**, S-P risky-claim gate | — | excluded, see above |

---

## Design patterns taken wholesale

These are worth more than any individual parameter:

1. **Semantic names, not effect names.** Scene sheets accept
   `TransitionReason.TOPIC_CHANGE`, never `TransitionKind.WIPE`. One table maps
   between them. You cannot ask for an effect without stating its meaning.

2. **Fail loudly, never silently degrade.** Unknown colour key, platform,
   caption kind, provider name, workflow binding — all raise. Upstream
   documents three shorts that shipped with "gold" rendered white because a
   lookup fell back to a default.

3. **Structural prohibition beats a lint rule.** Captions bind to a segment
   index, so mis-timing is impossible rather than detectable. Blacklisted
   animation patterns are simply never added to the template library.

4. **Stages hand off through JSON artifacts on disk**, forming a text-based
   EDL. Gates then become pure functions of those files, testable against
   fixtures with no ffmpeg and no GPU.

5. **Four fields per rule**: rule + parameters, landing module, mechanism,
   source link — plus an explicit record of what was considered and *rejected*,
   and why.

6. **Frequency caps live in the module docstring**, next to the function that
   could violate them.

---

## Number honesty

Upstream ships thresholds as `<fill in>` placeholders rather than presenting
someone else's calibration as fact, and grades every number it does quote. We
keep that discipline:

- 📏 measured by us on our own footage
- `[reported]` measured by someone else and quoted
- `[speculative]` inferred, not measured

**Almost everything in [editing-grammar.md](editing-grammar.md) is currently
`[reported]`.** The MiniMax H3 performance figures in
[model-h3.md](model-h3.md) are `[reported]` too. If you republish any of these
numbers, carry the grading with them.
