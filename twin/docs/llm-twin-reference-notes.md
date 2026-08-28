# llm-twin reference notes

`twin/PLAN.md` stays the terse, authoritative phase tracker. This file is
the detailed version — for anything below, PLAN.md names the phase and this
file names the reasoning, so PLAN.md doesn't have to carry it.

Source: `twin/reference/llm-twin` — a separate, more mature "digital twin"
pipeline (ten stages S0–S10, `twin-core`/`twin-ingest`/`twin-privacy`/
`twin-curate`/`twin-synth`/`twin-train`/`twin-eval`/`twin-ship`/`twin-serve`,
a `uv` workspace), evaluated for reuse against `twin/reference/SPEC.md` and
`EVAL.md` and the current `twin/src` code.

Two things constrained this pass before any compatibility question came up:
a peer effort was already active on Phase 3 (interview ingest) and Phase 5
(baseline + judge harness) in a sibling checkout — that territory was left
alone entirely — and most of llm-twin's own S6–S10 stages turned out to
either conflict with an already-decided SPEC MUST, or belong to a phase
gated behind the Phase 7 kill-switch decision. What's below is what
survived that filter.

## Ported

### Assistant-only loss masking (`train/loss_mask.py`, hardens Phase 4)

**PLAN.md**: Phase 4 ("最小 L2 + 精簡軌跡集 + 第一版（精簡）LoRA"), new bullet
"補強：assistant-only loss masking".

**The gap**: `train/run.py` builds `SFTTrainer` from a plain `SFTConfig` —
no `assistant_only_loss`, no `completion_only_loss`, no custom collator, no
verification. Every other choice in that file is deliberately explicit and
recorded (not a SPEC.md rule specifically about this — SPEC §11 item G is
about the base-model-capacity decision, not chat-template masking — but the
same discipline this codebase applies elsewhere: see
`LORA_RANK_FALLBACK_LADDER`'s docstring, or `TrainingConfig`'s "no default on
purpose" fields). Silently trusting TRL's default masking behavior for a
`messages`-format dataset is the one config choice in this file that wasn't
explicit. llm-twin's `packages/twin-train/src/twin_train/common.py` names
this exact class of bug directly: *"a wrong marker causes the entire
sequence to count toward loss with no error — the hardest silent bug to
catch"* — training doesn't fail, loss still drops, but the model learns to
imitate the other party's turns and the injected tool-name vocabulary
instead of the principal's own replies.

**Adapted, not copied.** llm-twin's `verify_masking()` is built on hardcoded
per-architecture marker strings (`RESPONSE_MARKERS = {"qwen": ("<|im_start|>user\n", "<|im_start|>assistant\n"), ...}`)
— written against an older TRL/transformers surface. Twin's pinned
`trl>=1.12`/`transformers>=5.16` expose a more robust, authoritative native
mechanism instead: `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`
plus `SFTConfig(assistant_only_loss=True)`. `train/loss_mask.py` is built on
that mechanism, not the marker-string approach.

**Verified live, not assumed** (2026-08-28, real network fetch of the
pinned `Qwen/Qwen3-8B` tokenizer at `train.model.DEFAULT_BASE_MODEL_REVISION`,
against the actually-installed `trl==1.12.0`):

- The real tokenizer's own chat template does *not* natively have
  `{% generation %}` markers (expected — most model templates don't ship
  with them).
- TRL 1.12 resolves this by **exact string equality** against a template
  baked into that TRL release (`trl.chat_template_utils.get_training_chat_template`
  — a hardcoded `if processing_class.chat_template == qwen3_chat_template: return qwen3_training_chat_template`
  chain, one `if` per supported model family, ending in `raise ValueError`
  if nothing matches). For the exact pinned model+revision+TRL version
  combination in use today, this match holds and the swap succeeds.
- Rendering twin's actual tool-call message shape (`formatting.trajectory_to_messages`'s
  output: a `tool_calls` assistant turn followed by a `tool` turn) through
  the resolved template correctly isolates only the assistant's tool-call
  span, including the stop token (`is_chat_template_stop_token_trained`
  returned `True` — the model will learn to stop, not just what to say).

That exact-string-match mechanism is exactly the kind of thing that can
silently stop matching on a future TRL upgrade or a base-model/revision
change — TRL itself fails loudly when that happens (raises `ValueError`
inside `SFTTrainer.__init__`), but only after the quantized 8B model has
already been constructed. `train/loss_mask.py:verify_assistant_masking` runs
the same resolution as a tokenizer-only pre-flight step in `train/run.py`,
before any model/GPU cost is paid — checked over every example in the
dataset, not just one, and printed (not just returned) since these runs are
unattended on Modal/Kaggle/Lightning: a report nothing reads is not a
pre-flight check. It gives a decoded sample of exactly what's in the masked
span, the automated equivalent of llm-twin's manual "read the printed masked
tokens before a real run" dry-run step.

`tests/unit/test_train_checkpoint_kill_resume.py`'s toy tokenizer needed its
chat template updated to include real `{% generation %}` markers (it
previously used a plain, non-generation-tagged custom template) — otherwise
`assistant_only_loss=True` would make that CI-critical test fail via TRL's
own internal resolution. Re-ran the full kill/resume test after the change;
the calibrated `MAX_PER_STEP_LOSS_DIFF = 0.01` threshold still holds
unchanged.

**Investigated, turned out not to be a bug**: rendering a real trajectory
through the real Qwen3-8B template initially looked like it always inserts
an empty `<think>\n\n</think>\n\n` scaffold into assistant turns, which read
as the exact train/infer mismatch llm-twin's `sft_qwen3_4b_t4.yaml` calls
out (it sets `enable_thinking: false` explicitly and warns train/infer must
match). Verified further before treating that as a real gap: `enable_thinking`
has **no effect on a completed assistant turn** — it only changes what's
inserted at the generation-prompt point (`add_generation_prompt=True`, i.e.
the moment right before a *new* response is generated):

```
completed turn,  enable_thinking=True/False/unset -> identical:
  "...<|im_start|>assistant\n<think>\n\n</think>\n\nhi there<|im_end|>\n"
generation prompt, enable_thinking=True/unset -> "...assistant\n"
generation prompt, enable_thinking=False      -> "...assistant\n<think>\n\n</think>\n\n"
```

So the empty `<think></think>` in every training example's completed
assistant turn isn't a masking artifact or a missed config flag — it's
Qwen3's standard SFT format for a non-reasoning reply, and it's exactly
what should sit inside the assistant-masked span: training the model to
always close its think block immediately is what makes `enable_thinking=False`
at inference (an empty think block pre-filled into the generation prompt)
land on a response the model has actually been trained to continue
naturally. `train/formatting.py` needs no change for this.

Where `enable_thinking=False` *will* matter is real: whenever Phase 11's
serve/inference path is built and calls `apply_chat_template(...,
add_generation_prompt=True)` to generate a live response, it MUST pass
`enable_thinking=False` there — that's the one call site the flag actually
affects, and it doesn't exist yet. Left as a note for whoever picks up
Phase 11, not a training-side fix.

### `TrainingConfig` effective-batch-size ceiling (`run.py`, hardens Phase 4)

**PLAN.md**: same Phase 4 bullet as above.

`run.py`'s `gradient_accumulation_steps` field already carried a comment —
"Keep per_device_train_batch_size * this < ~32 (LoRA-specific ceiling)" —
that nothing enforced. Added a `model_validator` that raises if the product
exceeds `MAX_EFFECTIVE_BATCH_SIZE = 32`, inspired by llm-twin's
`twin_train/common.py:guard_config()` pattern (pre-flight sanity checks
instead of a comment nobody re-reads). Kept to exactly this one
already-documented invariant — no new thresholds invented (e.g. no
epoch-count check: twin uses `max_steps`, not `num_train_epochs`, so
llm-twin's "epoch>3 is probably overfitting" check doesn't translate here).

## Evaluated, not ported

- **ORPO/DPO alignment, preference-pair generation** (llm-twin S6,
  `twin_train/align.py` + `twin_synth/prefs.py`). SPEC §5.3/D13: *"v1 MUST NOT
  使用 RLHF/DPO"* — reasoning given is "no preference data yet; human-in-the-loop
  hasn't produced rejection samples." No v2 phase exists anywhere in
  PLAN.md's Phase 0–14 roadmap to attach this to; SPEC's own "v2 考慮偏好學習"
  is a one-sentence contingency, not a planned phase.

- **`PersonaCard` single-source-of-truth system prompt** (llm-twin S0,
  `twin_core/persona.py`, driving train/serve/judge prompts from one YAML
  card). EVAL §3.4 defines a static persona-prompt baseline (**B1**)
  specifically as the weak strawman both **B2** (raw interview-transcript
  injection) and **T** (the actual twin) are expected to beat — D19 restates
  the same finding (persona-type baselines underperform interview-type).
  Adopting it as twin's actual mechanism would also invert SPEC §3's C1
  layering (memory *content* in L2, recall *style*/policy in L3) by putting
  identity content into a static prompt instead.

- **Multi-provider `TeacherPool`** (llm-twin S4, `twin_synth/teachers.py` —
  Cerebras/Groq/Gemini/OpenRouter failover). Not architecturally blocked:
  SPEC §5.2 already makes Teacher swappable via a `Protocol`
  (`teacher/base.py`), and `teacher/gemini.py`'s own docstring says as much
  explicitly. But it's new scope, not a hardening — D9's "few calls, large
  batches" ledger discipline has no multi-provider-aware ledger today
  (`TeacherCallLedger` is single-provider-RPD-scoped), and EVAL §6.1's
  teacher/judge cross-provider requirement ("若日後更換 teacher，MUST 確保其與
  judge 非同一供應商") would need explicit enforcement against whatever pool
  is built. Deferred, not excluded — a real candidate if/when Gemini's free
  tier becomes the actual bottleneck.

- **Coarse-to-fine memory / vector store** (llm-twin S9's plain
  numpy+JSONL `MemoryStore`). The real target is Phase 9 (SPEC §4.5–§4.7:
  three-tier Fragment→Episode→Period granularity, salience decay, and
  MUST-NOT-resolve conflict preservation via `conflicts_with`) — considerably
  richer than llm-twin's flat cosine-similarity store. Phase 9 sits behind
  the Phase 7 kill-switch and Phase 8 (exposure collection) in the critical
  path. Today's `memory/retrieve.py` is an explicit placeholder documented
  as something Phase 9 *replaces*, not extends — building toward it now
  would be racing ahead of the project's own critical-path discipline.
  (If/when Phase 9 is picked up, the "cheap embedding search without a
  vector DB" pattern is worth revisiting then — it fits SPEC's zero-cost
  goal G5 — but only as one ingredient of the period-first drill-down, not
  as a replacement for the salience/conflict logic.)

- **Bias-mitigated bidirectional arena** (llm-twin S7, `twin_eval/arena.py` —
  position-bias swap+consistency check, verbosity disclosure, multi-provider
  judge rotation). No suite in `EVAL.md` §3–§8 is pairwise/A-vs-B: S1 is
  direct-answer accuracy, S2 is task completion, S3 is proactivity F1, and
  S4 (§8) is source-identification ("who wrote this, the principal or the
  twin"), not "which is better." There's no current suite to attach this
  pattern to. Also Phase 5/12 territory (peer session, and gated) regardless.

- **HF Hub shipping, GGUF quantization, model merging** (llm-twin S8,
  `twin_ship/*`). No such stage appears anywhere in `SPEC.md` or `PLAN.md`.
  The twin's lifecycle as specified stops at an encrypted LoRA adapter in
  R2, loaded directly by the agent runtime (SPEC §7.2, §8) — there is no
  distribution/export/publish concept, consistent with SPEC's framing of
  this as a personal-use tool, not a product (§8: "本專案為個人自用之開源工具，
  非產品").

- **Proactive "should_speak" self-judged loop + drift-resistant session**
  (llm-twin S9, `twin_serve/proactive.py` + `session.py`). Conceptually the
  closest analogue to Phase 11's tick loop / three-level send gate — the
  self-judged "is there something I'd genuinely regret not saying" gate,
  the end-of-context persona-anchor reinjection against lost-in-the-middle
  drift, and excluding avoid-phrase replies from history so the model
  doesn't imitate its own mistakes are all directly relevant techniques.
  Phase 11 sits after the Phase 7 kill-switch and Phases 8–10 in the
  critical path, so nothing was built now — noted here for whoever picks up
  Phase 11 once the kill-switch decision is past.
