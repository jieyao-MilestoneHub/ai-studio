# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code under `twin/`. It is read together with the repository root `CLAUDE.md`,
not instead of it — but nothing in the root file's ai-studio-specific
sections (RunPod, ComfyUI, the editing grammar, MiniMax/Flux licensing)
applies here. This is a separate subsystem in the same monorepo.

## What this is

A framework for building a personal "digital twin" agent: feed in one
person's (the **principal's**) historical data and self-report interviews,
get back an agent whose judgment, tool-use style, and proactivity — including
the tendency *not* to act — resembles that person in situations they've never
actually faced.

**Status (2026-08-30): the first real LoRA run is done —
`run_e6a366ee73958e69` (Qwen3-8B, r=64, 1000 steps), final adapter on R2,
verified; Phase 3-B's voice interviewer is replaced by a minimal text
interviewer (`ingest/interviewer.py`, `examples/run_text_interview.py`);
self-report fragments are always `split=train` (decided 2026-08-30,
`ingest.split.decide_self_report_split`); S1 candidate answers are
generated on Modal (`modal run launch/modal_app.py::s1_candidates`) and
paired with R2 later by `examples/prepare_s1_eval_round.py`. Phases 0, 1, 2,
3-A, 4 and 5 are code-complete and
tested (guardrails/accounts, L1 ingest incl. the real-export driver
`examples/ingest_line_export.py`, S1 item bank + wave collection, interview
post-processing/ingest, LoRA training slice, baseline + judge harness).
Real data has landed: a 96,750-fragment store, a frozen 70-item S1 bank,
and Wave 1 answers — **project day 0 is 2026-08-29T03:58Z; Wave 2 opens
2026-09-12** (PLAN.md Phase 2/6). Next: conduct the text interview, ingest
it, write `~/twin-data/data/persona.txt`, generate B1/B2 candidates; then
Wave 2 → judge alignment → Phase 7 kill switch. Phases 6+ are spec-only. All real data
lives under `~/twin-data/` (outside every checkout — see `twin/.env`), never
at the checkout-relative defaults.** `twin/` has its own package skeleton — `twin/pyproject.toml`,
`src/twin/`, its own `uv.lock` and import-linter contracts, entirely
independent of the root package's. See `twin/PLAN.md` for the authoritative
phase-by-phase status — it is updated far more often than this file's prose
is, so treat PLAN.md as current and this paragraph as a summary that can lag.
The subsystem's normative behavior is still fully described by three
documents in `twin/reference/`:

| File | Role |
|---|---|
| `SPEC.md` | The system spec. Only rules meeting its own §0 bar (a real tradeoff, a falsifiable reason, an observable failure symptom) are binding — MUST/MUST NOT/SHOULD/MAY. All items that didn't meet that bar were decided on 2026-08-27; §11 keeps the original questions for history. Implementation MUST NOT depend on anything short of a decided rule. |
| `EVAL.md` | The four acceptance suites (S1 persona fidelity, S2 tool use, S3 proactivity, S4 blind test) and their pass thresholds. `SPEC.md` §9 defers all pass/fail criteria here on purpose — the spec never defines a passing score itself. |
| `INTERVIEW.md` | The onboarding interview protocol that produces self-report data (`SPEC.md` §2.2) — currently the dominant source of persona fidelity, per D19. |
| `reference/2411.10109 實作筆記.md` | Working notes on Park et al., arXiv:2411.10109, the paper `SPEC.md`/`EVAL.md` draw their normalization method, baseline design, and interview structure from. |

`twin/docs/` exists but is currently empty; `twin/README.md` is currently
empty.

**Before writing any code here, trace the task to a `SPEC.md` §-number and an
`EVAL.md` observation point** — use the `spec-trace` skill rather than
working from memory of this file or prior conversation summary. `SPEC.md` §2
definitions carry "not" clauses that are as binding as the definition itself
and are easy to lose in a paraphrase; §10 is the only table that records
failure symptoms per decision, and a task that reproduces one of those
symptoms has already been decided against.

## Architecture (as specified, not yet built)

Four layers, hard boundaries (`SPEC.md` §3):

```
L4  Agent Runtime + MCP tools + Surface (LINE)
L3  Base model + Twin LoRA adapter
L2  Memory store (fragment / episode / period)
L1  Raw data → fragments → trajectories
```

- **C1** — memory *content* lives in L2; *recall style* is learned in L3.
  Violating this looks like the twin fabricating specific names, dates, or
  places it was never actually told.
- **C2** — tool schemas/lists are injected by L4 at inference time and MUST
  NOT enter L3 weights. Otherwise every new MCP tool needs a retrain, which
  defeats the point of pluggable tools (G4).
- **C3** — L3 learns *policy* (selection criteria, give-up thresholds, tone,
  action/no-action tendency), not knowledge and not specific tool names.
- **C4** — `recall()` is an ordinary tool behind the same interface as any
  other plugin, not a bespoke memory subsystem.

Runtime execution is a **tick loop**, not request-response (`SPEC.md` §6.2,
D28): each tick, the twin chooses zero or more tool calls, and replying is
itself a tool call (`reply`) — calling zero tools *is* `no_action`, with no
special branch. Outbound `reply` is intercepted at the runtime layer by a
three-level send gate (L0 draft-only → L1 whitelist → L2 fully automatic,
`SPEC.md` §6.5) that downgrades automatically on any failed eval round and
upgrades only manually with two consecutive passing rounds — never edit the
gate level based on a single good run.

## The non-negotiables

These are the twin-side equivalent of the root file's "Money" section: quiet
failures that don't surface until much later, if ever.

- **`split` is written once, at ingest, and is read-only after that.**
  (`SPEC.md` §4.8, D21) Deciding it at train time instead is undetectable
  time leakage — every metric still looks normal. The `data-contract` skill
  and `data-hygiene` agent both exist specifically to catch this.
- **A `no_action` sample is only `negative_class: hard` if it carries real
  exposure evidence.** (`SPEC.md` §4.3, §4.11, D20) Without an exposure
  record you cannot distinguish "chose not to act" from "never saw it."
  Mislabeling this is the project's explicitly named "failure mode #1" — a
  twin that posts constantly — and it's invisible in the S3 score itself.
- **Memory conflicts are kept, never resolved.** (`SPEC.md` §4.7, D22) Two
  inconsistent fragments about the same event both stay, cross-tagged via
  `conflicts_with`. Auto-resolving to "the correct version" produces
  something more accurate than the principal, which is a more detectable
  failure in the S4 blind test, not less.
- **Judge is Claude Code, scripted, never conversational.** (`EVAL.md` §6) One
  fresh subagent per shard, rubric read from a file rather than paraphrased
  into a prompt, no judge ever sees another shard's samples or a prior
  round's score, and totals are computed by a script, never spoken by the
  judge. The `eval-harness` skill and `eval-judge` agent enforce this
  isolation end to end — read `eval-harness` before running or modifying any
  eval plumbing.
- **No cross-suite score, ever.** (`EVAL.md` §1.4) S1–S4 measure abilities
  that trade off against each other by design (more proactive directly costs
  "knows when to stay quiet"); a weighted total makes attribution impossible
  and is explicitly listed as anti-pattern #9 in `EVAL.md` §12.
- **The Teacher's GCP project must never have billing enabled.** (`SPEC.md`
  §5.2, D8) This is the twin-side equivalent of the root file's "`pod down`
  terminates" warning — enabling billing on that project doesn't degrade
  service, it silently ends the free tier and every subsequent call is
  billed from the first token.
- **Third-party guardrail (`SPEC.md` §8 guardrail 2) is in place.**
  `twin/.gitignore`, a root-level `.pre-commit-config.yaml` (`language:
  fail`, scoped to `^twin/(data|adapters|transcripts|eval)/`), and a
  defense-in-depth copy in the root `.gitignore` all hard-block those four
  directories from version control — verified end to end with a real
  blocked commit attempt, not just file presence. `pre-commit install` still
  needs to have been run in any given clone for the hook to actually fire
  (it writes to `.git/hooks/`, which is never itself version-controlled);
  confirm that before starting real ingest work in a fresh checkout.

## Skills and agents (twin-specific)

| use | when |
|---|---|
| `spec-trace` skill | before writing any code, or when scope creeps mid-task ("while I'm here...") — traces the task to a `SPEC.md` §-number and an `EVAL.md` observation point first |
| `data-contract` skill | writing or changing ingest, fragment/trajectory schema, retrieval, or the send-gate reflow path |
| `eval-harness` skill | running or scripting any of the S1–S4 suites, or touching rubric/shard/judge plumbing |
| `spec-auditor` agent | before declaring any task under `twin/` done — checks a diff against `SPEC.md` MUST/MUST NOT and decision log, and `EVAL.md` §12 anti-patterns |
| `data-hygiene` agent | after touching L1/L2, and after every data re-ingest — checks time leakage, split contamination, negative quality, exposure capture |
| `eval-judge` agent | invoked by the `eval-harness` skill, not directly — one fresh instance per shard, never reused across shards |

## Infrastructure

**Training code landed 2026-08-27 (PLAN.md Phase 4 — `train/{formatting,model,checkpoint,reproducibility,run}.py`, root `train.py`, `launch/*`); everything below is now backed by real, tested code, not just spec.** The one item still genuinely spec-only is the LINE surface adapter (Phase 11).

- Training: Modal Starter ($30/mo credits, idle doesn't bill) for the primary
  loop, Kaggle (T4×2, ~30h/week) for long runs, Lightning AI as backup.
  `launch/modal.sh`+`modal_app.py`, `launch/kaggle.sh`+`kaggle_kernel.py`,
  `launch/lightning.sh` exist; only `modal_app.py`'s shape has been checked
  against live Modal docs — Kaggle/Lightning need verification before a real
  run (see twin/PLAN.md's Phase 4 "已知偏離" notes).
- Cross-cloud hub: Cloudflare R2 (zero egress, 10GB free tier) — fragments,
  trajectories, and adapters only; raw media never leaves local storage
  (`SPEC.md` §7.2, §4.2). Account live and bucket `twin-checkpoints` created
  2026-08-28; `train/checkpoint.py`'s bare `fsspec.core.url_to_fs(uri)` reaches
  it correctly with zero code changes, purely via `AWS_ACCESS_KEY_ID`/
  `AWS_SECRET_ACCESS_KEY`/`AWS_ENDPOINT_URL_S3`/`AWS_DEFAULT_REGION=auto` in
  `twin/.env` — verified against the real bucket, not guessed. Those vars only
  reach the real process env because `train.py` now calls `load_dotenv()`
  (`Settings` alone never writes to `os.environ`). Adapter weights (both intermediate
  checkpoints and the final adapter) are encrypted before upload (`SPEC.md`
  §8: "Adapter 為個資...MUST 加密儲存") — see `core/encryption.py` and
  `Settings.adapter_encryption_key` (`TWIN_ADAPTER_ENCRYPTION_KEY`, no
  default; generate one with `examples/generate_adapter_encryption_key.py`).
- Base model: **Qwen/Qwen3-8B** (dense, Apache-2.0) — decided 2026-08-27 after
  comparing current 8B-class, Apache-2.0-or-equivalent, Traditional-Chinese-
  capable candidates (see `twin/PLAN.md`'s Phase 4 section for the comparison
  and rejected alternatives). Shared across every twin; individual
  differences live only in the LoRA adapter (`twin/src/twin/train/model.py`).
- Provider coupling is deliberately confined to `launch/*` and `teacher.py`
  (`SPEC.md` §7.1, D12) — `train.py` itself never imports a cloud SDK
  (enforced by an import-linter contract). `launch/modal_app.py` and
  `launch/kaggle_kernel.py` are the actual vendor-coupled files (Modal/Kaggle
  both require a real code artifact, not pure CLI flags) — they live outside
  `src/twin/` so import-linter's `root_package="twin"` scan never reaches
  them, but this is a documented deviation from `PLAN.md` §3.1's literal
  "launch/ 只放 shell" tree comment, not a silent one.
- Training framework: TRL (`trl>=1.12`) + PEFT (`>=0.20`) + Accelerate
  (`>=1.14`) + bitsandbytes (`>=0.50`), current-as-of-2026-08 pins. `SPEC.md`
  §7.6 explicitly forbids a hand-rolled trainer, because checkpoint-resume
  correctness (§7.4: adapter weights, optimizer state, LR schedule, RNG
  state, global step, dataloader cursor) is easy to get subtly wrong, and
  wrong in a way that produces a fake convergence curve rather than an
  error — `tests/unit/test_train_checkpoint_kill_resume.py` verifies this
  with a real `SIGKILL` against a real (toy) subprocess, not a mock.
  Unsloth is deliberately NOT used for the production/checkpoint-critical
  training loop (undocumented resume guarantees, a recurring history of
  resume-related GitHub issues) — only plain TRL+PEFT+bitsandbytes, whose
  checkpoint contract is documented and tested.
