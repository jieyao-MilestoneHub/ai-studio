---
name: editing-rule
description: Add or change a rule in the editing grammar (pacing, cuts, transitions, captions, audio, format). Use whenever the task touches src/videogen/editing/, src/videogen/gates/, or docs/editing-grammar.md — "add a transition rule", "change the caption read-speed limit", "why is this gate failing", "port another rule from the upstream kit".
---

# Landing an editing rule

A rule that exists only in someone's head is taste. A rule that exists only in
code is undocumented. A rule with no gate is a suggestion. All four steps, in
order, or it does not land.

## 1. Document it first, with all four fields

In `docs/editing-grammar.md`:

| field | content |
|---|---|
| **Rule** | the rule and its **exact parameters** — "whip 0.30s", not "a short whip" |
| **Lands in** | the module that implements it *and* the gate that asserts it |
| **Mechanism** | *why* it works. The reason, not the ritual. |
| **Source** | where the number came from |

Grade the source honestly: 📏 measured by us, `[reported]` quoted from someone
else, `[speculative]` inferred. **Most of the grammar is currently
`[reported]`** — do not silently promote a number to 📏 without measuring it on
our own footage.

Writing the mechanism first is the useful discipline: if you cannot say why a
rule works, you have copied a habit rather than ported a technique.

## 2. Implement it in `editing/`

Constraints that are enforced, not advisory:

- `editing` may not import `providers`, `render`, `runtime`, `storage`, or
  `media`. It is pure functions over the core data model. `lint-imports` fails
  the build otherwise.
- **Semantic names, not effect names.** An author writes
  `TransitionReason.TOPIC_CHANGE`; one table maps it to `TransitionKind.WIPE`.
  Never let a caller name the effect directly.
- **Fail loudly.** Unknown colour key, platform, caption kind → raise. Never
  fall back to a default. Upstream shipped three videos where "gold" silently
  rendered white.
- **Frequency caps go in the module docstring**, next to the function that
  could violate them.

## 3. Assert it in `gates/`

- Gates are **pure functions of JSON artifacts on disk**. They may not import
  `providers`, `render`, or `runtime`. `delivery_gate` reads `probe.json`; it
  does not shell out to ffprobe.
- Put the rule body in its own gate file. Upstream's phrasing:
  不集中才不會互相污染.
- Decide PRE or POST. **PRE gates run before any GPU-second is spent** and must
  be pure functions of `plan.json`. At 2–6 minutes of GPU per clip, a gate that
  runs after generation is a receipt, not a check.
- Use `GateRun.assert_that(condition, rule_id, message)` — the message describes
  the *violation*, so a reader sees what went wrong without inverting the rule.

## 4. Add a fixture that fails without it

`tests/fixtures/runs/bad_<rule_id>/` plus a `selftest` call.

**A rule with no failing fixture is not merged.** A gate that silently stops
working is otherwise indistinguishable from a gate that is being satisfied —
which is the exact failure this whole layer exists to prevent.

## Before porting anything from upstream, check the boundary

We inherit `Hao0321/video-autopilot-kit`'s **craft** — rhythm, transitions,
captions, audio, gate discipline. We do **not** inherit its **evidence
epistemics**: `proof_stage.py`, the risky-claim regex gate, "real screenshots
only". Those keep *claims* honest about *real footage*; our material is
generative by design, so they are a category error here, not a standard we are
failing. See `docs/attribution.md`.

Also check the three recorded conflicts before assuming a rule transfers:

1. **Generated clips already contain camera motion.** Ken Burns on top
   double-moves the frame. Enforced in `core/models.Shot`.
2. **A 5s model clip contains no cut.** A visual event is a cut, a caption
   change, **or a sub-cut** — rhythm is carried by captions.
3. **Cost makes gate ordering architectural**, hence PRE/POST.

## Verify

```bash
uv run pytest tests -q
uv run ruff check --no-cache src tests
uv run lint-imports
```
