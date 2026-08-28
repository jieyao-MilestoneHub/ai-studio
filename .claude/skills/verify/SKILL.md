---
name: verify
description: Run the full local verification sweep before committing or handing work back — lint, layering contracts, tests, and the offline end-to-end generation. Use after any code change, when asked "does this still work", or before a commit.
---

# Verify

Run all of it. Each check catches something the others do not.

```bash
export PATH="/c/ffmpeg/ffmpeg-master-latest-win64-gpl/bin:$PATH"   # ffmpeg is installed but not on PATH

# ai-studio (repo root)
uv run ruff check --no-cache src tests examples
uv run lint-imports
uv run mypy
uv run pytest tests -q
uv run ai-studio doctor
uv run ai-studio generate "a test clip" --provider stub

# fun_workflow (its own venv; ai-studio installed editable from ..)
cd fun_workflow
uv run ruff check --no-cache src tests
uv run lint-imports
uv run mypy
uv run pytest tests -q
AI_STUDIO_RUNS_DIR=/tmp/verify-runs uv run funapp worker --max-ticks 1
cd ..
```

Run **both** halves whenever ai-studio changed: fun_workflow imports it, so
a green root sweep says nothing about the callers on the other side.

## What each one is actually for

| check | catches |
|---|---|
| `ruff check` | style and a real class of bugs (unused imports, mutable defaults) |
| **`lint-imports`** | **architecture drift** — five contracts at the root, the spine in fun_workflow. This is the one people forget, and it is the one that stops the design eroding. |
| `mypy` | strict on both `src/` trees; ai-studio ships `py.typed` so fun_workflow's calls into it are checked too |
| `pytest` | root: format geometry, the H3 prompt schema, the turbo-trap guard, the ledgers, the benchmark fold; fun: the webhook, queue, worker, drain, drama machine, and `test_layering.py` (only `cli` may import `ai_studio.runtime`) |
| `ai-studio doctor` | ffmpeg and its required filters, python version, credentials |
| `generate --provider stub` | the generation pipeline end to end, offline, free |
| `funapp worker --max-ticks 1` | the composition root imports and one idle tick, with no pod and an empty queue |

## Gotchas

- **`ruff format` cannot write in this sandbox**; `ruff check --fix --no-cache`
  can, and is how the `ai_studio` / `fun_workflow` import blocks get sorted.
- **`--no-cache` is required** — ruff cannot create `.ruff_cache` here.
- **ffmpeg is not on PATH**; export it first or `doctor` and the stub provider
  both fail.
- The Bash working directory persists between calls. `cd` back to the repo root
  or use absolute paths.
- **Always pass `encoding="utf-8"`** to `open()` and `subprocess`. The Windows
  default is cp950 and throws on any non-ASCII byte. Keep CLI output ASCII —
  em-dashes render as mojibake in the console.

## Expected state

At the time of writing (2026-08-28, after the split): root ruff/mypy clean,
**5/5 contracts kept**, **353 tests passing**; fun_workflow ruff/mypy clean,
**1/1 contract kept**, **339 tests passing**; doctor green, stub producing
864×480 @ 24fps with audio.

If `lint-imports` reports a broken contract, do not add an exemption. The
contract is the architecture — the import is the thing that is wrong.
