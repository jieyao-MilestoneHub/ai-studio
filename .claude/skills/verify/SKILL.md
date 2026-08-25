---
name: verify
description: Run the full local verification sweep before committing or handing work back — lint, layering contracts, tests, and the offline end-to-end generation. Use after any code change, when asked "does this still work", or before a commit.
---

# Verify

Run all of it. Each check catches something the others do not.

```bash
export PATH="/c/ffmpeg/ffmpeg-master-latest-win64-gpl/bin:$PATH"   # ffmpeg is installed but not on PATH

uv run ruff check --no-cache src tests examples
uv run lint-imports
uv run pytest tests -q
uv run ai-studio doctor
uv run ai-studio generate "a test clip" --provider stub
```

## What each one is actually for

| check | catches |
|---|---|
| `ruff check` | style and a real class of bugs (unused imports, mutable defaults) |
| **`lint-imports`** | **architecture drift** — the five layering contracts. This is the one people forget, and it is the one that stops the design eroding. |
| `pytest` | the format geometry, the H3 prompt schema, and the turbo-trap guard |
| `ai-studio doctor` | ffmpeg and its required filters, python version, credentials |
| `generate --provider stub` | the whole pipeline end to end, offline, free |

## Gotchas

- **`ruff --fix` and `ruff format` cannot write in this sandbox.** Run
  `ruff check` and apply the fixes with the edit tools.
- **`--no-cache` is required** — ruff cannot create `.ruff_cache` here.
- **ffmpeg is not on PATH**; export it first or `doctor` and the stub provider
  both fail.
- The Bash working directory persists between calls. `cd` back to the repo root
  or use absolute paths.
- **Always pass `encoding="utf-8"`** to `open()` and `subprocess`. The Windows
  default is cp950 and throws on any non-ASCII byte. Keep CLI output ASCII —
  em-dashes render as mojibake in the console.

## Expected state

At the time of writing: ruff clean, **5/5 contracts kept**, **54 tests passing**,
doctor green, stub producing 864×480 @ 24fps with audio.

If `lint-imports` reports a broken contract, do not add an exemption. The
contract is the architecture — the import is the thing that is wrong.
