# assets/metrics

Timestamped snapshots of what this project's GPUs have actually done. No
generated media lives in this repository; these numbers are the showcase.

| file | what | source |
|---|---|---|
| `sessions-<stamp>.json` | every pod session this calendar month, per GPU tier and per session: minutes, USD, $/hr, datacenter, VRAM, quantisation, why it closed | `runs/.spend_ledger.json` + `logs/sessions/*.json` |
| `measured-<stamp>.json` | every figure measured on our own hardware -- the 📏 rows in `docs/` -- with date and source | `ai_studio.benchmark.measured.MEASURED` |
| `benchmark-<stamp>.json` | per-`(kind, gpu_tier)` means over real renders (seconds, cost, VRAM, frames/s), one entry per month; **only written once at least one day has been folded** | `runs/benchmark/<month>.json`, folded daily by `ai-studio archive` |

`<stamp>` is UTC, `YYYYMMDDTHHMMSSZ`, so files sort by time and nothing is
overwritten. Every snapshot carries `generated_at`, `kind`, `note`, `data`.

Nothing in a snapshot identifies a request, a user, a group or a pod.

Regenerate:

```bash
uv run ai-studio metrics export          # writes new snapshots here
uv run ai-studio metrics readme          # re-renders the block in README.md from the latest of each kind
```

The docs grade every figure (`CLAUDE.md`, "Number honesty"): 📏 measured by
us, `[reported]` quoted, `[speculative]` inferred. Only 📏 is exported. A
figure that is not here has not been measured.
