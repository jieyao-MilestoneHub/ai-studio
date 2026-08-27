# Observability: tracing a request, and what gets kept

Since 2026-08-28. Before it, the worker configured no logging at all — every
`_log.info` (job durations, `_built_by`, a drama's six clips) went to Python's
last-resort handler and was dropped; 📏 after a full day of renders,
`journalctl -u ai-studio-worker | grep -c "done in"` returned 0. The first live
`/短劇` (44 minutes, two pods, a lease-boundary resume) left **no** worker line.

## Two sinks, one record

Every process calls `configure_logging()` once (`core/observability.py`,
wired by `cli/main.py::_setup_logging`) and writes the same record two ways:

| sink | where | format |
|---|---|---|
| journald | `journalctl -t ai-studio-worker` (each unit has its own `SyslogIdentifier`) | `2026-08-28T01:12:48.616+08:00 INFO ai_studio.worker [job_id=103 token=5h_3-ye_cJU kind=drama] clip 3/6` |
| JSONL | `logs/<service>/<YYYY-MM-DD>.jsonl` (local day) | one object per line, below |

Services: `webhook`, `worker`, `reap`, `close`, `gc`, `archive`, `session`.

```json
{"ts":"2026-08-27T16:35:22.569+00:00","local":"2026-08-28T00:35:22.569+08:00",
 "level":"INFO","logger":"ai_studio.drama","service":"worker","msg":"clip 4/6",
 "job_id":103,"token":"5h_3-ye_cJU","kind":"drama","stage":"clip",
 "sha256":"9cfac2093e1a","cost_usd":0.039453,"seconds":195.2}
```

| field | meaning |
|---|---|
| `ts` / `local` | the same instant, UTC and Asia/Taipei, both with offset and ms |
| `service` | which process wrote it |
| `job_id`, `token`, `kind` | the request — bound once around the render (`bind()`), inherited by every line inside |
| `stage` | `accept · prepare · claim · swap · render · character · keyframe · clip · level · concat · deliver · archive` |
| `seconds`, `cost_usd`, `sha256`, `built_by`, `model`, `pod_id`, `outcome`, `reason`, … | per-stage facts; only keys in `EXTRA_FIELDS` are written, so a stray `extra=` can never break a line |

## Tracing one request

The `token` is the same string as the status URL (`/q/<token>`) and the drama
run directory (`runs/drama/<token>/`).

```bash
# everything one request did, in order, across every service
grep -h '"token":"5h_3-ye_cJU"' logs/*/*.jsonl | jq -c '[.local,.service,.stage,.msg,.seconds,.cost_usd]'

# what the worker was doing today, human-readable
journalctl -t ai-studio-worker --since today

# every model swap and what it cost in seconds
jq -c 'select(.stage=="swap")' logs/worker/*.jsonl

# why a pod opened or closed
jq -c 'select(.msg|test("pod (opened|closed)|refused"))' logs/session/*.jsonl logs/reap/*.jsonl

# where a delivered file came from
grep '"path":"files/OD5_XWwNXUU.png"' files/index.jsonl
```

A completed request reads: `accepted` → `prepared` (`built_by`, rewrite
`seconds`) → `claimed` → `evicted` (the swap) → `submitted` / `fetched`
(`polls`, `seconds`) → `job N done in Ns` → `delivered` or `delivery held (push
quota)` → `pulled` (讓我看看). A drama adds `character`, `keyframe n/6`,
`clip n/6`, `leveled`, `drama assembled`, and `paused at video: Ns left on the
lease` / `resuming drama` around a lease boundary.

The reaper (`session reap`, every minute) logs to journald **only on a
transition** (held → active → closed) and DEBUG to JSONL every minute —
the per-minute `held:` line was ~65 % of journald volume before.

## Other records with timestamps

| record | written by | when |
|---|---|---|
| `logs/sessions/<pod>-<opened>.json` | `close_session` | every close — the pod's tier, quantisation, opened/closed, minutes, cost, and **why** (`idle`, `window over`, `scheduled close`, `manual`) |
| `logs/pods/<pod>/{setup,inference,comfy,dl-logs}.log` | `pull_pod_logs`, before `pod delete` | the pod's own logs (they die with the pod); bounded to 5 MB each and 30 s; a failure is a warning, the delete never waits |
| `runs/drama/<token>/state.json` | `save_state` | `created_at`/`updated_at`, per-artifact `created_at`, per-stage `started_at`/`finished_at` |
| `runs/drama/<token>/render_manifest.json` | assembly | `generated_at`, token, job id, spend, face-repair verdict, stage timings, every ffmpeg argv |
| `files/index.jsonl` | the worker, on completion | `{ts, token, job_id, kind, path, bytes, sha256}` — the only map from a random filename back to its request |
| `runs/spend-<YYYY-MM>.json` | the ledger, on month rollover | the retired month (it used to be discarded on the 1st) |
| pod `setup.log` / `inference.log` | `pod_setup.sh`, `inference_server.py` | UTC ISO timestamps on every line since 2026-08-28 |

## Backup: `ai-studio archive`

Daily at **03:00 Asia/Taipei** (`ai-studio-archive.timer`, after `gc` at
02:30). Same disk by decision — one NVMe, no NAS — an off-box copy later only
needs a destination added (rclone/rsync the `archive/` tree).

1. `sqlite3.Connection.backup()` of `runs/queue.sqlite3` (a WAL database must
   not be `cp`'d), then `PRAGMA wal_checkpoint(TRUNCATE)`.
2. tar of: JSONL days **before today**, `logs/sessions`, `logs/pods`,
   `runs/drama/*/{state,render_manifest}.json`, the ledger and retired months,
   `files/index.jsonl`, the snapshot → `zstd -19 -T0` (stdlib `lzma`
   fallback) → `archive/YYYY-MM-DD/ai-studio-<local stamp>.tar.zst`.
3. list the archive back; a missing member discards the tar and raises.
4. `manifest.json`: `ts`, host, git sha, every member's size + sha256, bytes
   before/after.
5. prune — **only what a manifest names**: hot logs older than
   `AI_STUDIO_LOG_HOT_DAYS` (30); archives older than
   `AI_STUDIO_ARCHIVE_KEEP_DAYS` (365); `runs/_dryrun`, `runs/_stub`, `out/`
   at 30 d; empty `runs/drama/<token>` dirs; `chat_turns` older than 30 d
   (the rolling chat window — `jobs` rows are the audit trail and are never
   deleted); `VACUUM`.

📏 First run (2026-08-28): 6 members, 280 KB → 50 KB. A second run the same
day skips the tar and only prunes. `ai-studio archive --dry-run` prints the
plan and touches nothing. `ai-studio doctor` shows both directories' size and
the last archive date.

Restore:

```bash
tar -I zstd -xf archive/2026-08-28/ai-studio-2026-08-28T011248+0800.tar.zst -C /tmp/restore
sqlite3 /tmp/restore/runs/snapshots/queue-2026-08-28.sqlite3 'select count(*) from jobs'   # or python's sqlite3
```

journald itself is bounded by `/etc/systemd/journald.conf.d/ai-studio.conf`
(`SystemMaxUse=1G`, `MaxRetentionSec=30day`, written by `jetson_setup.sh`) —
the JSONL trace is the durable record, the journal only covers the hot window.

## Settings

| env | default | |
|---|---|---|
| `AI_STUDIO_LOG_DIR` | `logs` | |
| `AI_STUDIO_LOG_LEVEL` | `INFO` | `DEBUG` adds the per-minute reaper lines to JSONL only |
| `AI_STUDIO_ARCHIVE_DIR` | `archive` | |
| `AI_STUDIO_LOG_HOT_DAYS` | `30` | |
| `AI_STUDIO_ARCHIVE_KEEP_DAYS` | `365` | `0` keeps every archive |

## Operator note: the installed timers

The Jetson's installed units predate this work: `ai-studio-gc.timer` is a bare
`18:30` and `ai-studio-close.timer` is `20:05 UTC`, while the repo says
`02:30 Asia/Taipei` / `04:05 Asia/Taipei`, and there is no archive timer or
journald drop-in yet. `sudo bash deploy/jetson_setup.sh <ngrok-domain>`
converges all of it (units, `SyslogIdentifier`, the four timers, the drop-in).
