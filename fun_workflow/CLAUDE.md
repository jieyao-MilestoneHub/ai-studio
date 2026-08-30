# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code under `fun_workflow/`. It is read together with the repository root
`CLAUDE.md`, not instead of it — the root file's **Money**, **Two traps** and
**Platform traps** sections apply here in full, because this package is what
actually spends the money. Its architecture sections describe ai-studio, the
package this one is built on.

## What this is

The LINE group's playground: everything between a message in a group chat
and a rendered file back in it. A FastAPI service + LINE bot lets the group
trigger any model ai-studio serves — `/影片` for video, `/圖片` for image,
photo then `/圖影` / `/圖圖` for image-to-video / image-to-image, photo/audio/
video then `/說圖` / `/說音` / `/說影` to describe it, `/himonkey` for plain-text
chat via gpt-oss-20b, video then `/影音` to get its audio track as an M4A
(ffmpeg on the host, no pod), `/短劇` for a one-minute six-beat drama with a
stable lead (gpt-oss-20b screenplay into a fixed beat template → Flux
keyframes → H3 image-to-video with in-clip cuts → timeline → ffmpeg splice;
`docs/drama.md`), and「讓我看看」— quote-reply an earlier request to
pull that one result back as a free reply when push quota is gone. One
spelling each, no aliases. Full spec: `docs/line-bot.md`.

**The boundary.** ai-studio (the root package, installed editable from `..`)
owns the pod, the models, the money and the measurements. This package owns
*who asked, what they get back, and when*:

| here | there (ai-studio) |
|---|---|
| `api/main.py` — `POST /callback`, `GET /q/{token}` (the status page users open), `/files`, `/healthz` | — |
| `bots/line/` — signature, Reply/Push/Content clients, trigger parsing, caps, media pairing | — |
| `pipeline/queue.py` — the SQLite request queue (`runs/queue.sqlite3`), the audit trail | `runtime/` — pod lifecycle, budget, daily open cap |
| `pipeline/worker.py` — the always-on loop; `pipeline/drain.py` — one render per kind | `providers/`, `comfy/`, `inference/` — the models |
| `pipeline/convert_worker.py` — which rewriter each request needs; `prompts/understanding.py` — the questions `/說圖 /說音 /說影` send (`prompt` + `audio_prompt`), their rewriter, and `compose_answer` (【畫面】/【聲音】) | `prompts/{convert,flux}` + `pipeline/pod_llm` — the H3/Flux rewriting itself; the pod server, which holds no wording |
| `core/kinds.py` — `JobKind`, the queue's discriminator (with DRAMA); `pipeline/idle.py` — the grace each kind earns a quiet pod | `core/enums.MediaKind` — what a model serves; `runtime.session.touch_activity(grace_minutes=)` — the clock |
| `pipeline/drama.py`, `prompts/drama.py`, `core/drama_spec.py` (the beat template, the framing rules), `workflows/flux_dev_i2i_face.json`, `deploy/pod_setup.d/face_repair.sh` — `/短劇` | `prompts/h3`, `providers/flux` (takes the face graph as `i2i_face_workflow=`), `editing/{transitions,rhythm}`, `render/timeline`, `media.assemble` — the grammar the drama is cut by, `pod_setup.sh` (runs the shipped extension) |
| `bots/line/limits.py` — every LINE ceiling, passed into ai-studio's tools as parameters | `media.poster/extract_audio(max_bytes=)`, providers' `max_output_chars` |
| `prompts/chat.py` — the `/himonkey` persona | `providers/chat` |
| `config/settings.py` — `FunSettings`: LINE credentials, caps, delivery dirs, drama knobs; `.studio` is ai-studio's `Settings` | `config/settings.py` — GPU, money, logs |
| `storage/index.py` (delivery index), `storage/gc.py` (chat_turns, empty drama dirs, what to archive) | `storage/archive.py` — the daily tar, generic |
| `deploy/{jetson,vps}_setup.sh` — the always-on host | `deploy/{pod_setup.sh,inference_server.py}` — the pod |

The status page shows `使用的 GPU` and `GPU 租用價格`: that data is
ai-studio's (`Job.gpu_tier`/`gpu_usd_per_hr`, persisted at claim from the
session; `ai_studio.benchmark` for the aggregates). This side renders it and
must not compute it.

## Commands

```bash
cd fun_workflow
uv sync --group dev                       # installs ai-studio from .. editable

uv run pytest tests -q
uv run ruff check --no-cache src tests    # --fix sorts the ai_studio / fun_workflow import blocks
uv run lint-imports
uv run mypy

uv run funapp preflight --skip-suite      # signature+dedupe, queue->rewrite, hold, push (opt-in), /files range
uv run funapp drama-dryrun                # the whole /短劇 machine offline: stub Flux/H3, real ffmpeg
AI_STUDIO_RUNS_DIR=/tmp/x uv run funapp worker --max-ticks 1   # one idle tick, no pod

uv run funapp line serve                  # the always-on service (needs LINE_CHANNEL_SECRET)
uv run funapp line capture-group          # discover a group id
uv run funapp worker                      # the loop: opens a pod on demand, renders, pushes
uv run funapp drain                       # render on a pod someone opened by hand
uv run funapp reap                        # every minute: close a quiet pod (never with work queued)
uv run funapp gc | archive                # daily chores; archive wraps `ai-studio archive`
```

From the repo root the same commands are `uv run --project fun_workflow
funapp ...`. **Run production commands from the repo root** (the systemd
units do): `runs/`, `files/`, `incoming/`, `logs/` and `.env` are resolved
from the cwd, and the queue database is `runs/queue.sqlite3` there.

## Layering

Enforced by `import-linter` (`pyproject.toml`) plus one test:

```
cli → api → bots → pipeline → prompts → storage → config → core
```

- **Only `cli` may import `ai_studio.runtime`, and nothing imports
  `ai_studio.cli`** (`tests/unit/test_layering.py`; import-linter cannot
  forbid a subpackage of an external package). The checklist machinery both
  preflights share is `ai_studio.checks`. The worker loop takes the session, providers, LLM and
  delivery through `pipeline.worker.WindowHost`, implemented by
  `cli.main._RuntimeHost`. That injection is what keeps every pipeline test
  free of a pod, and it is the seam the two packages were split along.
- `api` gets "is a pod warm" as `create_app(is_warm=...)` from `cli`, for
  the same reason.
- `prompts` does no I/O (same rule as upstream).

### Invariants

**Enqueue first, convert later, claim only `parsed`.** LINE needs its `200`
in under two seconds, so the webhook only writes a row. The worker's
`prepare` phase converts queued rows on the pod (one gpt-oss residency per
batch), and only a `parsed` job ever reaches a GPU. A request that never
becomes a valid prompt never costs a GPU-second.

**`event_id UNIQUE` and a single-statement `claim_next`.** LINE redelivers
webhooks; two drainers must not pay for the same clip. Both are enforced by
the database, not by code that remembers to check.

**Delivery order is complete → push → mark.** A push that fails leaves the
row undelivered so「讓我看看」can hand it over free; `quota-exhausted*`
outcomes flip a `meta` flag the webhook reads to tell the next requester.

**The reaper never closes a pod with work queued.** `funapp reap` computes
`hold` from the queue and passes it to `ai_studio.runtime.session.close_if_idle`;
that bool is the only thing this side tells the pod runtime about the queue.
The grace was already recorded by the worker with its last render
(`_RuntimeHost.touch_activity` → `pipeline.idle.grace_for`).

**Every question is ours; the pod holds no wording.** `convert_question`
returns `(prompt, audio_prompt, how)`; a bare `/說影` sends
`VIDEO_DEFAULT_QUESTION` to the frame model and `VIDEO_AUDIO_QUESTION` to
the audio model (the split found them sharing the frame question), and
`compose_answer` joins the two `sections` under 【畫面】/【聲音】. `funapp
rewrite-question` prints what a job would send.

**Caps are wired at the composition root or they do not exist.**
`WebhookHandler` defaults every cap to off; `api.main.create_app` passes each
one from `FunSettings`. `tests/unit/test_drain_wiring.py` is the record of
the time a configured-but-unpassed guard cost real money.

## Money (what this side adds)

- Every path that opens a pod goes through `ai_studio.runtime.session.ensure_pod`
  — the monthly guard and the daily open cap live there, not here. Do not add
  a second way.
- `AI_STUDIO_MAX_JOBS_PER_USER_PER_DAY`, `..._CHAT_MESSAGES_PER_USER_PER_DAY`,
  `..._MAX_DRAMAS_PER_DAY` (group-wide; a drama is ~15–30 GPU-minutes) are
  checked **before** enqueue. `AI_STUDIO_DRAMA_ENABLED` (default **false**
  since 2026-08-30) switches `/短劇` off entirely, ahead of those caps. `AI_STUDIO_MAX_CHAT_MONTH_USD` is a sub-ceiling
  checked in `pipeline.drain.render_chat` before submit.
- LINE push messages have a monthly quota; a 429 degrades to one text message
  and the pull path, never to a retry loop.

## Deploy

`deploy/jetson_setup.sh <static-domain>` (the always-on box on the desk,
ngrok front) or `deploy/vps_setup.sh <hostname>` (fresh VPS, Caddy front).
Both write `ai-studio.service` (`funapp line serve`), `ai-studio-worker.service`
(`funapp worker`), and four timers: `reap` (`funapp reap`, every minute),
`close` (`ai-studio session close`, 04:05 Asia/Taipei), `gc` (`funapp gc`,
02:30), `archive` (`funapp archive`, 03:00). Every unit runs
`uv run --project <repo>/fun_workflow ...` with `WorkingDirectory=<repo>`.
Cutover from a checkout that predates this package: `git pull; uv sync;
(cd fun_workflow && uv sync); sudo bash fun_workflow/deploy/jetson_setup.sh
<domain>; systemctl daemon-reload; systemctl restart ai-studio ai-studio-worker`.
In-flight `running` jobs are requeued by `release_running` on the worker's
next start; nothing under `runs/` moves.

## Docs

- `docs/line-bot.md` — the triggers, the two-second budget, capture mode,
  allowlists, caps, push-vs-reply economics, the setup runbook.
- `docs/drama.md` — `/短劇`: screenplay → stills → clips → assembly, the
  resume state file, cost and lease gates, face repair.
- Root `docs/schedule.md` and `docs/observability.md` — the pod's lifecycle
  and the logs/archive, shared with ai-studio.
