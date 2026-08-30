# fun_workflow

The LINE group's playground, built on [ai-studio](../README.md): everything
between a message in a group chat and a rendered file back in it. A FastAPI
webhook takes the request in under two seconds, a SQLite queue holds it, a
worker opens a GPU pod on demand, rewrites the request into the model's best
prompt, renders, and pushes the result back — with a status page per request
that names the GPU it ran on and what that GPU rents for.

ai-studio owns the pod, the models, the money and the measurements. This
package owns *who asked, what they get back, and when*. The two are one git
repository and two Python packages: this one installs ai-studio editable
from `..`, and nothing in ai-studio imports back.

## What the group can say

| trigger | with | you get |
|---|---|---|
| `/影片 <描述>` | text | a ~10 s MiniMax H3 clip |
| `/圖片 <描述>` | text | a Flux.1-dev image |
| photo, then `/圖影 <描述>` | a photo | that photo animated (image-to-video) |
| photo, then `/圖圖 <描述>` | a photo | that photo re-rendered (image-to-image) |
| photo, then `/說圖 [問題]` | a photo | what moondream3 sees (English) |
| audio, then `/說音 [問題]` | an audio clip | what Qwen2-Audio hears (繁體中文) |
| video, then `/說影 [問題]` | a video | 【畫面】 from Qwen2.5-VL and 【聲音】 from Qwen2-Audio |
| video, then `/影音` | a video | its audio track as an M4A — ffmpeg on the host, no pod |
| `/himonkey <話>` | text | a gpt-oss-20b reply that remembers the last few turns |
| quote-reply an earlier request with 「讓我看看」 | — | that result again, as a free reply, when the month's push quota is gone |

One spelling per trigger, no aliases. Every request is rewritten on the pod
by gpt-oss-20b into the shape its model wants (the H3 shot schema, a Flux
prompt, a per-model question) before a GPU-second is spent on it. The full
spec — the two-second webhook budget, capture mode, allowlists, per-user
and per-day caps, push-vs-reply economics — is
[docs/line-bot.md](docs/line-bot.md).

## Run it

```bash
cd fun_workflow
uv sync --group dev                      # installs ai-studio from .. editable

uv run pytest tests -q                   # 350+ tests, no GPU, no network
AI_STUDIO_RUNS_DIR=/tmp/x uv run funapp worker --max-ticks 1   # one idle tick of the loop

uv run funapp line capture-group         # discover your group's id (LINE exposes no API for it)
uv run funapp line serve                 # the always-on service: webhook, status pages, /files
uv run funapp worker                     # opens a pod when work arrives, renders, pushes
uv run funapp reap                       # every minute: close a quiet pod, never one with work queued
uv run funapp preflight                  # signature + dedupe against the deployed secret, queue → rewrite, hold, push (opt-in), /files range
```

Credentials go in the repo-root `.env` (`LINE_CHANNEL_SECRET`,
`LINE_CHANNEL_ACCESS_TOKEN`, `LINE_ALLOWED_GROUP_ID`, optional
`LINE_ALLOWED_USER_IDS`) next to ai-studio's RunPod key; `.env.example`
lists every name. **Run production commands from the repo root**: `runs/`
(the queue, session state), `files/` (delivered media), `incoming/`
(uploaded media) and `logs/` resolve from the working directory, which is
what the systemd units do.

`deploy/jetson_setup.sh <ngrok-static-domain>` (an always-on box you own)
or `deploy/vps_setup.sh <hostname>` (a fresh Debian VPS with Caddy) install
the service, the worker and four timers: `reap` every minute, `close` at
04:05, `gc` at 02:30 and `archive` at 03:00 Asia/Taipei. Every unit runs
`uv run --project <repo>/fun_workflow funapp …` with the repo root as its
working directory.

## Where the money and the numbers are

Every path that opens a pod goes through `ai_studio.runtime.session.ensure_pod`
— the monthly budget guard and the daily open cap live there, not here. This
side adds the caps that are checked *before* a request is even queued
(`AI_STUDIO_MAX_JOBS_PER_USER_PER_DAY`, `…_MAX_CHAT_MESSAGES_PER_USER_PER_DAY`,
`…_MAX_DRAMAS_PER_DAY`) and a sub-ceiling on chat spend.

The status page each result links to shows `使用的 GPU` and `GPU 租用價格`.
Those come from ai-studio (persisted onto the job at claim time, so they
survive the pod closing), and every real render logs a benchmark line that
`ai-studio archive` folds into a monthly per-GPU report — see
[ai-studio's README](../README.md#measured-on-the-gpu) and
[`assets/metrics/`](../assets/metrics/).

## Layout

```
src/fun_workflow/
  api/        POST /callback, GET /q/{token} (the status page), /files, /healthz
  bots/line/  signature check, Reply/Push/Content clients, trigger parsing, caps, media pairing, LINE limits
  pipeline/   the SQLite queue, the worker loop, drain, prompt conversion, the drama pipeline (switched off), idle grace
  prompts/    the questions the understanding models are asked, the /himonkey persona, the screenwriter
  core/       JobKind (what a request is; ai-studio's MediaKind is what a model serves), the drama spec
  storage/    the delivery index, gc
  cli/        funapp — the composition root, the only module that touches the pod runtime
deploy/       jetson_setup.sh, vps_setup.sh, pod_setup.d/face_repair.sh (shipped to the pod for drama keyframes)
workflows/    flux_dev_i2i_face.json (the keyframe graph with FaceDetailer)
docs/         line-bot.md, drama.md
```

Layering is enforced (`uv run lint-imports`, `tests/unit/test_layering.py`):
only `cli` may import `ai_studio.runtime`, and the worker loop takes the
session, providers and delivery through `pipeline.worker.WindowHost`, which is
why every pipeline test runs without a pod. [CLAUDE.md](CLAUDE.md) has the
invariants for anyone (person or agent) changing this package.

## Third-party content

The queue database, `files/`, `incoming/` and the chat history hold whatever
the group sends and gets back — messages, photos, voices, the results. None
of it is in this repository (`.gitignore`), and **you, running this bot, are
responsible for what your group generates and for the licences of the
models that generate it**: MiniMax H3's licence excludes several
jurisdictions and Flux.1-dev's is non-commercial (see the model docs under
[`../docs/`](../docs/)).
