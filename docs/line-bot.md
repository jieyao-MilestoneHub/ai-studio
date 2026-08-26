# The LINE bot

Someone says `/影片 一隻橘貓走在雨中` in a group; the finished clip is pushed
back into that group as a reply to their message, and a status page carries the same
thing for anyone who wants to look again later.

## Three triggers, one queue, one pod

| trigger | produces | model |
|---|---|---|
| `/影片 <描述>` | a video clip from text | MiniMax H3 |
| `/圖片 <描述>` | a still image | Flux.1-dev |
| a photo, then `/圖影 <描述>` | a video clip whose first frame is that photo | MiniMax H3 (I2V graph) |
| a photo, then `/圖圖 <描述>` | that photo re-rendered under the description | Flux.1-dev (img2img graph) |

**Length is optional, on the video triggers.** `/影片15s` or `/圖影15秒`
asks for a 15-second clip; the number sits flush against the trigger and the
rest is the prompt. The default is 10.1 s (243 frames), the ceiling 15.08 s
(362) — both 📏 measured on a 4090 to be stable in 24 GB, with 15 s costing
about four times the render time. Anything off the model's 17k+5 frame grid
or out of range is snapped and clamped, never refused. Image triggers have no
length: a number in a `/圖片` prompt is prompt.

**One spelling per trigger, no aliases.** `生成`, `畫圖`, `/gen` and `/img`
used to work and were retired together: a bare word that is also ordinary
Chinese is a request nobody meant, paid for in GPU-minutes, and a leading
slash is not something anyone types by accident. Anything that does not start
with one of the four strings gets no reply at all.

**Only `/圖影` and `/圖圖` touch the photo cache.** A photo posted to the group
waits five minutes for a photo-trigger from the same sender, and the first one
claims it. `/影片` after a photo is still text-to-video and leaves the photo
where it is; `/圖片` never looks. A photo-trigger with no photo cached (none
sent, sent by someone else, or older than five minutes) queues nothing and
replies with what to do, naming the trigger that was used — the user asked for
*their* picture, and a picture of something else is not that.

Both enqueue into the same SQLite `jobs` table (tagged `media_kind`), both wait
for the same on-demand pod, and both render on it —
`ai-studio worker` picks the H3 or Flux ComfyUI provider per job by
`media_kind`, not by a second queue or a second pod. See
[schedule.md](schedule.md) for how the pod is now opened by the day's first
request rather than by a timer, and [runpod.md](runpod.md) for what
downloading two model sets instead of one costs at window open.

**Delivery is identical for both**: the finished media is **pushed back into
the group** that asked for it, with the requester **@-mentioned** — a video
message object for `/影片` and `/圖影`, an image message object for `/圖片` and `/圖圖`, each with a
JPEG poster, followed by a text message carrying the mention and the status
link. A Flux image is fast to *generate* (`[speculative]`, ~15–40s) but still
only happens once a pod is up, which on a cold start is a couple of minutes after the
original message, so nothing about the delivery path differs between the two
kinds.

⚠️ **Flux.1-dev ships under a non-commercial licence** from Black Forest Labs
(Flux.1-schnell is Apache-2.0, unrestricted). If this bot's use is ever
commercial rather than personal/internal, that needs a separate commercial
licence from Black Forest Labs, or a switch to schnell. See
[model-flux.md](model-flux.md).

## Why it is shaped this way

Three LINE rules decide the whole architecture.

**1. A webhook must return 200 within two seconds.** So `POST /callback` does an
HMAC, a few comparisons, one SQLite insert and one reply — nothing else. The
prompt conversion and the GPU render both happen after the response has gone
out. It also means the receiver **cannot scale to zero**: a cold start does not
reliably fit in two seconds, so this side runs on a small always-on host.

**2. A reply token is single-use and lives about a minute.** The clip does not
exist for minutes at best, and until the next window at worst. So the reply is
only ever an acknowledgement — it physically cannot carry the result.

**3. Only replies are free.** Quoting the pricing page: *the number of messages
is counted by the number of people you send a message to* — so one push into a
twenty-person group costs twenty messages, whether it carries a video or a line
of text. Reply is the sole method not counted.

**So: reply acknowledges, push delivers.** Rules 1 and 2 are physics and shape
the code. Rule 3 is a price, and it is the one that got re-decided.

### The decision that was reversed

This document used to conclude from rule 3 that **nothing is ever pushed** and
that delivery is a link to a status page. On cost alone that was right, and it
still is. It was reversed anyway, on purpose:

- The product is a clip that lands in the group where somebody asked for it,
  addressed to them. A link to a status page is a receipt for a thing that
  happened somewhere else.
- The quota is now **managed** rather than avoided. This is one private group
  (`LINE_ALLOWED_GROUP_ID` is a hard allowlist), the volume is ~50 renders a
  month, and 429 is handled explicitly: `bots/line/push.py` degrades to a
  single text message with the link and logs a WARNING. What it never does is
  fail quietly — a delivery that goes silent is indistinguishable from a
  broken bot, so the user asks again and that costs another conversion and
  another GPU slot.

The constraints the link-based delivery avoided are therefore now met rather
than dodged. A LINE *video message* needs mp4 under 200MB on HTTPS with TLS
1.2+, a poster image under 1MB at a matching aspect ratio, and a host that
supports **HTTP range requests** — the last being the awkward one, since a
naive `StreamingResponse` does not.

- The poster comes from `media.poster()`: first frame of a clip, thumbnail of
  an image, always a **new JPEG** (a 1024×1024 Flux PNG routinely clears 1MB
  on its own), shrunk down a width/quality ladder until it fits, aspect ratio
  preserved throughout. If it cannot be made, delivery degrades to text and a
  link rather than being abandoned — a thumbnail is not worth losing a clip
  that cost GPU-minutes over.
- Range requests: `/files/{name}` is a Starlette `FileResponse`, which answers
  `Range` with a 206 and a correct `Content-Range` 📏 (verified on Starlette
  1.6.0, and pinned by tests in `tests/unit/test_api.py` — a video message
  fails in a very hard-to-trace way without it).
- Volume is still trivial: a measured 5-second clip is **0.99MB** 📏, so ~50 a
  month is ~50MB, and a directory on the host is the entire storage layer.

### `done` is not `delivered`

The queue carries a `delivered_at` timestamp beside the job state. Rendering
and announcing fail independently — the clip can exist while the push is
refused for quota — and without the distinction a worker restart re-pushes
everything it finds finished, at full per-recipient price, telling the user
nothing new. The order is **complete → push → mark delivered**: a push that
succeeds and a mark that then fails sends one extra message, while the reverse
loses the delivery entirely. One duplicate beats one silence.

A **failed** request is delivered too, for the same reason and more urgently.
On success the user eventually sees something appear; on failure, silence is
the only signal they get.

## What runs where

| | where | when |
|---|---|---|
| Webhook, status pages, file downloads, queue | small VPS (or your machine + a tunnel) | always |
| Prompt conversion LLM | RunPod serverless, scale-to-zero | on demand |
| MiniMax H3 + Flux.1-dev | same RunPod GPU pod | on demand, any hour; reaped 5/10 min after the last render |

Two separate RunPod resources on purpose. A measured H3 run peaked at **43.3GB**
📏 on a 48GB card, leaving no room for a useful instruct model beside it — and
the LLM has to answer when the pod does not exist, because conversion happens
when the request arrives, not twelve hours later.

## Setting it up

### 1. LINE console

Four things only you can do:

- **Messaging API → Use webhook: on**
- **Allow bot to join group chats: on** (off by default; without it the account
  leaves any group it is added to)
- **In LINE Official Account Manager, turn OFF auto-reply and greeting
  messages** — otherwise they talk over the bot
- Set the Webhook URL to `https://<your-host>/callback`, then press **Verify**

Verify sends `{"destination": ..., "events": []}` and needs a 200. The handler
returns one; a handler that indexed `events[0]` would fail here.

### 2. Credentials

```bash
cp .env.example .env
# LINE_CHANNEL_SECRET         Basic settings
# LINE_CHANNEL_ACCESS_TOKEN   Messaging API -> issue a long-lived token
# LINE_ALLOWED_GROUP_ID       leave empty for now
# AI_STUDIO_PUBLIC_BASE_URL    https://<your-host>
```

`.env` is gitignored and a test asserts it (`tests/unit/test_gitignore.py`).

> If a token has ever been pasted somewhere it might be logged — a chat, a
> ticket, a screenshot — reissue it in the console once everything works. Old
> tokens die instantly and it costs nothing.

### 3. Find the group id

**There is no API for this.** LINE documents that an account cannot list the
groups it belongs to, so the id has to be read off a live event.

```bash
uv run ai-studio line capture-group
```

With `LINE_ALLOWED_GROUP_ID` empty the bot runs in **capture mode**: it answers
the trigger word with the group's id and *accepts no work*. An unset allowlist
must never mean "serve everyone".

Add the account to the group, say `/影片 test`, and the id is printed to the
console and replied into the chat. Put it in `.env` and restart.

### 4. Decide who inside the group may spend money

The group allowlist answers *which chat*, not *which people*. Those are
different questions, and the second one is the expensive one: **group membership
is not yours to control.** Any existing member can invite someone, and from the
moment that person joins they can trigger a render — at roughly 5 minutes of GPU
time each.

```bash
LINE_ALLOWED_USER_IDS=Uabc...,Udef...
```

Empty means *any member of the group*. That is a legitimate choice for a group
whose roster you trust, and it is the default because requiring a user list up
front would make the bot unusable before anyone had ever spoken to it. But it is
a choice, so `ai-studio line serve` prints it in yellow at startup rather than
letting it pass as a default.

With the list set, the gate **fails closed**: LINE does not always include
`source.userId` on an event, and "we could not tell who this was" must not
resolve to "let them spend a GPU-hour".

Collecting the ids needs no extra tooling. Every accepted request logs
`user=<id>` and every refusal logs the id it refused, so the log is enough to
authorise someone:

```
2026-08-25 11:03:22 INFO    ai_studio.webhook | callback ok: wrong_user (U7d3...)
```

Two deliberate limits on the gate:

- It sits in front of the **paid action only**. Ordinary chat from a stranger is
  ignored in silence — a bot that answered every message in a group it barely
  serves would be intolerable.
- A refusal costs nothing to send, because replies are free, so the person is
  told rather than left wondering.

**Joins and leaves are reported.** Nothing polls the roster, so `memberJoined` is
the only notice a change produces; it is logged at WARNING, with a second line
if no user allowlist is set:

```
WARNING ai_studio.webhook | member(s) JOINED the group: U7d3...
WARNING ai_studio.webhook |   no LINE_ALLOWED_USER_IDS set: they can trigger a render now
```

Both lines are asserted in `tests/unit/test_api.py`: two warnings with an open
allowlist, one with a closed one, and none at all for a join in a group this bot
does not serve. They live in the FastAPI route rather than in `WebhookHandler`,
which is why the handler-level tests could not reach them and they went
unasserted for a while.

> **Still not verified: whether this account can read the roster directly.**
> LINE has a *Get group chat member user IDs* endpoint, which would let the
> roster be read instead of watched. The docs do not make clear whether it is
> limited to verified or premium accounts, so nothing here depends on it.
>
> One command settles it, on the VM with real credentials:
>
> ```bash
> curl -H "Authorization: Bearer $LINE_CHANNEL_ACCESS_TOKEN" >   https://api.line.me/v2/bot/group/$LINE_ALLOWED_GROUP_ID/members/ids
> ```
>
> **200** = available to this account. **403**, or a body saying *not available
> for this account*, = verified/premium only. Replace this box with the answer
> and the date it was obtained — an open question that stays open for three
> months stops being a question and starts being folklore.

### 5. Serve

```bash
uv run ai-studio line serve --port 8000
```

## Public HTTPS

LINE needs a publicly reachable HTTPS URL with a certificate from a normally
trusted CA. Self-signed will not do.

### Cloudflare Tunnel — no VPS needed

A **named** tunnel, not a quick one: the free `trycloudflare.com` hostname
changes on every restart, and you would be re-pasting the webhook URL into the
console each time.

```bash
cloudflared tunnel login
cloudflared tunnel create ai-studio
cloudflared tunnel route dns ai-studio vg.example.com
cloudflared tunnel run --url http://localhost:8000 ai-studio
```

Then `AI_STUDIO_PUBLIC_BASE_URL=https://vg.example.com`.

Caveat worth stating plainly: **the machine has to stay awake.** Webhooks
arriving while it sleeps are lost, and the 11:00 `session open` will not fire
either — `--terminate-after` guarantees a pod gets closed, never that one gets
opened.

### ngrok on a box you already own — the Jetson

cloudflared's control channel (UDP/TCP 7844) is blocked egress on the network
the Jetson sits on; ngrok tunnels over 443 and gets through. Reserve a
**static domain** in the ngrok dashboard (the free plan includes one): the
ephemeral hostname changes on every restart, and `AI_STUDIO_PUBLIC_BASE_URL`
is baked into reply links, `/files` media URLs and the worker's delivery URLs
at process start, so an ephemeral name means re-pasting the webhook URL and
restarting both services every time the tunnel bounces.

```bash
ngrok config add-authtoken <token>            # once, as the service user
sudo bash deploy/jetson_setup.sh <name>.ngrok-free.app
```

That writes three services (`ai-studio-ngrok`, `ai-studio`, `ai-studio-worker`)
and the two closing timers, all running as you from the repo checkout, and
masks the sleep targets. It installs no tools: `uv`, `runpodctl`, `ngrok` and an
ffmpeg ≥ 8.0 (for `colordetect`) must already be in `~/.local/bin` — `ai-studio
doctor` checks the ffmpeg half, the script checks the rest before writing a
unit. Then `AI_STUDIO_PUBLIC_BASE_URL=https://<name>.ngrok-free.app`.

The service binds `127.0.0.1:8000` only; ngrok forwards to it on the same host.
`/files` still answers `Range` with 206 through ngrok, which is what LINE's
video messages need.

### A small VPS — the previous setup

**Hetzner CAX11** (2 vCPU ARM, 4GB, 40GB) is the cheapest thing that works,
around EUR 3-4/month. Location **Singapore**: closest to LINE's servers for the
two-second budget, and closest to the people downloading the clips. ARM is fine —
the always-on side is pure Python and SQLite with no compiled wheels of
consequence.

One command on a fresh Ubuntu box:

```bash
sudo bash deploy/vps_setup.sh vg.example.com
```

It installs uv, ffmpeg (for the `ffprobe` the provider uses to check what it
actually fetched), Caddy for automatic TLS, a systemd service for the webhook,
and three systemd timers for the window.

**No domain?** Pass the box's IP with dots as dashes plus `.sslip.io`:

```bash
sudo bash deploy/vps_setup.sh 203-0-113-7.sslip.io
```

sslip.io resolves that to the IP with no account and no DNS to manage, and Let's
Encrypt issues a real certificate for it. That matters because LINE requires
HTTPS from a normally trusted CA — a bare IP or a self-signed cert is refused.

Two deliberate choices in the generated units:

- The web service is **not** socket-activated and never scales to zero. A cold
  start does not reliably fit LINE's two-second budget.
- The window timers use `Persistent=false`. A missed `open` must **not** fire
  late: booting a GPU pod at 3am because the box was down at 11:00 is exactly
  the unattended spend this project exists to avoid. `--terminate-after` on the
  pod still guarantees that a pod which *did* open gets closed, even if this box
  dies mid-window.

The window scheduler lives on this host rather than on a laptop, so it does not
depend on someone's machine being awake — see [schedule.md](schedule.md).

Credentials are not written by the script. Put them in `/srv/ai-studio/.env`
yourself (`chmod 600`), then `systemctl restart ai-studio`.

## The flow, end to end

```
/影片 一隻橘貓走在雨中
  → verify signature over the RAW body
  → group allowlist
  → dedupe on webhookEventId
  → enqueue, reply with a link, return 200        (< 2s)
  → background: LLM turns it into an H3 prompt    (seconds)
  → the worker sees queued work and opens a pod (any hour)
  → it renders                                    (~5 min each)
  → push: video + text, @ the requester           (billed per recipient)
  → mark delivered, so a restart cannot re-push
```

A user can also ask `好了嗎` at any time — that is a free reply, so polling by
asking costs nothing.

### Outside business hours

Business hours are **11:00-13:00 Asia/Taipei**
([schedule.md](schedule.md)). Outside them the bot **still accepts the
request** — it goes into the queue and waits for the next window. Refusing
would mean whatever someone thought of at midnight is simply lost, which is a
worse outcome than waiting until eleven, and the request costs nothing to hold.

What changes is only the acknowledgement. In hours:

```
收到 ✓ 排隊第 1 位,正在解析你的描述
進度與下載 → https://<host>/q/<token>
```

Out of hours:

```
收到 ✓ 排隊第 1 位
營業時間 11:00-13:00,已排入下一個時段(約 08/26 11:00),完成後會在群組通知你
進度與下載 → https://<host>/q/<token>
```

The queue position is true either way; on its own at 03:00 it reads as
"shortly" and is off by eight hours, so out of hours it is followed by when
"shortly" actually is. The next opening comes from `runtime.hours.next_open`,
which is also what the worker and the pod lease read — there is one copy of
11:00 and 13:00 in the codebase, in `runtime/hours.py`.

The status page says the same thing independently (`api/main.py`'s `_STATE_ZH`:
「已排入佇列,GPU 服務時段 11:00-13:00」), so a user who follows the link out of
hours is not told something different from what the reply told them.

### The per-user daily cap

`AI_STUDIO_MAX_JOBS_PER_USER_PER_DAY` (default 10, `0` disables) is checked
**before** the request is enqueued, so a refusal does not also spend an LLM
conversion on something that will never run. It counts every request the user
has had accepted since Taipei midnight, failures included — the cap is on
asking, not on succeeding, or a user whose prompts keep failing validation
would have an unlimited allowance. Together with the group and user
allowlists, that is the whole of the authorisation model: no token buckets, no
priorities, no concurrency above one.

## The details that bite

- **Never `await request.json()` before verifying.** The signature is over the
  raw bytes; parsing and re-encoding changes them and it will never match. There
  is a test for exactly this.
- **Dedupe on `webhookEventId`.** LINE redelivers when it does not get a 2xx, and
  says the same event may arrive more than once. The queue enforces it with a
  UNIQUE index rather than in Python, so it holds across processes and restarts.
- **A failed reply must not fail the request.** A non-2xx makes LINE redeliver
  the event, and repeated failures make it suspend delivery to the bot
  altogether. The 200 matters more than the message.
- **Loading animations are one-to-one only** and cannot be used in a group, which
  is why the acknowledgement text carries the queue position instead.
- **`--terminate-after` is set on every pod.** It is the backstop for the case
  where this host dies mid-window.

## Measured: the conversion endpoint

Deployed as `runpod-workers/worker-vllm` on `ADA_24` with
`--model-reference https://huggingface.co/Qwen/Qwen2.5-7B-Instruct:main`,
`workers-min 0`, `idle-timeout 120`. RunPod pinned `:main` to a specific commit
(`a09a3545…`) at create time, so the endpoint is reproducible.

| | 📏 measured |
|---|---|
| **Cold call, total** | **182.9 s** |
| — of which `delayTime` (queue + model load) | 173.9 s |
| — of which `executionTime` | 2.7 s |
| **Warm call, total** | **6.0 s** |
| — `delayTime` warm | 0.1 s |
| Cost per cold call, at $1.10/hr | ≈ $0.054 |
| Cost per warm call | ≈ $0.001 |

**The plan's assumption was wrong and this corrects it.** RunPod's documentation
says host-side caching makes cold starts "drop to seconds", and the golden path
that measured it used a **0.92GB** model. Ours is a 7B at roughly 15GB, and it
takes about **three minutes** to come up. `--model-reference` is still doing its
job — the weights are not downloaded and download time is not billed — but the
load into VRAM is real work.

Consequences worth designing around:

- A user waits up to three minutes to see how their sentence was parsed. That is
  acceptable here only because the clip itself does not arrive until the next
  window; it would not be acceptable if conversion were on the critical path to
  an immediate answer.
- `idle-timeout 120` means a burst of messages in a group pays **one** cold start,
  not one each. At 50 requests a month all landing cold, the endpoint costs about
  $2.70/month; clustered, far less.
- Creating the endpoint is free. With `workers-min 0` nothing bills until a
  request arrives.

### On the model's schema adherence

The measured conversion of *"一隻橘毛走在下雨天的路上，然後我只看到類似梵谷畫的
像素貓"* produced a valid prompt, but Qwen2.5-7B followed the schema loosely: it
returned **one** shot rather than splitting the style change into a second one as
instructed, and used the mood word *"melancholic"* in `non_diegetic_music`, which
the schema explicitly forbids.

So the validation layer is earning its place — the output is schema-*valid*
because `prompts/h3.py` enforces structure — but the *instructions* are only
partly obeyed. A larger model, or a few-shot example in the system prompt, is the
obvious next lever if shot splitting matters.

## Cost

| | per month |
|---|---|
| GPU pod, up to 2h/day (depends on which ladder rung serves) | $21–60, ~$48 expected |
| VPS | $4–6 |
| LLM serverless, ~50 video conversions | ~$0.10 |
| Image prompt conversion (Flux, one LLM call per request, same endpoint) | ~$0.002 each, in the line above |
| **LINE messages** | **$0** — replies only |
| **Object storage** | **$0** — ~50MB on the host |

Which rung served a run is recorded in `job.gpu_tier` and shown on the status
page, because rungs 3 and 4 have 24GB and must run the softer `low_vram` LoRA
mode. Falling down the ladder is a quality change, not only a price change.

**"$21–60" is what the ladder alone can cost, and its own worst case
($60.24 at rung 1) already exceeds a $50/month target before the VPS is even
added.** That range was a description, not an enforced number, until
`runtime.budget.MonthlyBudgetGuard` — see [schedule.md](schedule.md) for the
demand gate and budget guard that make `AI_STUDIO_MAX_MONTH_USD` (default $50)
an actual ceiling: skip a window with nothing queued, refuse to open one the
remaining month's budget can't cover, and shrink one it can only partly cover.

Flux's prompt path (`prompts/flux.py`) makes one LLM call per request, like
H3's — a Chinese-to-English translation validated as JSON, two attempts, then
a template fallback that costs nothing. It used to be a strip/truncate pass
with no LLM call; that stopped being true in `9e43143`. What it still does
not need is H3's structured shot schema — see [model-flux.md](model-flux.md).
