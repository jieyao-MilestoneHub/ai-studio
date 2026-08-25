# The LINE bot

Someone says `生成 一隻橘貓走在雨中` in a group; a link to the finished clip
appears on a status page they can open any time.

## Two triggers, one queue, one pod

| trigger | also matches | produces | model |
|---|---|---|---|
| `生成` | `/生成`, `/gen` | a video clip | MiniMax H3 |
| `畫圖` | `/畫圖`, `/img` | a still image | Flux.1-dev |

Both enqueue into the same SQLite `jobs` table (tagged `media_kind`), both wait
for the same 11:00–13:00 window, and both render on the same pod — `session
drain` picks the H3 or Flux ComfyUI provider per job by `media_kind`, not by a
second queue or a second pod. See [schedule.md](schedule.md) for how that
window now decides *whether* to open (a demand gate: no queued work of either
kind, no spend that day) and [runpod.md](runpod.md) for what downloading two
model sets instead of one costs at window open.

**Delivery is identical for both**: a link on the status page, never an inline
image or video message. This isn't a missed opportunity for a faster image
path — a Flux image is fast to *generate* (`[speculative]`, ~15–40s) but still
only happens during the scheduled window, which can be hours after the
original message. By then the reply token that triggered it (single-use, ~1
minute) is long dead, so there is no way to attach the result to that original
reply regardless of how quickly it's ready. The status-page link is the only
delivery mechanism this architecture has, for either kind.

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
exist for minutes at best, and until tomorrow's window at worst. So the reply is
only ever an acknowledgement, and the result is never pushed.

**3. Only replies are free.** Quoting the pricing page: *the number of messages
is counted by the number of people you send a message to* — so one push into a
twenty-person group costs twenty messages, whether it carries a video or a line
of text. Reply is the sole method not counted.

Hence **delivery is a link in a free reply, and nothing is ever pushed**. The bot
costs nothing to run in LINE terms at any volume.

That also removed a pile of constraints. A LINE *video message* would need mp4
under 200MB on HTTPS with TLS 1.2+, a poster image under 1MB at a matching
aspect ratio, and a host that supports **HTTP range requests** — the last being
the awkward one, since a naive `StreamingResponse` does not. A link needs none of
it. And the volume is trivial: a measured 5-second clip is **0.99MB** 📏, so ~50
a month is ~50MB, and a directory on the host is the entire storage layer.

## What runs where

| | where | when |
|---|---|---|
| Webhook, status pages, file downloads, queue | small VPS (or your machine + a tunnel) | always |
| Prompt conversion LLM | RunPod serverless, scale-to-zero | on demand |
| MiniMax H3 + Flux.1-dev | same RunPod GPU pod | 11:00–13:00 Asia/Taipei, only if the queue is non-empty |

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
# VIDEOGEN_PUBLIC_BASE_URL    https://<your-host>
```

`.env` is gitignored and a test asserts it (`tests/unit/test_gitignore.py`).

> If a token has ever been pasted somewhere it might be logged — a chat, a
> ticket, a screenshot — reissue it in the console once everything works. Old
> tokens die instantly and it costs nothing.

### 3. Find the group id

**There is no API for this.** LINE documents that an account cannot list the
groups it belongs to, so the id has to be read off a live event.

```bash
uv run videogen line capture-group
```

With `LINE_ALLOWED_GROUP_ID` empty the bot runs in **capture mode**: it answers
the trigger word with the group's id and *accepts no work*. An unset allowlist
must never mean "serve everyone".

Add the account to the group, say `生成 test`, and the id is printed to the
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
a choice, so `videogen line serve` prints it in yellow at startup rather than
letting it pass as a default.

With the list set, the gate **fails closed**: LINE does not always include
`source.userId` on an event, and "we could not tell who this was" must not
resolve to "let them spend a GPU-hour".

Collecting the ids needs no extra tooling. Every accepted request logs
`user=<id>` and every refusal logs the id it refused, so the log is enough to
authorise someone:

```
2026-08-25 11:03:22 INFO    videogen.webhook | callback ok: wrong_user (U7d3...)
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
WARNING videogen.webhook | member(s) JOINED the group: U7d3...
WARNING videogen.webhook |   no LINE_ALLOWED_USER_IDS set: they can trigger a render now
```

> Not verified: LINE has a *Get group chat member user IDs* endpoint, which would
> let the roster be read directly instead of watched. I could not confirm from
> the docs whether it is limited to verified or premium accounts, so nothing here
> depends on it. Worth checking if you ever want a real roster sync.

### 5. Serve

```bash
uv run videogen line serve --port 8000
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
cloudflared tunnel create videogen
cloudflared tunnel route dns videogen vg.example.com
cloudflared tunnel run --url http://localhost:8000 videogen
```

Then `VIDEOGEN_PUBLIC_BASE_URL=https://vg.example.com`.

Caveat worth stating plainly: **the machine has to stay awake.** Webhooks
arriving while it sleeps are lost, and the 11:00 `session open` will not fire
either — `--terminate-after` guarantees a pod gets closed, never that one gets
opened.

### A small VPS — the chosen setup

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
yourself (`chmod 600`), then `systemctl restart videogen`.

## The flow, end to end

```
生成 一隻橘貓走在雨中
  → verify signature over the RAW body
  → group allowlist
  → dedupe on webhookEventId
  → enqueue, reply with a link, return 200        (< 2s)
  → background: LLM turns it into an H3 prompt    (seconds)
  → 11:00: the window opens, the drainer renders  (~5 min each)
  → the status page shows the download
```

A user can also ask `好了嗎` at any time — that is a free reply, so polling by
asking costs nothing.

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
| Image prompt conversion (Flux, no LLM call — see below) | $0 |
| **LINE messages** | **$0** — replies only |
| **Object storage** | **$0** — ~50MB on the host |

Which rung served a run is recorded in `job.gpu_tier` and shown on the status
page, because rungs 3 and 4 have 24GB and must run the softer `low_vram` LoRA
mode. Falling down the ladder is a quality change, not only a price change.

**"$21–60" is what the ladder alone can cost, and its own worst case
($60.24 at rung 1) already exceeds a $50/month target before the VPS is even
added.** That range was a description, not an enforced number, until
`runtime.budget.MonthlyBudgetGuard` — see [schedule.md](schedule.md) for the
demand gate and budget guard that make `VIDEOGEN_MAX_MONTH_USD` (default $50)
an actual ceiling: skip a window with nothing queued, refuse to open one the
remaining month's budget can't cover, and shrink one it can only partly cover.

Flux's own prompt path is a simple strip/truncate/validate pass with no LLM
call (`prompts/flux.py`), unlike H3's conversion, which is why it adds $0 to
the LLM serverless line above — see [model-flux.md](model-flux.md) for why
Flux doesn't need H3's structured schema.
