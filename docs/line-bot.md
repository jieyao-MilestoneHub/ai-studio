# The LINE bot

Someone says `生成 一隻橘貓走在雨中` in a group; a link to the finished clip
appears on a status page they can open any time.

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
| MiniMax H3 | RunPod GPU pod | 11:00–13:00 Asia/Taipei |

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

### 4. Serve

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

### A small VPS

Anything with 1 vCPU and 512MB is ample; the process is a web server, a SQLite
file and ~50MB of clips.

```bash
uv sync --extra line
uv run videogen line serve --host 0.0.0.0 --port 8000
```

Put nginx or Caddy in front for TLS, or run cloudflared on the box. Keep it up
with systemd:

```ini
[Unit]
Description=videogen
After=network-online.target

[Service]
WorkingDirectory=/srv/video-gen
ExecStart=/usr/local/bin/uv run videogen line serve --port 8000
Restart=always
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
```

Put the window scheduler on this host too — see
[schedule.md](schedule.md) — so it does not depend on a laptop being awake.

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

## Cost

| | per month |
|---|---|
| GPU pod, 2h/day (depends on which ladder rung serves) | $21–60, ~$48 expected |
| VPS | $4–6 |
| LLM serverless, ~50 conversions | ~$0.10 |
| **LINE messages** | **$0** — replies only |
| **Object storage** | **$0** — ~50MB on the host |

Which rung served a run is recorded in `job.gpu_tier` and shown on the status
page, because rungs 3 and 4 have 24GB and must run the softer `low_vram` LoRA
mode. Falling down the ladder is a quality change, not only a price change.
