# Running on RunPod

## Topology

```
┌─ Pod (persistent) ──────────────┐        ┌─ ComfyUI (same pod, port 8188) ─┐
│  pipeline orchestration          │ submit │  MiniMax H3                     │
│  ffmpeg assembly                 │ ─────▶ │  model loaded once              │
│  gates                           │        │  864x480 clips + native audio   │
│  (later) FastAPI + LINE          │ ◀───── │                                 │
└──────────────────────────────────┘  poll  └─────────────────────────────────┘
                 │
                 ▼
      object storage (R2) → public HTTPS + range requests → LINE
```

ffmpeg assembly runs on the pod, not on a GPU worker: it is CPU-bound (paying
GPU rates for x264 is waste), it needs the whole run directory co-located, it
runs for minutes, and it must stay debuggable on a laptop with no GPU.

## Before you deploy

```bash
uv run ai-studio pod capacity     # licence-safe DC with actual stock
```

Three things that will otherwise cost you money quietly:

1. **RTX 4090 stock is thin, and the licence narrows it further.** At the time
   of writing, 4090 capacity existed only in `EU-RO-1` (EU — excluded by the
   model licence) and `EUR-IS-2` (Iceland — allowed), both reported LOW. There
   was no 4090 stock in CA-MTL. Check, do not assume.

2. **Never configure an auto-deploy reservation.** Requesting a GPU with no
   stock can leave a standing order that starts billing the moment capacity
   appears — including at 3am while you are asleep. `pod capacity` raises
   instead of queueing, for exactly this reason.

3. **Top up deliberately.** The console defaults to a $150 top-up; choose
   "Other" and start at $10. **Leave Auto-Pay off** — it is the only thing
   standing between a bug and an unbounded bill.

## Deploy

```bash
uv run ai-studio pod up --template-id <official-comfyui-template-id>  # cw3nka7d08 for standard GPUs, see runtime.session.TEMPLATE_COMFYUI_STANDARD
uv run ai-studio pod status
```

`pod up` checks capacity, deploys into a licence-safe datacenter, and then
**verifies host RAM before accepting the machine**. If the allocated host has
less than 60 GB it terminates immediately and tells you to retry — RAM is not
selectable through the API, so redeploying for a different machine is the only
lever available.

Readiness: ComfyUI copies itself into `/workspace` on first boot, so the proxy
returns **502 for roughly the first four minutes**. That is expected. The log
line to wait for is:

```
[ComfyUI-Manager] All startup tasks have been completed.
```

If the pod is still not running after **5 minutes**, watch it. After **10**,
terminate and redeploy — a machine that cannot pull its image is billing you
while it fails. `pod status` prints both warnings.

Then point the provider at it:

```bash
export AI_STUDIO_COMFY_URL="https://<pod-id>-8188.proxy.runpod.net"
```

## The 100-second wall

Pod HTTP goes through `https://<pod-id>-<port>.proxy.runpod.net`, which sits
behind Cloudflare and **severs any connection held open past ~100 seconds with
a 524**. An H3 clip takes 2–6 minutes.

There is therefore no "just await the render". Everything queues and polls, and
that constraint is why `ClipProvider` is `submit` / `poll` / `fetch` / `cancel`
rather than one `generate()`.

For phase 2, front the pod with a **named Cloudflare Tunnel** rather than the
proxy hostname: LINE needs a stable HTTPS webhook URL with a valid certificate,
and the proxy hostname changes whenever the pod is recreated.

## Storage: no network volume for weights

A network volume bills **$0.07/GB/month whether or not you use it** — 100 GB is
$7/month, charged while the pod is terminated. Re-downloading the 42 GB `fl2va`
set takes about 9 minutes at datacenter speeds, which on a 4090 costs about
$0.11.

**$7/month against $0.11/download: it takes ~64 downloads to break even.** At
four or five sessions a month, re-downloading is roughly 60× cheaper. Use the
template's 200 GB container disk.

**This got more expensive to redownload once Flux.1-dev joined the same pod,
and the conclusion still doesn't flip.** A dual-trigger LINE bot (H3 video +
Flux image, see [line-bot.md](line-bot.md)) downloads both model sets on every
window open: H3's measured **54.7 GB** int8 working set plus Flux.1-dev's
`[speculative]` **~17–23 GB** (fp8 transformer + T5-XXL text encoder + CLIP-L +
VAE — see [model-flux.md](model-flux.md), unmeasured on this project's own
hardware) plus the NSFW LoRA's measured **0.69 GB** 📏. Combined, that's
**~72.7–78.7 GB** once a day, not 42 GB — call it
~$0.20/session at datacenter speeds instead of $0.11. A 150–200 GB network
volume sized to hold both model sets still bills **$10.50–$14/month whether or
not a window opens that day**, against roughly $6/month for 30 sessions of
redownloading both sets — the redownload is still the cheaper side by a wide
margin, and the session opens only once a day (not per LINE message), so the
volume's main selling point — skipping a redownload on every request — was
never actually on the table here. It also still pins the pod to one
datacenter, on top of the already-thin RTX 4090 pool.

**Request-driven pod opening does not change this either.** Since PLAN.md
Phase 1 the pod is created by the day's first LINE request rather than by a
03:00 UTC timer, which is easy to misread as "a pod per request" — it is not.
"Instant" means a pod opened by the request itself, at any hour, and
`ensure_pod` reuses the day's pod for every later request, with
`AI_STUDIO_MAX_POD_OPENS_PER_DAY` (default 2) as the hard backstop. So the
download still happens **at most once a day**, the arithmetic above is
unchanged, and the conclusion — no network volume — stands. What actually
changed is that a day with no LINE messages now downloads nothing at all,
which makes the redownload side cheaper still.

A volume also **pins the pod to one datacenter**, which shrinks the GPU pool —
and 4090 stock is already thin. RunPod's own golden path documents a live case
where a datacenter pin left workers throttled and a job queued for over four
minutes.

⚠️ **This conclusion does not survive the three understanding models
(`/說圖` `/說音` `/說影`).** moondream3 + Qwen3-Omni-Captioner (downloaded
full-precision, quantized on load — see
[model-qwen3-omni-captioner.md](model-qwen3-omni-captioner.md)) + Tarsier2
add **~99 GB** 📏 (moondream3 18.5 + Qwen3-Omni-Captioner 63.4 + Tarsier2
16.6, read off each repo's file list on 2026-08-27, after excluding the
spare checkpoint formats the loaders never open), and `/himonkey`'s
gpt-oss-20b **13.8 GB** 📏 more, bringing a from-scratch download to
**~182 GB** rather than the ~72.7–78.7 GB the arithmetic above was built
on — which is why the `ai-studio-weights` volume was grown from 100 to
200 GB ($14/month) on 2026-08-27. At that size the
break-even math above (redownload wins by ~60×) needs to be redone, not
assumed to still hold — a persistent network volume becomes the
better-justified default once these three models are in the mix, which is
why [schedule.md](schedule.md)'s "GPU hand-off" and "disk line on the bill"
sections now recommend one for any deployment using these commands. Treat
this section's "no network volume" conclusion as applying only to the
H3+Flux-only configuration it was measured against.

**Finished renders are a different question.** LINE requires the mp4 on a public
HTTPS host that supports HTTP range requests, ≤200 MB, plus a ≤1 MB poster at
matching aspect. That means object storage (R2 or equivalent) regardless. Model
weights on container disk, deliverables on R2 — no conflict.

> If you ever do want a volume, note RunPod exposes volumes over an
> S3-compatible API (`s3api-<DC>.runpod.io`, bucket = volume id), so the same
> `storage/s3.py` client covers R2, AWS, and a RunPod volume. S3 API keys for
> volumes are console-only — there is no CLI or REST call to mint one.

## Shut down

```bash
uv run ai-studio pod down <pod-id>
```

**This terminates.** Stopping a pod does not stop billing — it keeps the
container disk and keeps charging for it, which over a few days is real money.
`PodManager` deliberately exposes no `stop()`: every time it would be reached
for, terminate is the right answer.

Download your outputs first. They live on the pod's container disk and go with
it.

## What a session costs

RTX 4090 Secure at **$0.74/hr** (verified against the live catalogue — the
$0.69 quoted in some write-ups is the RTX 5090 *community* rate; community 4090
is $0.34/hr but runs on third-party consumer hardware that can be pre-empted
without warning):

| step | time | cost |
|---|---|---|
| boot + download 42 GB | ~12 min | $0.15 |
| 20 × 5s clips, turbo-12 | ~100 min | $1.23 |
| collect + terminate | ~3 min | $0.04 |
| **total** | **~2 h** | **~$1.40** |
