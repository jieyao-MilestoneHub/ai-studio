# The service window

One pod, one window a day. The window exists because a session's fixed cost is
~20 minutes (boot, 51GB weight download, node install) while a clip is ~5
minutes — so opening a pod per request would spend 80% of the money on setup.

Since the LINE bot's image trigger (`/圖片`, see [line-bot.md](line-bot.md))
shares this same pod, both model sets — H3 and Flux.1-dev — download on every
open. See [runpod.md](runpod.md) for the updated download math (~72–78GB
combined, up from H3's ~54.7GB alone).

## The window

| | |
|---|---|
| Local | **11:00 – 13:00** Asia/Taipei |
| UTC | 03:00 – 05:00 |
| Length | 2.0 h |
| Capacity | ~100 usable minutes ÷ ~5 min = **~20 clips/day**, ~600/month |

One FIFO queue serves both the video trigger (`/影片`, and `/圖影` for image-to-video) and the image
trigger (`/圖片`, see [line-bot.md](line-bot.md)) — `ai-studio worker`
dispatches each claimed job to the H3 or Flux provider by `media_kind`. An
image job's generation time is `[speculative]` but expected in the 15–40s
range, negligible against a clip's 2–6 minutes, so mixing images in barely
dents clip capacity. What it does cost is the extra download at window open
(above), and a checkpoint swap in ComfyUI whenever the queue alternates
kinds — not yet measured, worth watching if the queue interleaves heavily.

At the intended volume of ~50 clips a month that is roughly 30x headroom, so the
window is sized by *how long someone is willing to wait*, not by throughput.

## The capacity ladder

Placement is not a preference; it is whatever is available in a licence-permitted
datacenter at the moment the day's first request arrives. Four rungs, price **strictly descending**, and only the
cheapest one waits:

| # | GPU | where | VRAM | $/hr | 2h×30 | LoRA mode | on refusal |
|---|---|---|---|---|---|---|---|
| 1 | L40S Secure | `OC-AU-1` | 48 | $1.004 | $60.2 | **bypass** (sharpest) | next rung |
| 2 | L40S Community | `OC-AU-1` | 48 | $0.804 | $48.2 | **bypass** | next rung |
| 3 | RTX 4090 Secure | `EUR-IS-2` | 24 | $0.754 | $45.2 | low_vram (softer) | next rung |
| 4 | RTX 4090 Community | `EUR-IS-2` | 24 | $0.354 | $21.2 | low_vram (softer) | **retry and wait** |

**The datacenter column is load-bearing and it is not the same for every rung.**
L40S and the 4090 are not offered in the same places, and only some of those
places H3's licence permits.

Descending on purpose: take the best card available *now*, and if it is gone move
on immediately, because wall-clock inside a two-hour window is worth more than
the price difference. Only the cheapest rung is worth blocking on — what you
eventually get is the least expensive option, and a softer clip beats no clip.

**The quality line sits between rungs 2 and 3.** A measured run peaked at
**43.3GB** 📏, so anything under 48GB has to merge the LoRA instead of bypassing
it, which the node pack itself calls *softer on quantized bases*. Each rung's
quantisation follows from the same fact: fp8 (native on Ada and newer) at 48GB,
int8 at 24GB. Which rung served a run is recorded in `job.gpu_tier` and shown on
the status page — **falling down the ladder changes the output, not just the
bill.**

Expected monthly cost lands in **$21–60**, around $48, depending on which rung
answers each day.

### A refusal does not tell you why

This is the trap that already cost this project its best quality tier for weeks.

Every rung used to pin `EUR-IS-2`. 📏 Checked against `/catalog/gpus` on
2026-08-25, **L40S is not offered in Iceland at all** — its secure stock is in
EU-NL-1, OC-AU-1, US-NC-1, US-TX-3 and US-TX-4. So rungs 1 and 2 were asking for
a card that was never there, and RunPod refused them exactly the way it refuses
a datacenter that is merely empty. Read from the outside, "sold out" and "we
don't stock that here" are the same event.

The consequence was invisible and expensive in quality rather than money: the
window fell through to a 24GB rung **every single time**, so the sharpest
`bypass` path — the reason 48GB was chosen at all — was never once used, and the
logs looked like ordinary bad luck.

Two things now prevent a repeat:

```bash
ai-studio pod placement     # every rung vs the catalog; exits 1 on a dead rung
```

and a test (`test_pod_placement.py`) that fails if any rung's datacenter does not
offer its card. Both distinguish **`not-offered`** (never going to work) from
**`empty`** (try again later) from **`unverifiable`** (community cloud publishes
no per-datacenter breakdown, so we cannot know before trying).

### `LOW` still means "probably none"

Genuine scarcity is real on top of that: the catalogue reported `LOW` while ten
consecutive attempts refused in one session. `LOW` is not a promise.

So the ladder is not defensive over-engineering; it is the normal path. And
critically: **waiting is done in our own retry loop, never by leaving a request
with RunPod.** A standing order that fills when capacity appears would start
billing unattended, including overnight.

### One consequence for latency

`OC-AU-1` is Sydney; `EUR-IS-2` is Iceland. Both are fine for driving ComfyUI
over the proxy, and Sydney is materially closer to Taiwan than Iceland — but the
pod's own downloads (54.7GB of weights 📏) come from Hugging Face, so setup time
depends on the host's link rather than on distance from us.

## Who opens the pod

**The first request that arrives inside business hours does.** Not a timer.

The window is unchanged — 11:00–13:00 Asia/Taipei, the table above still
holds — and so is the reason for it: setup is ~20 minutes against ~5 for a
clip, so the pod is still opened at most once a day and the queue still
absorbs everything that arrives in between. What changed is the trigger. A
timer that fires at 03:00 UTC opens a pod whether or not anybody asked for
anything, and a `drain` timer that ticks every five minutes makes someone who
asked at 11:00:10 wait until 11:05 for no reason. "Instant" here means
*instant inside business hours*, and neither of those was.

`ai-studio worker` is the loop. It runs as a service, always:

```bash
ai-studio worker
```

- Outside 11:00–13:00 it sleeps 60s at a time and does nothing else. It does
  not open a pod, it does not drain, it does not fail.
- Inside the window it checks the queue every 10 seconds. A `parsed` job — not
  merely `queued`; an unconverted request may never become a valid prompt —
  calls `runtime.session.ensure_pod()`, which reuses the day's pod if one is
  live and otherwise creates it with a lease that runs to 13:00.
- Concurrency is 1. One pod, one ComfyUI, one model resident in VRAM.
- It stops claiming new work 8 minutes before 13:00 and only finishes what it
  holds. Starting a render at 12:58 buys four minutes of GPU that
  `--terminate-after` then throws away.

The single source of 11:00 and 13:00 is `runtime/hours.py`. It is in `runtime`
rather than in `session.py` because of the layer contract: `bots` (L6) writes
the out-of-hours reply and reaches *down* for it, while `pipeline` (L4) sits
below `runtime` and cannot import it at all — the worker takes those three
functions by protocol and `cli.main` injects them.

## Two timers, and both of them only close things

```bash
# every 5 min — close early if the pod has gone quiet
ai-studio session reap --idle-minutes 30

# 13:00 — close, unconditionally. Idempotent and safe when nothing is up.
ai-studio session close
```

Nothing on a schedule opens anything any more. That is the point: every
scheduled task left can only ever *reduce* what is billing.

`ensure_pod` passes `--terminate-after` set to the end of business hours plus
10 minutes. That is the backstop, not the mechanism: **if the worker dies, if
the reaper never fires, if this code crashes — the pod still terminates
itself.** The buffer covers a clip mid-render at the bell. Three independent
ways for a machine to stop billing, and only the last needs no process alive.

The reaper's idle window is **30 minutes**, and the number is now measured
rather than guessed. It was 10 for one evening (2026-08-26), on the reasoning
that a request-opened window should be only as long as the work needs. That
evening's cold open settled it the other way: creating the pod, pulling 68 GB
of weights and restarting ComfyUI took 📏 **~15 minutes and $0.18** on an RTX
4090, while a Flux image then took 📏 12 s — and the 10-minute reaper closed
the first pod of the night *before its first job*, so the whole cold open was
paid twice. Every reopen pays it again, so the grace has to be longer than the
cold open, not shorter. Thirty is the cold open plus the gap between two
messages in a group chat. The pod's own `--terminate-after` and the 13:05
`close` timer still bound the worst case.

### The network volume, and why the cold open stopped mattering

With `AI_STUDIO_NETWORK_VOLUME_ID` set, every window pod mounts that volume
at `/workspace` and `deploy/pod_setup.sh` finds the weights already there, so
a cold open is a ComfyUI restart, not a 68 GB download. The price is the
placement: network volumes are secure-cloud only and mount only in their own
datacenter, so the ladder collapses to the 4090 secure rung in that
datacenter (`runtime.session.candidates_for_volume`), with `wait=True`
because waiting for stock there is cheaper than a full download elsewhere.
Iceland's `EUR-IS-2`, where the 4090 stock usually is, does not offer
volumes; `EUR-IS-1` does, and is equally licence-safe. 100 GB there is
$7/month `[reported]`, against $0.18 and fifteen minutes per open without it.

### What `ensure_pod` checks before it creates anything

Three gates, cheapest first, all of them before a pod exists:

1. **Business hours.** Outside them it raises `OutsideBusinessHours` — its own
   type, because the caller's answer to it is specific and cheap: leave the
   request in the queue for tomorrow. A generic failure would be
   indistinguishable from "the ladder is empty", which *is* worth retrying.
2. **Opens per day.** `AI_STUDIO_MAX_POD_OPENS_PER_DAY` (default 2: the day's
   window, plus one more if the reaper closed it and a later request needs the
   shop reopened), counted against `pod_opens` in the queue database on the
   Asia/Taipei day. This is the failure the monthly guard cannot see — a
   worker that crash-loops opens a fresh pod on every restart, and every one
   of them is individually inside budget.
3. **Monthly budget guard.** `runtime.budget.MonthlyBudgetGuard` reads
   `AI_STUDIO_MAX_MONTH_USD` (default $50) and `AI_STUDIO_VPS_MONTHLY_USD`
   (default $5, reserved off the top) against a running ledger
   (`runs/.spend_ledger.json`, rolled over on the Asia/Taipei calendar month).
   If what's left this month can't cover even a ~20-minute session at the
   ladder's *priciest* rung, it refuses outright. If there's *some* budget but
   not enough for the full window at the worst-case rate, the guard shrinks
   the lease instead of refusing, so a few expensive early-month days degrade
   the window length gracefully rather than blow the cap on day three.

Same guard and same pessimistic arithmetic as before; it has moved from the
CLI's `session open` onto the path that actually creates pods. The old demand
gate ("skip if the queue is empty") is gone because it no longer has anything
to guard: nothing opens a pod except a request.

This exists because the ladder's own worst case already exceeds $50/month on
GPU alone — rung 1 at $1.004/hr × 2h × 30d is $60.24 — so "$50/month" was a
target, not an enforced number, until this guard. It is intentionally
pessimistic: the guard checks against the *most expensive* rung because which
rung actually answers isn't known until after `open_session()` has already
created the pod (and set `--terminate-after`), so refusing or throttling has
to happen before that, on the worst case. A cheaper rung answering just means
the real month comes in under budget, never over it.

**The ledger is fed from `close_session()` itself, not from the CLI's
`session close` command.** At the intended ~50 renders/month, the pod is
almost always closed early by `reap`'s `close_if_idle()` — "~30x headroom"
above — long before the scheduled 13:00 `session close` ever runs, so by the
time that command fires, the state file is already gone and there is nothing
left for it to record. Recording inside `close_session()` means every path
that actually ends a window — the idle reaper, the past-window check, and the
explicit scheduled close — books its cost exactly once, in one place,
regardless of which one happened to fire.

### Windows Task Scheduler

```powershell
$repo = "C:\Users\USER\Desktop\Develop\ai-studio"
$uv   = (Get-Command uv).Source

schtasks /Create /TN "ai-studio-reap"  /SC MINUTE /MO 5 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run ai-studio session reap --idle-minutes 30"

schtasks /Create /TN "ai-studio-close" /SC DAILY /ST 13:00 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run ai-studio session close"
```

`ai-studio worker` is not a scheduled task — it is a service that stays up. On
Windows that means a console you leave running, or NSSM; on the VPS it is
`ai-studio-worker.service` with `Restart=always` (see
[`deploy/vps_setup.sh`](../deploy/vps_setup.sh)); on the Jetson it is the same
unit written by [`deploy/jetson_setup.sh`](../deploy/jetson_setup.sh), which
also masks the systemd sleep targets so the box cannot suspend under it.

⚠️ Task Scheduler does not run while the machine is asleep, and neither does
the worker. `--terminate-after` guarantees a pod gets *closed*, never that one
gets *opened* — so if the machine is unreliable, put the worker and the timers
on whatever host serves the LINE webhook.

### cron, if these live on the webhook host

```cron
# UTC. 13:00 Asia/Taipei = 05:00 UTC. Nothing here opens a pod.
*/5 * * * * cd /srv/ai-studio && uv run ai-studio session reap --idle-minutes 30
0  5 * * *  cd /srv/ai-studio && uv run ai-studio session close
```

## Checking on it

```bash
ai-studio session status     # rung, rate, elapsed, spent so far, when it closes
```

With no state file it also lists any running pods, because the failure that costs
money is a pod nobody is tracking.

## The disk line on the bill is real

Rates above include **$0.014/hr** for our 80GB volume plus 20GB container disk.
That is measured twice, not estimated: a pod listed at $0.44/hr reported
`currentSpendPerHr: 0.454`, and one listed at $0.74 reported `0.754`. 📏

Reconcile a session against the **balance delta** (`runpodctl user`), not the
billing API — the latter lagged badly in testing and once reported pod ids that
did not match the pods that had actually run.

## Licence

MiniMax H3's licence excludes the US, EU, UK and South Korea, so every rung sits
in Iceland. Iceland and Norway are EEA but **not** EU; Romania, Czechia, France,
the Netherlands and Sweden are. The permitted list is
`runtime.pod.LICENCE_SAFE_DATACENTERS`, and a test asserts every rung is in it.
