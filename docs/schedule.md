# The service window

One pod, one window a day. The window exists because a session's fixed cost is
~20 minutes (boot, 51GB weight download, node install) while a clip is ~5
minutes — so opening a pod per request would spend 80% of the money on setup.

Since the LINE bot's image trigger (`畫圖`/`/img`, see [line-bot.md](line-bot.md))
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

One FIFO queue serves both the video trigger (`生成`/`/gen`) and the image
trigger (`畫圖`/`/img`, see [line-bot.md](line-bot.md)) — `session drain`
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
datacenter at 11:00. Four rungs, price **strictly descending**, and only the
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

## Three scheduled tasks

```bash
# 11:00 — open. Walks the ladder; raises having created nothing if all four fail.
ai-studio session open --until 13:00

# every 5 min — close early if the window has gone quiet
ai-studio session reap --idle-minutes 20

# 13:00 — close, unconditionally. Idempotent and safe when nothing is up.
ai-studio session close
```

`session open` also passes `--terminate-after` set to window end + 10 minutes.
That is the backstop, not the mechanism: **if the scheduler never fires, if the
machine sleeps, if this code crashes — the pod still terminates itself.** The
buffer covers a clip mid-render at the bell.

The reaper matters because a window sized for peak demand is mostly idle at ~50
clips a month, and idle minutes cost exactly what working ones do.

### The timer fires blindly; `session open` decides whether to spend

The systemd timer (or Task Scheduler entry) that calls `session open` at 11:00
is deliberately dumb and unconditional — it fires every day, windows or not.
Two checks live inside the command instead, so "the LINE bot triggers spend"
and "the pod only ever boots 11:00–13:00" are both true at once rather than in
tension:

1. **Demand gate.** Before anything else, `session open` checks whether the
   queue has any `queued`/`parsed` job at all — video or image. An empty queue
   means skip entirely: no capacity check, no pod, no spend that day. A day
   with zero LINE messages now costs $0, not one ladder rung's worth of an
   empty window.
2. **Monthly budget guard.** `runtime.budget.MonthlyBudgetGuard` reads
   `AI_STUDIO_MAX_MONTH_USD` (default $50) and `AI_STUDIO_VPS_MONTHLY_USD`
   (default $5, reserved off the top) against a running ledger
   (`runs/.spend_ledger.json`, rolled over on the Asia/Taipei calendar month).
   If what's left this month can't cover even a ~20-minute session at the
   ladder's *priciest* rung, `session open` refuses outright — the same
   `PodError`-style loud failure as an empty ladder, logged rather than
   silently skipped. If there's *some* budget but not enough for the full
   window at the worst-case rate, the guard shrinks `--until` instead of
   refusing, so a few expensive early-month days degrade the window length
   gracefully rather than blow the cap on day three.

This exists because the ladder's own worst case already exceeds $50/month on
GPU alone — rung 1 at $1.004/hr × 2h × 30d is $60.24 — so "$50/month" was a
target, not an enforced number, until this guard. It is intentionally
pessimistic: the guard checks against the *most expensive* rung because which
rung actually answers isn't known until after `open_session()` has already
created the pod (and set `--terminate-after`), so refusing or throttling has
to happen before that, on the worst case. A cheaper rung answering just means
the real month comes in under budget, never over it.

**The ledger is fed from `close_session()` itself, not from the CLI's
`session close` command.** At the intended ~50 renders/month, the window is
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

schtasks /Create /TN "ai-studio-open"  /SC DAILY /ST 11:00 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run ai-studio session open --until 13:00"

schtasks /Create /TN "ai-studio-reap"  /SC MINUTE /MO 5 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run ai-studio session reap"

schtasks /Create /TN "ai-studio-close" /SC DAILY /ST 13:00 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run ai-studio session close"
```

⚠️ Task Scheduler does not run while the machine is asleep. `--terminate-after`
guarantees a pod gets *closed*, never that one gets *opened* — so if the machine
is unreliable, put the scheduler on whatever host serves the LINE webhook.

### cron, if the scheduler lives on the webhook host

```cron
# UTC. 11:00 Asia/Taipei = 03:00 UTC.
0  3 * * *  cd /srv/ai-studio && uv run ai-studio session open --until 13:00 --tz Asia/Taipei
*/5 * * * * cd /srv/ai-studio && uv run ai-studio session reap
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
