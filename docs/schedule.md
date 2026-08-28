# The pod's lifecycle

One pod, opened by the first request and closed minutes after the last
render. There is no daily window and no business hours (decision
2026-08-27): with the model weights on a network volume a cold open is a
ComfyUI restart (~1 minute), so there is nothing left to amortise by
batching requests into a fixed slot. What protects money instead is the
monthly budget guard, the daily open cap and the three closers described
below.

Since the image trigger (`/圖片`, see [line-bot.md](../fun_workflow/docs/line-bot.md))
shares this same pod, both model sets — H3 and Flux.1-dev — live on the same
volume. See [runpod.md](runpod.md) for the download math (~72–78GB combined,
up from H3's ~54.7GB alone; a one-time cost per volume, not per open).

One FIFO queue serves both the video trigger (`/影片`, and `/圖影` for image-to-video) and the image
trigger (`/圖片`, see [line-bot.md](../fun_workflow/docs/line-bot.md)) — `funapp worker`
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

**The first request does, at any hour.** Not a timer, and not a window.

There are no business hours any more (2026-08-27). The 11:00–13:00 window
existed to amortise a ~15-minute, 68 GB cold open; with the weights on a
network volume a cold open is a ComfyUI restart, so there is nothing left to
amortise and the fixed window only ever cost idle GPU-minutes and made
people wait until eleven. What protects money now is the monthly budget
guard, the per-day open cap (15, since pods are short-lived), and the three
closers below.

`funapp worker` (in `fun_workflow/`) is the loop. It runs as a service, always:

```bash
uv run --project fun_workflow funapp worker
```

- With nothing queued it checks every 10 seconds and does nothing else.
- The moment anything is *queued* — even before the LLM conversion has
  finished — it calls `runtime.session.ensure_pod()`, so the pod's cold start
  overlaps the conversion instead of following it. Only `parsed` work is
  claimed: an unconverted request may never become a valid prompt, and one
  never reaches a GPU.
- A fresh pod is provisioned by the worker itself: `deploy/pod_setup.sh` is
  copied up over SSH and started detached (`runtime.session.provision`), and
  the worker waits for the H3 node pack to appear before submitting. On a
  provisioned volume the script's fast path makes that about a minute.
- Concurrency is 1. One pod, one ComfyUI, one model resident in VRAM.
- The lease is `LEASE_HOURS` (2) from opening; it stops claiming new work 8
  minutes before the lease ends. The reaper closes long before that.

The single source of the lease and the day boundary is `runtime/hours.py`. It is in `runtime`
rather than in `session.py` because of the layer contract: `bots` (L6) writes
the out-of-hours reply and reaches *down* for it, while `pipeline` (L4) sits
below `runtime` and cannot import it at all — the worker takes those three
functions by protocol and `cli.main` injects them.

## Four timers, and none of them can open a pod

| timer | `OnCalendar` | does |
|---|---|---|
| `ai-studio-reap` | every minute | `funapp reap` — close a quiet pod after its per-kind grace; never with work queued. Logs to journald only on a transition, DEBUG to JSONL each minute |
| `ai-studio-gc` | `02:30 Asia/Taipei` | `gc` — sweep delivered media and received photos past `AI_STUDIO_FILES_RETENTION_DAYS` |
| `ai-studio-archive` | `03:00 Asia/Taipei` | `archive` — snapshot the queue db, tar+zstd the JSONL traces / session and pod records / drama state / ledger, verify, then prune (docs/observability.md) |
| `ai-studio-close` | `04:05 Asia/Taipei` | `session close` — the daily hard close, a backstop behind the reaper and `--terminate-after` |

The zone is spelled out in every `OnCalendar`: a bare `20:05` is read in the
box's local zone, and on 2026-08-27 that terminated a pod 46 minutes into a
render (pod `i5s1j69xkcihnn`). Nothing on a schedule opens anything. That is
the point: every scheduled task can only ever *reduce* what is billing, or
compress what already happened.

`ensure_pod` passes `--terminate-after` set to the lease end (`now +
LEASE_HOURS`) plus 10 minutes. That is the backstop, not the mechanism: **if
the worker dies, if the reaper never fires, if this code crashes — the pod
still terminates itself.** The buffer covers a clip mid-render. Three
independent ways for a machine to stop billing, and only the last needs no
process alive.

**The reaper's grace is whatever the last render recorded, and it never
closes a pod with work waiting.** `funapp reap` runs every minute. The pod
runtime keeps only the clock: `touch_activity(label, grace_minutes=)` writes
the caller's number into the session state and `close_if_idle` reads it
back (`DEFAULT_GRACE_MINUTES`, 10, if nothing was recorded). The table of
what each kind earns is fun_workflow's `pipeline/idle.py`: image 5, video
10, the three understanding kinds 5, drama 10 (and `pipeline.drama` touches
activity after every still and clip, so a half-hour drama never looks
idle), chat 15 — the longest, because a chat conversation pauses and
resumes in a way a render request does not, and it is sized against the
reopen fixed cost rather than any observed pause. The numbers differ because
the reloads do: 📏 Flux comes back into VRAM in ~15 s, H3's 32B text encoder
in 60–90 s, so a video pod is worth holding longer. The understanding grace
is `[speculative]` in a stronger sense than the others — nothing has
measured a lazy-load cost for any of the three understanding models yet,
and they are not even the same size as each other, so one shared number is
a starting point, not three considered answers.
Tuned by how often the reaper log shows a pod closed and reopened within a
few minutes. And a pod with anything queued is *held*, whatever the clock
says: closing a pod a job is about to land on costs a cold open **and** the
wait, the one move with no upside. This replaces the fixed 30-minute window,
which in turn replaced a one-evening 10 that closed the first pod of the
night before its first job — with the weights on a network volume a reopen
is a ~1-minute ComfyUI restart, not the 📏 15-minute, $0.18 download it was,
so the grace can be short again.

### The GPU hand-off between ComfyUI and the inference server

Idle grace is about *closing* the pod; this is about what happens *while it
stays open* and the queue alternates between a generation job and an
understanding one. Both share the one 24GB card, and neither H3, Flux, nor
any of the three understanding models, nor `/himonkey`'s gpt-oss-20b,
comfortably coexists with another model resident at the same time — so
exactly one of {ComfyUI's checkpoint, one of the inference server's four
backends} may be loaded at once, never both. Chat and the three
understanding kinds are the *same* side of this split: they share one
`ModelSlot` in `deploy/inference_server.py`, and `make_room_for()` groups
`MediaKind.CHAT` with them.

The hand-off is **pull-based**, not each side pinging the other:
`pipeline.drain.make_room_for()` runs in the one place that already knows
which job is about to run (`drain_window`'s and `worker._run_one`'s
dispatch point) and evicts *the other side's* model before every submit —
`ComfyClient.free_memory()` (`POST /free` on ComfyUI, port 8188) before a
generation job, `InferenceClient.unload()` (`POST /unload` on
`deploy/inference_server.py`, port 8189) before an understanding or chat
one.
Centralizing this where the dispatch decision is already made means neither
provider needs to know the other's endpoint exists.

This is a real cost this project has not measured: every kind-switch in the
queue now pays a model-load, not just the first job of the window. A queue
that alternates `/影片`, `/說圖`, `/影片`, `/說圖` pays four loads where a
queue of four `/影片`s pays one. Watch the reaper/worker log for how often
this actually happens in practice before assuming it is negligible.

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
   shop reopened), counted in `runs/.pod_opens.json` (`runtime.opens.PodOpenLedger`) on
   the Asia/Taipei day. This is the failure the monthly guard cannot see — a
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
  /TR "cmd /c cd /d $repo && `"$uv`" run ai-studio session reap"

schtasks /Create /TN "ai-studio-close" /SC DAILY /ST 13:00 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run ai-studio session close"
```

`funapp worker` is not a scheduled task — it is a service that stays up. On
Windows that means a console you leave running, or NSSM; on the VPS it is
`ai-studio-worker.service` with `Restart=always` (see
[`fun_workflow/deploy/vps_setup.sh`](../fun_workflow/deploy/vps_setup.sh)); on
the Jetson it is the same unit written by
[`fun_workflow/deploy/jetson_setup.sh`](../fun_workflow/deploy/jetson_setup.sh),
which also masks the systemd sleep targets so the box cannot suspend under it.

⚠️ Task Scheduler does not run while the machine is asleep, and neither does
the worker. `--terminate-after` guarantees a pod gets *closed*, never that one
gets *opened* — so if the machine is unreliable, put the worker and the timers
on whatever host serves the LINE webhook.

### cron, if these live on the webhook host

```cron
# UTC. 13:00 Asia/Taipei = 05:00 UTC. Nothing here opens a pod.
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

⚠️ **This figure predates the three understanding models.** With moondream3
+ Qwen3-Omni-Captioner (downloaded at full precision, quantized on load) +
Tarsier2 added, the volume needs closer to 165GB total (H3 + Flux + the
three understanding models, per `deploy/pod_setup.sh`'s own disk-headroom
check) rather than 80GB, and container disk should be sized generously
(≥100GB, per the user's own recommendation) even when using a network
volume for the weights themselves. Re-measure the disk-hours line once a
real pod has run with this feature; do not carry the $0.014/hr figure
forward unexamined.

Reconcile a session against the **balance delta** (`runpodctl user`), not the
billing API — the latter lagged badly in testing and once reported pod ids that
did not match the pods that had actually run.

## Licence

MiniMax H3's licence excludes the US, EU, UK and South Korea, so every rung sits
in Iceland. Iceland and Norway are EEA but **not** EU; Romania, Czechia, France,
the Netherlands and Sweden are. The permitted list is
`runtime.pod.LICENCE_SAFE_DATACENTERS`, and a test asserts every rung is in it.
