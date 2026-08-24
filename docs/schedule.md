# The service window

One pod, one window a day. The window exists because a session's fixed cost is
~20 minutes (boot, 51GB weight download, node install) while a clip is ~5
minutes — so opening a pod per request would spend 80% of the money on setup.

## The window

| | |
|---|---|
| Local | **11:00 – 13:00** Asia/Taipei |
| UTC | 03:00 – 05:00 |
| Length | 2.0 h |
| Capacity | ~100 usable minutes ÷ ~5 min = **~20 clips/day**, ~600/month |

At the intended volume of ~50 clips a month that is roughly 30x headroom, so the
window is sized by *how long someone is willing to wait*, not by throughput.

## The capacity ladder

Placement is not a preference; it is whatever is available in a licence-permitted
datacenter at 11:00. Four rungs, price **strictly descending**, and only the
cheapest one waits:

| # | GPU | VRAM | $/hr | 2h×30 | LoRA mode | on refusal |
|---|---|---|---|---|---|---|
| 1 | L40S Secure | 48 | $1.004 | $60.2 | **bypass** (sharpest) | next rung |
| 2 | L40S Community | 48 | $0.804 | $48.2 | **bypass** | next rung |
| 3 | RTX 4090 Secure | 24 | $0.754 | $45.2 | low_vram (softer) | next rung |
| 4 | RTX 4090 Community | 24 | $0.354 | $21.2 | low_vram (softer) | **retry and wait** |

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

### `LOW` means "probably none"

The catalogue reported `LOW` for both L40S and 4090 while every rung refused —
ten consecutive refusals in one earlier session, and all four rungs refusing on
another. On the run that finally succeeded, rungs 1 and 2 were both gone and it
landed on rung 3.

So the ladder is not defensive over-engineering; it is the normal path. And
critically: **waiting is done in our own retry loop, never by leaving a request
with RunPod.** A standing order that fills when capacity appears would start
billing unattended, including overnight.

## Three scheduled tasks

```bash
# 11:00 — open. Walks the ladder; raises having created nothing if all four fail.
videogen session open --until 13:00

# every 5 min — close early if the window has gone quiet
videogen session reap --idle-minutes 20

# 13:00 — close, unconditionally. Idempotent and safe when nothing is up.
videogen session close
```

`session open` also passes `--terminate-after` set to window end + 10 minutes.
That is the backstop, not the mechanism: **if the scheduler never fires, if the
machine sleeps, if this code crashes — the pod still terminates itself.** The
buffer covers a clip mid-render at the bell.

The reaper matters because a window sized for peak demand is mostly idle at ~50
clips a month, and idle minutes cost exactly what working ones do.

### Windows Task Scheduler

```powershell
$repo = "C:\Users\USER\Desktop\Develop\video-gen"
$uv   = (Get-Command uv).Source

schtasks /Create /TN "videogen-open"  /SC DAILY /ST 11:00 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run videogen session open --until 13:00"

schtasks /Create /TN "videogen-reap"  /SC MINUTE /MO 5 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run videogen session reap"

schtasks /Create /TN "videogen-close" /SC DAILY /ST 13:00 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run videogen session close"
```

⚠️ Task Scheduler does not run while the machine is asleep. `--terminate-after`
guarantees a pod gets *closed*, never that one gets *opened* — so if the machine
is unreliable, put the scheduler on whatever host serves the LINE webhook.

### cron, if the scheduler lives on the webhook host

```cron
# UTC. 11:00 Asia/Taipei = 03:00 UTC.
0  3 * * *  cd /srv/video-gen && uv run videogen session open --until 13:00 --tz Asia/Taipei
*/5 * * * * cd /srv/video-gen && uv run videogen session reap
0  5 * * *  cd /srv/video-gen && uv run videogen session close
```

## Checking on it

```bash
videogen session status     # rung, rate, elapsed, spent so far, when it closes
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
