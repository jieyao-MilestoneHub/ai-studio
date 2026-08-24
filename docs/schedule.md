# The service window

One pod, one window a day. The window exists because a session's fixed cost is
~20 minutes (boot, 51GB weight download, node install) while a clip is ~5
minutes — so opening a pod per request would spend 80% of the money on setup.

## The window

| | |
|---|---|
| Local | **11:00 – 14:48** Asia/Taipei |
| UTC | 03:00 – 06:48 |
| Length | 3.8 h |
| GPU | RTX 4090 24GB, Community, EUR-IS-2 |
| Rate | $0.34/hr GPU + ~$0.014/hr disk = **$0.354/hr** |
| Per day | $1.35 |
| **Per month** | **$40.4** |

$40.4 against a $50 ceiling leaves 19% of headroom, which is why the idle reaper
below is not optional — a 3.8h window at ~50 clips/month is mostly idle, and idle
minutes cost exactly what working ones do.

The disk figure is measured, not assumed: a pod whose GPU list price was
$0.44/hr reported `currentSpendPerHr: 0.454`, and that delta is our 80GB volume
plus 20GB container disk.

## Three scheduled tasks

```bash
# 11:00 — open. Walks a candidate ladder; raises rather than queueing if 4090
# stock is absent, because a request for absent capacity can leave a standing
# order that bills the moment capacity appears.
videogen session open --until 14:48

# every 5 min — close early if the window has gone quiet
videogen session reap --idle-minutes 20

# 14:48 — close, unconditionally. Idempotent and safe to run when nothing is up.
videogen session close
```

`session open` also passes `--terminate-after` set to window end + 10 minutes.
That is the backstop, not the mechanism: **if the scheduler never fires, if the
machine sleeps, if this code crashes — the pod still terminates itself.** The
buffer covers a clip that is mid-render at the bell.

### Windows Task Scheduler

```powershell
$repo = "C:\Users\USER\Desktop\Develop\video-gen"
$uv   = (Get-Command uv).Source

schtasks /Create /TN "videogen-open"  /SC DAILY /ST 11:00 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run videogen session open --until 14:48"

schtasks /Create /TN "videogen-reap"  /SC MINUTE /MO 5 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run videogen session reap"

schtasks /Create /TN "videogen-close" /SC DAILY /ST 14:48 /F `
  /TR "cmd /c cd /d $repo && `"$uv`" run videogen session close"
```

Verify with `schtasks /Query /TN videogen-open /V /FO LIST`, and remove with
`schtasks /Delete /TN videogen-open /F`.

⚠️ Task Scheduler does not run while the machine is asleep or off. The
`--terminate-after` backstop covers the case where `close` never fires, but
`open` simply will not happen — so if the machine is unreliable, the scheduler
belongs on whatever host serves the LINE webhook instead.

### cron (if the scheduler moves to a Linux host)

```cron
# UTC. 11:00 Asia/Taipei = 03:00 UTC.
0  3 * * *  cd /srv/video-gen && uv run videogen session open --until 14:48 --tz Asia/Taipei
*/5 * * * * cd /srv/video-gen && uv run videogen session reap
48 6 * * *  cd /srv/video-gen && uv run videogen session close
```

## Checking on it

```bash
videogen session status     # rate, elapsed, spent so far, when it closes
```

With no state file it also lists any running pods, because the failure that
costs money is a pod nobody is tracking.

## Capacity is the real risk, not cost

4090 stock in the one licence-safe datacenter that carries it is reported `LOW`,
and `LOW` has meant *none* in practice: a create was refused ten consecutive
times in one session. `session open` therefore walks a ladder —

1. RTX 4090 · EUR-IS-2 · Community — $0.34/hr
2. RTX 4090 · EUR-IS-2 · Secure — $0.74/hr
3. RTX 5090 · EUR-IS-1 · Secure — $0.99/hr
4. L40S · EUR-IS-2 · Community — $0.79/hr

— and raises if every rung fails, having created nothing.

**A window that fails to open is cheap. A pod nobody closed is not.** That
asymmetry is why the ladder refuses rather than waits.

Note the rungs are not equivalent: entries 1, 2 and 3 have under 48GB of VRAM,
and a measured run peaked at **43.3GB** in the turbo LoRA's `bypass` mode. Those
need `low_vram=True`, which the node pack documents as *softer on quantized
bases*. Only the L40S runs bypass mode unchanged. Falling down the ladder is a
quality change, not just a price change — `result.json` should record which rung
served the run.

## Licence

MiniMax H3's licence excludes the US, EU, UK and South Korea, so every rung sits
in Iceland or another permitted region. Iceland and Norway are EEA but **not**
EU; Romania, Czechia, France, the Netherlands and Sweden are. The permitted list
is `runtime.pod.LICENCE_SAFE_DATACENTERS`.
