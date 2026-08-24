---
name: runpod-session
description: Run a MiniMax H3 generation session on a RunPod pod safely. Use whenever the task involves starting, using, or shutting down the GPU pod — "spin up the pod", "generate some clips", "run inference on RunPod", "the pod is stuck", "how much did that cost". Encodes the money guardrails; skipping it is how a pod gets left billing overnight.
---

# RunPod session

Every expensive mistake on this platform is a quiet one. Follow the sequence.

## 1. Check capacity — never let RunPod queue for you

```bash
uv run videogen pod capacity
```

Refuses rather than queueing, deliberately. **Never configure an auto-deploy
reservation**: a request with no stock can become a standing order that starts
billing the moment capacity appears, including at 3am.

If it raises, stock genuinely is not there. Wait and retry. Do not widen the
datacenter list past `runtime.pod.LICENCE_SAFE_DATACENTERS` — that list is a
licensing constraint, not a preference. **MiniMax H3's licence excludes the US,
EU, UK, and South Korea.** Iceland and Norway are EEA but not EU; Romania,
Czechia, France, the Netherlands and Sweden are EU.

## 2. Deploy

```bash
uv run videogen pod up --template-id <official-comfyui-cuda13-template-id>
```

Template must be the **official RunPod ComfyUI, CUDA 13** one — the community
templates crowd it out in search. On CUDA 12.8 H3's quantised fast paths fall
back to a slow route.

`pod up` verifies host RAM and terminates immediately if the machine has less
than 60 GB. That is not paranoia: a 31 GB host crashed part-way through a second
consecutive generation. RAM is not selectable through the API, so redeploying
for a different machine is the only lever.

## 3. Wait for readiness — and know when to give up

ComfyUI copies itself into `/workspace` on first boot, so the proxy returns
**502 for roughly the first four minutes**. Expected. Ready when the log says:

```
[ComfyUI-Manager] All startup tasks have been completed.
```

```bash
uv run videogen pod status
```

- still not running after **5 min** → watch it
- still not running after **10 min** → **terminate and redeploy.** A machine
  that cannot pull its image is billing you while it fails.

## 4. Point the provider at it

```bash
export VIDEOGEN_COMFY_URL="https://<pod-id>-8188.proxy.runpod.net"
```

**The proxy severs any connection held open past ~100 seconds with a 524**
(Cloudflare). An H3 clip takes 2–6 minutes, so never hold a request open — the
provider queues and polls, which is why `ClipProvider` is
submit/poll/fetch/cancel rather than one `generate()`.

## 5. Generate

```bash
uv run videogen generate "<structured H3 prompt>" --provider comfyui \
    --workflow workflows/<yours>.json
```

Two things that decide whether the output is any good:

- **Use `videogen.prompts.h3` to build the prompt.** Free prose scored 26.0
  against 367.6 for the official structured schema, while a five-fold
  resolution increase changed nothing. A blurry result is a prompt problem.
- **The workflow must pass `validate_graph()`** — it runs automatically. See
  the turbo trap in `workflows/README.md`: the broken wiring runs ~4× *faster*
  and produces comb artifacts, so it looks like a win on a benchmark.

**Download outputs before shutting down.** They live on the pod's container
disk and go with it.

## 6. Shut down — terminate, not stop

```bash
uv run videogen pod down <pod-id>
```

**Stopping a pod does not stop billing.** It keeps the container disk and keeps
charging for it. `PodManager` deliberately exposes no `stop()`: every time it
would be reached for, terminate is the right answer.

Then confirm nothing is left running:

```bash
uv run videogen pod status      # "no pods. Nothing is billing."
```

## Cost reference

RTX 4090 Secure **$0.74/hr** (live-verified; the $0.69 in some write-ups is the
RTX 5090 *community* rate). A typical 2-hour session with 20 clips is ~$1.40.

Before anything that spends, `videogen generate` prints an estimate and refuses
above `VIDEOGEN_MAX_COST_USD`. Reconcile afterwards with the RunPod MCP
`get-billing` tool.

Keep Auto-Pay **off** in the console and top up in small amounts — it is the
only thing standing between a bug and an unbounded bill.
