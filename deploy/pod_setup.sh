#!/usr/bin/env bash
# Bring a fresh Runpod ComfyUI pod to "ready to generate MiniMax H3".
#
# Runs ON the pod. Invoke over SSH:
#   ssh <pod> 'bash -s' -- 48 < deploy/pod_setup.sh
# where the argument is the card's VRAM in GB, which decides the weight set.
#
# ─── Two traps this script exists to make impossible ────────────────────────
#
# 1. NEVER `git stash --include-untracked` in the ComfyUI directory.
#    The image's /start.sh sources $CU/.venv-cu128/bin/activate, and that venv
#    is *untracked*. Stashing it makes the container crash-loop every ~17
#    seconds — while billing — with:
#      /start.sh: line 205: .../.venv-cu128/bin/activate: No such file
#    Observed live. Use `git checkout -f` instead: it leaves untracked files be.
#
# 2. Use the container's own venv pip, not the system pip.
#    ComfyUI runs from .venv-cu128; installing requirements with system pip puts
#    them where ComfyUI cannot see them.
#
# 3. When driving this remotely, never `pkill -f 'hf download'` from inside an
#    ssh one-liner: the pattern matches the shell running that very command,
#    so it kills the session. Kill by pid, or put the script in a file first.
#
# A fourth trap lives on the *calling* side, not here: under Git Bash, MSYS
# rewrites a bare `/workspace` argument into `C:/Program Files/Git/workspace`,
# which also crash-loops the container. Export MSYS2_ARG_CONV_EXCL='*' before
# calling runpodctl by hand. (`runtime/session.py` is immune — subprocess with
# an argv list does not pass through the MSYS runtime.)

set -uo pipefail

VRAM_GB="${1:-48}"
CU=/workspace/runpod-slim/ComfyUI
M="$CU/models"
COMFY_TAG="v0.33.3"   # 0.26.2, which the image ships, has no native H3 support
TURBO_REPO="https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo"

log() { printf '[setup %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { printf '[setup %s] FATAL: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }

# ── weight set follows the card, not preference ────────────────────────────
# A measured run peaked at 43.3GB. At 48GB the LoRA can be applied in bypass
# mode (sharpest) and fp8 is computed natively on Ada and newer. At 24GB fp8
# would be emulated and bypass would not fit, so int8 plus a merged LoRA it is.
if [ "$VRAM_GB" -ge 48 ]; then
  DIT="diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
  QUANT="fp8"
else
  DIT="diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  QUANT="int8"
fi
log "card has ${VRAM_GB}GB -> ${QUANT} weights"

# ── 0. a network volume that was set up before needs only a restart ───────
# With AI_STUDIO_NETWORK_VOLUME_ID every pod mounts the same /workspace, so
# the upgrade, the node pack and the weights from the first run are already
# here. The marker is per weight set (int8 vs fp8) and per version of this
# script's provisioning steps: bump SETUP_VERSION when they change and the
# next pod re-provisions instead of trusting a stale volume. Bumped to 2 when
# the three understanding models were added, 3 when gpt-oss-20b joined the
# download set, 4 when Qwen2-Audio and Qwen2.5-VL replaced Qwen3-Omni-Captioner
# and Tarsier2, 5 when the FaceDetailer install moved out of this script into
# a host-shipped extension (step 7) -- extensions run on every open and are
# not gated by this marker, so they never need a bump.
SETUP_VERSION=5
MARKER="/workspace/.ai-studio-setup-v${SETUP_VERSION}-${QUANT}"
if [ -f "$MARKER" ] && [ -d "$CU/custom_nodes/ComfyUI-MiniMax-H3-Turbo" ]; then
  log "volume already provisioned ($MARKER); skipping download and install"
  FAST_PATH=1
else
  FAST_PATH=0
fi

# ── 1. wait for the image to finish copying itself into /workspace ─────────
log "waiting for ComfyUI to appear in /workspace (first boot copies it)"
for i in $(seq 1 60); do
  [ -d "$CU/custom_nodes" ] && { log "  present after $((i * 5))s"; break; }
  sleep 5
done
[ -d "$CU/custom_nodes" ] || die "ComfyUI never appeared at $CU"

if [ "$FAST_PATH" = 0 ]; then
# ── 2. upgrade ComfyUI FIRST, before the network is saturated ─────────────
# Learned the hard way: starting the 51GB download first left `git fetch`
# hanging behind it for over eight minutes, and the upgrade never happened.
# The upgrade is small, so do it while the link is still free.
cd "$CU" || die "cannot cd $CU"
[ -x .venv-cu128/bin/pip ] || die ".venv-cu128 is missing — see trap 1 in this file's header"

log "upgrading ComfyUI to $COMFY_TAG (its own venv, no stash)"
# --force: the image ships a baked-in v0.26.2 tag that does not match the
# object the remote has at that name, and a plain `fetch --tags` refuses to
# clobber a local tag pointing elsewhere -- exit 1, no other refs updated
# either. Observed live on runpod/comfyui:cuda12.8. We want whatever origin
# has, unconditionally, so force is correct here rather than a narrower fix.
git fetch --tags --force --quiet origin || die "git fetch failed"
git -c advice.detachedHead=false checkout --quiet -f "tags/$COMFY_TAG" \
  || die "checkout $COMFY_TAG failed"
log "  now $(git describe --tags)"
[ -f comfy_extras/nodes_minimax_h3.py ] || die "no native H3 nodes after upgrade"

./.venv-cu128/bin/pip install -q -r requirements.txt 2>&1 \
  | grep -viE 'warning|notice' | tail -3

log "installing the understanding-model stack (transformers/accelerate/bitsandbytes/...)"
# Reuses this venv's already-matching torch+CUDA build rather than a fresh
# one, per the decision to keep the ComfyUI template and layer this stack on
# top of it instead of switching base images (see docs/architecture.md).
# Unverified: whether ComfyUI's own pinned transformers version (if any)
# conflicts with what moondream3/Qwen3-Omni-Captioner/Tarsier2's
# trust_remote_code modeling code needs -- check `pip check` output on the
# first real deployment, before trusting this venv serves both processes.
# kernels==0.16.0: what transformers 5.16 needs to run gpt-oss-20b's native
# MXFP4. Without it the loader says "defaulting to dequantizing the model
# to bf16" -- 40GB, an OOM on a 24GB card (observed live 2026-08-27). The
# version window is transformers' own (0.16 <= v < 0.17).
./.venv-cu128/bin/pip install -q --upgrade transformers accelerate bitsandbytes \
  soundfile librosa pillow fastapi 'uvicorn[standard]' python-multipart \
  'kernels==0.16.0' 'qwen-vl-utils[decord]==0.0.8' 2>&1 \
  | grep -viE 'warning|notice' | tail -3
# python-multipart: FastAPI's File()/Form()/UploadFile support is an optional
# feature dependency, not pulled in by fastapi or uvicorn[standard] on their
# own -- without it, inference_server.py's `@app.post("/submit")` route (File/
# Form parameters) raises at import time, before the process ever binds a
# port. Confirmed by reproducing the identical `RuntimeError: Form data
# requires "python-multipart" to be installed` locally; this line was missing
# it before /himonkey's chat modality made the /submit route's Form/File
# fields something a test actually tried to import and exercise.

log "installing the turbo node pack"
cd custom_nodes || die "no custom_nodes"
rm -rf ComfyUI-MiniMax-H3-Turbo
git clone --depth 1 --quiet "$TURBO_REPO" || die "clone failed"
[ -f ComfyUI-MiniMax-H3-Turbo/__init__.py ] || die "turbo pack looks empty"

# ── 3. weights, in the background: this is the long pole ─────────────────
mkdir -p "$M"/{diffusion_models,text_encoders,vae,loras} /workspace/dl-logs
# hf_transfer is NOT preinstalled on the image, and on its huggingface_hub the
# old switch is a no-op: it warns "HF_HUB_ENABLE_HF_TRANSFER is not used anymore.
# Please use HF_XET_HIGH_PERFORMANCE instead". Measured consequence of getting
# this wrong: 5 MB/s instead of ~75 MB/s, which turns 51GB from 11 minutes into
# nearly three hours of billed pod time.
# --break-system-packages is required: the image's python is externally managed.
python3 -c 'import hf_transfer' 2>/dev/null ||
  pip install -q --break-system-packages hf_transfer 2>&1 | tail -1
python3 -c 'import hf_transfer' 2>/dev/null ||
  log 'WARNING: hf_transfer unavailable, downloads will run at ~5MB/s'
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/workspace/.hf
command -v hf >/dev/null || pip install -q --break-system-packages -U huggingface_hub

# ~192GB of weights against whatever /workspace actually has (52GB H3 + 17GB
# Flux + ~82GB for the three understanding models, moondream3 (48) +
# Qwen2-Audio (17) + Qwen2.5-VL (17), + ~41GB for the complete gpt-oss-20b
# repo -- whole repos, every checkpoint format they ship; see the `dl_repo`
# calls below). On a network volume `df` reports the
# whole cluster's free space, so this check only bites on a plain container
# disk. Caught here, loudly, before it is caught 30 minutes later as two
# silently-missing safetensors files: a download that dies mid-transfer
# because the disk filled exits same as a download that finished, and
# `pgrep` alone cannot tell those apart. Observed live: this volume can come
# back smaller than requested depending on how the pod was deployed.
# Counted against what is still to come, not the whole set: a re-run after a
# dropped SSH session (observed live -- the link died between "weights
# complete" and the rename step) finds most of it already on disk and little
# free, and must not refuse to finish what it started.
AVAIL_KB="$(df -k /workspace | awk 'NR==2{print $4}')"
HAVE_KB="$(du -sk "$M" 2>/dev/null | cut -f1)"
NEED_KB=$((192 * 1024 * 1024 - ${HAVE_KB:-0}))
[ "$AVAIL_KB" -ge "$NEED_KB" ] || die \
  "/workspace has $((AVAIL_KB / 1048576))GB free, need ~$((NEED_KB / 1048576))GB more headroom (~52GB H3 + ~17GB Flux + ~82GB understanding models + ~41GB gpt-oss-20b, $((${HAVE_KB:-0} / 1048576))GB already present)"

DL_PIDS=()
dl() {  # repo file destdir [already-present-path]
  # The optional 4th argument is the path the file is *renamed to* further
  # down (the Flux LoRA/UNet/VAE, whose in-repo names do not match what the
  # workflow JSON asks for). A re-run on a volume that already has the
  # renamed file must not fetch it again: 📏 2026-08-27 a re-provision on a
  # 281/300GB volume re-downloaded all three, hit "Disk quota exceeded", and
  # the FATAL that followed stopped the inference server from ever starting
  # -- with the files sitting right there under their final names.
  if [ -n "${4:-}" ] && [ -s "$4" ]; then
    log "  present, skipping download: $(basename "$4")"
    return 0
  fi
  nohup hf download "$1" "$2" --local-dir "$3" \
    > "/workspace/dl-logs/$(basename "$2").log" 2>&1 &
  DL_PIDS+=("$!:$(basename "$2")")
  log "  downloading $(basename "$2")"
}
dl_repo() {  # repo [exclude-glob ...] -- the whole repo, into the HF cache (HF_HOME)
  # No --local-dir: `transformers.from_pretrained(repo_id, ...)` in
  # inference_server.py looks the model up by repo id in HF_HOME's cache,
  # not in an arbitrary directory, so this has to populate that cache rather
  # than a flat directory the way `dl` does for ComfyUI's non-cache-aware
  # node loaders.
  #
  # Queued, not launched: whole repos download ONE AT A TIME, in a single
  # background chain started by `dl_repos_start` below, and without xet's
  # high-performance mode. Measured 2026-08-27: this container's cgroup caps
  # RAM at 62GB (memory.limit_in_bytes; `free` shows the host's 251GB and
  # lies), one repo download in high-performance mode alone sat at ~51GB
  # used, and four repos plus the eight single-file downloads in parallel
  # tripped the OOM killer twice (memory.events oom_kill 2) -- two `hf
  # download`s died with a bare "Killed", no ENOSPC, no traceback.
  #
  # Exclude globs are supported but unused: the repos are kept complete
  # (spare checkpoint formats included, ~57GB) by decision on 2026-08-27,
  # and the volume was sized to 300GB for it (250GB hit `Disk quota
  # exceeded` 📏 at 229GB of du, with ~28GB still to come plus xet scratch).
  printf '%s\n' "$*" >> /workspace/dl-logs/repos.list
  log "  queued $(basename "$1") (full repo, into HF_HOME cache, sequential${2:+, excluding: ${*:2}})"
}
dl_repos_start() {
  [ -s /workspace/dl-logs/repos.list ] || return 0
  : > /workspace/dl-logs/repos.failed
  cat > /workspace/dl-repos.sh <<'CHAIN'
rc=0
while read -r repo excludes; do
  [ -n "$repo" ] || continue
  # shellcheck disable=SC2086  -- the excludes are meant to word-split
  hf download "$repo" ${excludes:+--exclude $excludes} \
    > "/workspace/dl-logs/$(basename "$repo").log" 2>&1 \
    || { basename "$repo" >> /workspace/dl-logs/repos.failed; rc=1; }
done < /workspace/dl-logs/repos.list
exit $rc
CHAIN
  nohup env -u HF_XET_HIGH_PERFORMANCE bash /workspace/dl-repos.sh \
    > /workspace/dl-logs/repos.log 2>&1 &
  DL_PIDS+=("$!:repos")
  log "  downloading $(wc -l < /workspace/dl-logs/repos.list) repo(s) sequentially"
}
: > /workspace/dl-logs/repos.list
log "starting weight downloads (~52GB H3 + ~17GB Flux + ~82GB understanding models + ~41GB gpt-oss-20b)"
dl Comfy-Org/MiniMax-H3 "$DIT" "$M"
dl Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors "$M"
dl Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors "$M"
dl Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors "$M"
dl larryvrh/MiniMax-H3-Turbo-Lora minimax_h3_turbo_v4_step600_ema.safetensors "$M/loras"
# The Flux image LoRA. 687,476,088 bytes measured from the HF blobs API.
# Needs no trigger word: neither candidate's model card declares an
# instance_prompt -- this is an "unrestrain" adapter, not a concept LoRA.
dl Heartsync/Flux-NSFW-uncensored lora.safetensors "$M/loras" "$M/loras/flux_nsfw_uncensored_v1.safetensors"
# Flux.1-dev itself. black-forest-labs/FLUX.1-dev is the canonical source but
# is HF-gated (401 with no token, and this script has none configured).
# comfyanonymous's repackaging is the same fp8-scaled weights, ungated.
dl comfyanonymous/flux_dev_scaled_fp8_test flux_dev_fp8_scaled_diffusion_model.safetensors "$M/diffusion_models" "$M/diffusion_models/flux1-dev.safetensors"
dl comfyanonymous/flux_text_encoders clip_l.safetensors "$M/text_encoders"
dl comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors "$M/text_encoders"
# The Flux VAE (ae.safetensors) ships inside the same gated black-forest-labs
# repo. Comfy-Org/z_image re-hosts the byte-identical file ungated -- checked
# against several such repackagings, all serving the same file with no auth.
dl Comfy-Org/z_image split_files/vae/ae.safetensors "$M/vae" "$M/vae/ae.safetensors"

# The three understanding models , each a whole repo rather
# than one file -- see `dl_repo`'s comment. Qwen3-Omni-Captioner is
# downloaded at full precision and quantized to 4-bit on load by
# inference_server.py's bitsandbytes config, not pre-quantized here: this
# project has not confirmed a separately-published GGUF/AWQ build of this
# specific captioner variant exists, and quantizing at load from the
# standard safetensors repo needs no such artifact. That trades a larger
# one-time download (~60GB `[reported]` vs Q4's ~17-22GB *resident* size --
# the two numbers are not comparable, one is on-disk and one is in-VRAM) for
# not depending on an unverified third party's requantization.
dl_repo moondream/moondream3-preview
# Qwen2-Audio-7B-Instruct and Qwen2.5-VL-7B-Instruct replaced
# Qwen3-Omni-Captioner (does not fit 24GB on transformers 5) and Tarsier2
# (custom class pinned to transformers 4.47) on 2026-08-27 -- see
# docs/model-qwen3-omni-captioner.md and docs/model-tarsier2.md. Both fp16,
# ~17GB each, Apache-2.0, ungated.
dl_repo Qwen/Qwen2-Audio-7B-Instruct
dl_repo Qwen/Qwen2.5-VL-7B-Instruct
# gpt-oss-20b, the fourth backend of inference_server.py. Its
# repo is 41GB (📏), the sharded MXFP4 weights plus original/ and metal/, loaded by
# `GptOssChatBackend.load()` from HF_HOME. Staged here for the same reason
# the three above are: a `from_pretrained` that has to pull the weights from the
# Hub inside the first chat request's background thread would surface
# as a 10-minute silent stall (or an ENOSPC nobody sees) rather than as a
# failed setup step -- exactly what the headroom check above exists for.
dl_repo openai/gpt-oss-20b
dl_repos_start

# ── 4. wait for the weights, then restart ComfyUI ─────────────────────────
while pgrep -f 'hf download' >/dev/null; do
  log "  weights: $(du -sh "$M" 2>/dev/null | cut -f1), $(df -h /workspace | awk 'NR==2{print $4}') free"
  sleep 30
done
# `pgrep` above only proves every download process has *exited* -- not that
# it exited zero. Reap each one by the PID captured at launch and check its
# actual status; a process that dies from ENOSPC exits nonzero same as any
# other failure, and silently declaring victory here is exactly how the
# turbo LoRA custom node ends up loaded against a diffusion model that was
# never actually written to disk.
FAILED_DL=()
for entry in "${DL_PIDS[@]}"; do
  pid="${entry%%:*}"; name="${entry#*:}"
  if ! wait "$pid" 2>/dev/null; then
    if [ "$name" = repos ]; then
      while IFS= read -r failed_repo; do FAILED_DL+=("$failed_repo"); done \
        < /workspace/dl-logs/repos.failed
    else
      FAILED_DL+=("$name")
    fi
  fi
done
[ "${#FAILED_DL[@]}" -eq 0 ] || die \
  "download(s) failed: ${FAILED_DL[*]} -- see /workspace/dl-logs/<name>.log ($(df -h /workspace | awk 'NR==2{print $4}') free)"
fi  # FAST_PATH
log "weights complete: $(du -sh "$M" | cut -f1)"
# hf download keeps the remote filename, and "lora.safetensors" says nothing
# in a directory that also holds the H3 turbo LoRA -- and does not match the
# lora_name in workflows/flux_dev.json. Renaming here is what makes those two
# strings the same string. Fail loudly rather than leaving the graph pointing
# at a file that is not there: ComfyUI's own error for a missing LoRA is a
# line in a log nobody is reading at 11:04.
FLUX_LORA="$M/loras/flux_nsfw_uncensored_v1.safetensors"
if [ -f "$M/loras/lora.safetensors" ]; then
  mv "$M/loras/lora.safetensors" "$FLUX_LORA"
fi
[ -f "$FLUX_LORA" ] || die "flux LoRA missing: $FLUX_LORA (check dl-logs/lora.safetensors.log)"

# Same reasoning, two more files: the repackaged repos keep their own names
# and their own in-repo layout, neither of which matches what UNETLoader and
# VAELoader in workflows/flux_dev.json actually ask for.
FLUX_UNET="$M/diffusion_models/flux1-dev.safetensors"
if [ -f "$M/diffusion_models/flux_dev_fp8_scaled_diffusion_model.safetensors" ]; then
  mv "$M/diffusion_models/flux_dev_fp8_scaled_diffusion_model.safetensors" "$FLUX_UNET"
fi
[ -f "$FLUX_UNET" ] || die "flux unet missing: $FLUX_UNET (check dl-logs/flux_dev_fp8_scaled_diffusion_model.safetensors.log)"

FLUX_VAE="$M/vae/ae.safetensors"
if [ -f "$M/vae/split_files/vae/ae.safetensors" ]; then
  mv "$M/vae/split_files/vae/ae.safetensors" "$FLUX_VAE"
  rm -rf "$M/vae/split_files"
fi
[ -f "$FLUX_VAE" ] || die "flux vae missing: $FLUX_VAE (check dl-logs/ae.safetensors.log)"

[ -f "$M/text_encoders/clip_l.safetensors" ] || die "flux clip_l missing"
[ -f "$M/text_encoders/t5xxl_fp8_e4m3fn.safetensors" ] || die "flux t5xxl missing"

find "$M" \( -name '*minimax*.safetensors' -o -name 'flux1-dev.safetensors' \
  -o -name 'flux_nsfw_uncensored_v1.safetensors' -o -name 'ae.safetensors' \
  -o -name 'clip_l.safetensors' -o -name 't5xxl_fp8_e4m3fn.safetensors' \) \
  -printf '%s %p\n' \
  | awk '{printf "[setup]   %7.2f GB  %s\n", $1/1073741824, $2}'

log "restarting ComfyUI"
pkill -f 'main.py --listen'; sleep 3
PY="$CU/.venv-cu128/bin/python"
[ -x "$PY" ] || PY=$(command -v python3)
cd "$CU" || die "cannot cd $CU"
# Ask which flags exist rather than assuming. This script upgrades ComfyUI a
# few steps earlier, so the flag set is whatever that version supports -- and
# an unrecognised flag is not a warning, it is argparse exiting and no ComfyUI
# at all, discovered only after the weights have been paid for.
#
# --fast-disk matters more than it looks: ComfyUI does not release the 32B
# text encoder, and without it RSS climbs until the second model load hits
# swap. Same problem as the 31GB host crash, and a plausible cause of the
# "Using RAM pressure cache" line that puts the second clip timing in doubt.
HELP="$("$PY" main.py --help 2>&1 || true)"
EXTRA=""
for flag in --fast-disk --use-sage-attention; do
  case "$HELP" in
    *"$flag"*)
      # argparse recognising a flag and its dependency actually being
      # importable are two different questions. --use-sage-attention passed
      # this case on an image whose venv had no `sageattention` package, and
      # ComfyUI refused to start at all rather than warning and continuing --
      # discovered as an empty /object_info response with no other clue.
      case "$flag" in
        --use-sage-attention)
          "$PY" -c 'import sageattention' 2>/dev/null \
            && EXTRA="$EXTRA $flag" \
            || log "  --use-sage-attention supported but sageattention is not installed, skipping"
          ;;
        *) EXTRA="$EXTRA $flag" ;;
      esac
      ;;
    *) log "  $flag not supported by this ComfyUI, skipping" ;;
  esac
done
log "  launch flags:${EXTRA} --reserve-vram 0.7"

# --reserve-vram leaves headroom so a long clip does not OOM at the VAE stage.
# shellcheck disable=SC2086
nohup "$PY" main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
  $EXTRA --reserve-vram 0.7 > /workspace/comfy.log 2>&1 &

for i in $(seq 1 60); do
  sleep 5
  curl -sf -m 5 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 \
    && { log "ComfyUI ready after $((i * 5))s"; break; }
done

# ── 5. prove the nodes we need are actually registered ────────────────────
curl -s -m 30 http://127.0.0.1:8188/object_info | "$PY" -c '
import json, sys
info = json.load(sys.stdin)
need = ["MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler", "MiniMaxH3ImageToVideo"]
missing = [n for n in need if n not in info]
for n in need:
    print("[setup]   " + ("OK  " if n in info else "MISS") + " " + n)
sys.exit(1 if missing else 0)
' || die "required H3 nodes are not registered"

# ── 6. start the understanding server -- a second, separate process ───────
# deploy/inference_server.py is deposited at /workspace/inference_server.py
# by runtime.session.provision() *before* this script runs -- see that
# function's docstring for why it travels as a second file over the same
# one-file-at-a-time SSH transport rather than being embedded inline here.
# Started fresh every pod open (not gated by FAST_PATH): the process itself
# does not persist across a pod restart even when its pip packages and
# downloaded weights do.
# The MXFP4 Triton kernels that `kernels` fetches for gpt-oss-20b at load
# time. The server runs HF-offline, and `kernels` resolves its own version
# tags against the Hub -- a plain `hf download` of the repo is not enough
# ("Version 1 of 'kernels-community/gpt-oss-triton-kernels' is not
# available in the local cache", observed live 2026-08-27 even with the
# snapshot present). Let it fetch once, online, into the same HF_HOME.
log "caching gpt-oss MXFP4 kernels"
HF_HOME=/workspace/.hf "$PY" -c '
from kernels import get_kernel
get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)' 2>&1 | tail -1 \
  || die "could not cache kernels-community/gpt-oss-triton-kernels"

log "starting the understanding-model server on :8189"
pkill -f 'inference_server.py'; sleep 1
# The server inherits HF_HOME (weights) and HF_TOKEN (the gated Tarsier2
# repo) from this script's environment. Both are load-bearing -- observed
# live 2026-08-27 when it was restarted by hand without them: no HF_HOME
# re-downloaded 60GB of Qwen3-Omni into the 20GB container disk ("No space
# left on device"); no HF_TOKEN died probing Tarsier2's chat_template.jinja.
# NOT HF_HUB_OFFLINE=1, though it was tried: `kernels` 0.16 refuses to
# resolve gpt-oss's MXFP4 kernel version offline even when it is cached
# ("Version 1 of 'kernels-community/gpt-oss-triton-kernels' is not
# available in the local cache and Hugging Face Hub is in offline mode").
# If you restart this server by hand, restart it with the same three.
HF_HOME=/workspace/.hf \
  nohup "$PY" /workspace/inference_server.py > /workspace/inference.log 2>&1 &

for i in $(seq 1 30); do
  sleep 2
  curl -sf -m 5 http://127.0.0.1:8189/healthz >/dev/null 2>&1 \
    && { log "inference server ready after $((i * 2))s"; break; }
done
curl -sf -m 5 http://127.0.0.1:8189/healthz >/dev/null 2>&1 \
  || die "inference server did not answer /healthz -- check /workspace/inference.log"

# ── 7. host-shipped extensions: /workspace/pod_setup.d/*.sh -- BEST EFFORT ─
# Deposited by `runtime.session.provision(extras=...)` before this script
# runs: whatever the caller wants on the pod that this script has no
# business knowing about (a node pack for one workflow, a model for one
# feature). Run on every open, not gated by FAST_PATH or the marker, so each
# must be idempotent; and nothing in one may `die` -- a failed extension is
# logged and the pod stays usable for everything above. They inherit this
# script's paths through the environment.
export CU M PY EXTRA VRAM_GB QUANT
export -f log
for ext in /workspace/pod_setup.d/*.sh; do
  [ -f "$ext" ] || continue
  log "extension: $(basename "$ext")"
  bash "$ext" || log "  extension $(basename "$ext") failed (best effort); continuing"
done

touch "$MARKER"
log "done. quantisation=${QUANT} vram=${VRAM_GB}GB marker=$MARKER"
