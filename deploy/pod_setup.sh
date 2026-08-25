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

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] FATAL: %s\n' "$*" >&2; exit 1; }

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

# ── 1. wait for the image to finish copying itself into /workspace ─────────
log "waiting for ComfyUI to appear in /workspace (first boot copies it)"
for i in $(seq 1 60); do
  [ -d "$CU/custom_nodes" ] && { log "  present after $((i * 5))s"; break; }
  sleep 5
done
[ -d "$CU/custom_nodes" ] || die "ComfyUI never appeared at $CU"

# ── 2. upgrade ComfyUI FIRST, before the network is saturated ─────────────
# Learned the hard way: starting the 51GB download first left `git fetch`
# hanging behind it for over eight minutes, and the upgrade never happened.
# The upgrade is small, so do it while the link is still free.
cd "$CU" || die "cannot cd $CU"
[ -x .venv-cu128/bin/pip ] || die ".venv-cu128 is missing — see trap 1 in this file's header"

log "upgrading ComfyUI to $COMFY_TAG (its own venv, no stash)"
git fetch --tags --quiet origin || die "git fetch failed"
git -c advice.detachedHead=false checkout --quiet -f "tags/$COMFY_TAG" \
  || die "checkout $COMFY_TAG failed"
log "  now $(git describe --tags)"
[ -f comfy_extras/nodes_minimax_h3.py ] || die "no native H3 nodes after upgrade"

./.venv-cu128/bin/pip install -q -r requirements.txt 2>&1 \
  | grep -viE 'warning|notice' | tail -3

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

dl() {  # repo file destdir
  nohup hf download "$1" "$2" --local-dir "$3" \
    > "/workspace/dl-logs/$(basename "$2").log" 2>&1 &
  log "  downloading $(basename "$2")"
}
# black-forest-labs/FLUX.1-dev is a GATED repo: `hf download` returns 401
# without a token whose account has accepted the FLUX.1 [dev] Non-Commercial
# License once on the model page. Checked here, before 52GB of H3 starts, so
# the failure costs seconds rather than being discovered on a pod that has
# already spent twenty minutes downloading.
if [ -z "${HF_TOKEN:-}" ]; then
  die "HF_TOKEN is not set. black-forest-labs/FLUX.1-dev is gated: accept its
  licence once at https://huggingface.co/black-forest-labs/FLUX.1-dev then
  export HF_TOKEN=hf_... before running this script."
fi

log "starting weight downloads (~84GB)"
dl Comfy-Org/MiniMax-H3 "$DIT" "$M"
dl Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors "$M"
dl Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors "$M"
dl Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors "$M"
dl larryvrh/MiniMax-H3-Turbo-Lora minimax_h3_turbo_v4_step600_ema.safetensors "$M/loras"
# The Flux image LoRA. 687,476,088 bytes measured from the HF blobs API.
# Needs no trigger word: neither candidate's model card declares an
# instance_prompt -- this is an "unrestrain" adapter, not a concept LoRA.
dl Heartsync/Flux-NSFW-uncensored lora.safetensors "$M/loras"

# Flux.1-dev itself. NONE of these were downloaded before -- the workflow
# loaded four files the pod never fetched, which surfaces as a failure at
# submit time on a machine that is already billing. Sizes are from the HF
# file-tree API; the filenames must match workflows/flux_dev.json exactly and
# a test asserts that they do.
dl black-forest-labs/FLUX.1-dev flux1-dev.safetensors "$M/diffusion_models"   # 23.8 GB, gated
dl comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors "$M/text_encoders"  # 4.9 GB
dl comfyanonymous/flux_text_encoders clip_l.safetensors "$M/text_encoders"    # 246 MB
# The VAE via the Lumina repackage rather than the gated FLUX.1-dev copy: it is
# byte-identical (335,304,388) and ungated, and it is what ComfyUI's own Flux
# example page points at.
dl Comfy-Org/Lumina_Image_2.0_Repackaged split_files/vae/ae.safetensors "$M/vae"

# ── 4. wait for the weights, then restart ComfyUI ─────────────────────────
while pgrep -f 'hf download' >/dev/null; do
  log "  weights: $(du -sh "$M" 2>/dev/null | cut -f1)"
  sleep 30
done
log "weights complete: $(du -sh "$M" | cut -f1)"
# hf download keeps the remote filename, and "lora.safetensors" says nothing
# in a directory that also holds the H3 turbo LoRA -- and does not match the
# lora_name in workflows/flux_dev.json. Renaming here is what makes those two
# strings the same string. Fail loudly rather than leaving the graph pointing
# at a file that is not there: ComfyUI's own error for a missing LoRA is a
# line in a log nobody is reading at 11:04.
# `hf download` keeps the repo's directory structure, so the Lumina VAE lands
# at vae/split_files/vae/ae.safetensors. VAELoader looks in vae/ and nowhere
# else, so an unflattened file is a file ComfyUI cannot see.
if [ -f "$M/vae/split_files/vae/ae.safetensors" ]; then
  mv "$M/vae/split_files/vae/ae.safetensors" "$M/vae/ae.safetensors"
  rm -rf "$M/vae/split_files"
fi
for required in   "$M/diffusion_models/flux1-dev.safetensors"   "$M/text_encoders/t5xxl_fp8_e4m3fn.safetensors"   "$M/text_encoders/clip_l.safetensors"   "$M/vae/ae.safetensors"; do
  [ -f "$required" ] || die "flux weight missing: $required (check /workspace/dl-logs)"
done

FLUX_LORA="$M/loras/flux_nsfw_uncensored_v1.safetensors"
if [ -f "$M/loras/lora.safetensors" ]; then
  mv "$M/loras/lora.safetensors" "$FLUX_LORA"
fi
[ -f "$FLUX_LORA" ] || die "flux LoRA missing: $FLUX_LORA (check dl-logs/lora.safetensors.log)"
find "$M" -name '*minimax*.safetensors' -printf '%s %p\n' \
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
    *"$flag"*) EXTRA="$EXTRA $flag" ;;
    *)         log "  $flag not supported by this ComfyUI, skipping" ;;
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

log "done. quantisation=${QUANT} vram=${VRAM_GB}GB"
