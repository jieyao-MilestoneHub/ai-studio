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
# A third trap lives on the *calling* side, not here: under Git Bash, MSYS
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

# ── 2. weights, in the background: this is the long pole ──────────────────
mkdir -p "$M"/{diffusion_models,text_encoders,vae,loras} /workspace/dl-logs
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/workspace/.hf
command -v hf >/dev/null || pip install -q -U 'huggingface_hub[hf_transfer]'

dl() {  # repo file destdir
  nohup hf download "$1" "$2" --local-dir "$3" \
    > "/workspace/dl-logs/$(basename "$2").log" 2>&1 &
  log "  downloading $(basename "$2")"
}
log "starting weight downloads (~51GB)"
dl Comfy-Org/MiniMax-H3 "$DIT" "$M"
dl Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors "$M"
dl Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors "$M"
dl Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors "$M"
dl larryvrh/MiniMax-H3-Turbo-Lora minimax_h3_turbo_v4_step600_ema.safetensors "$M/loras"

# ── 3. upgrade ComfyUI and install the turbo nodes, while that runs ───────
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

# ── 4. wait for the weights, then restart ComfyUI ─────────────────────────
while pgrep -f 'hf download' >/dev/null; do
  log "  weights: $(du -sh "$M" 2>/dev/null | cut -f1)"
  sleep 30
done
log "weights complete: $(du -sh "$M" | cut -f1)"
find "$M" -name '*minimax*.safetensors' -printf '%s %p\n' \
  | awk '{printf "[setup]   %7.2f GB  %s\n", $1/1073741824, $2}'

log "restarting ComfyUI"
pkill -f 'main.py --listen'; sleep 3
PY="$CU/.venv-cu128/bin/python"
[ -x "$PY" ] || PY=$(command -v python3)
cd "$CU" || die "cannot cd $CU"
# --reserve-vram leaves headroom so a long clip does not OOM at the VAE stage.
nohup "$PY" main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
  --reserve-vram 0.7 > /workspace/comfy.log 2>&1 &

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
    print(f"[setup]   {\"OK  \" if n in info else \"MISS\"} {n}")
sys.exit(1 if missing else 0)
' || die "required H3 nodes are not registered"

log "done. quantisation=${QUANT} vram=${VRAM_GB}GB"
