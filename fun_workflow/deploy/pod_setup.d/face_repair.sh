#!/usr/bin/env bash
# FaceDetailer for /短劇 keyframes -- a pod_setup.d extension, BEST EFFORT.
#
# Shipped to /workspace/pod_setup.d/ by fun_workflow's worker
# (`ai_studio.runtime.session.provision(extras=...)`) and run by ai-studio's
# pod_setup.sh after everything it owns is up. Installs ComfyUI-Impact-Pack
# (FaceDetailer) + Impact-Subpack (UltralyticsDetectorProvider) and one bbox
# model. Used only by workflows/flux_dev_i2i_face.json, and only when
# providers/flux.py sees both nodes in /object_info; without them a drama
# renders plain image-to-image keyframes and records "face_repair: skipped".
#
# Nothing here may fail the pod: every step logs its outcome and the script
# always exits 0. Idempotent -- it runs on every pod open. Paths (CU, M, PY,
# EXTRA) and `log` come from pod_setup.sh's environment.

set -u
: "${CU:?pod_setup.sh exports CU}" "${M:?}" "${PY:?}"
EXTRA="${EXTRA:-}"
type log >/dev/null 2>&1 || log() { printf '[setup]   %s\n' "$*"; }

cd "$CU/custom_nodes" || { log "face-repair: no custom_nodes dir, skipping"; exit 0; }
for repo in ltdrdata/ComfyUI-Impact-Pack ltdrdata/ComfyUI-Impact-Subpack; do
  name="${repo##*/}"
  if [ -d "$name/.git" ]; then
    log "face-repair: $name already present"
  elif git clone --depth 1 --quiet "https://github.com/$repo" 2>/dev/null; then
    log "face-repair: cloned $name"
  else
    log "face-repair: WARNING could not clone $repo; FaceDetailer stays off"
    exit 0
  fi
  if [ -f "$name/requirements.txt" ]; then
    # Through ComfyUI's own venv, like everything else ComfyUI loads.
    "$CU/.venv-cu128/bin/pip" install -q -r "$name/requirements.txt" 2>&1 \
      | grep -viE 'warning|notice' | tail -2 \
      || log "face-repair: WARNING pip install for $name reported errors"
  fi
done
mkdir -p "$M/ultralytics/bbox" /workspace/dl-logs
if [ ! -f "$M/ultralytics/bbox/face_yolov8m.pt" ]; then
  hf download Bingsu/adetailer face_yolov8m.pt --local-dir "$M/ultralytics/bbox" \
    > /workspace/dl-logs/face_yolov8m.pt.log 2>&1 \
    && log "face-repair: downloaded face_yolov8m.pt" \
    || log "face-repair: WARNING face_yolov8m.pt download failed (see dl-logs)"
fi
# Impact-Pack registers nodes on ComfyUI start; restart so /object_info
# reflects them. Same launch line as pod_setup.sh's step 4, same flags.
pkill -f 'main.py --listen'; sleep 3
cd "$CU" || exit 0
# shellcheck disable=SC2086
nohup "$PY" main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
  $EXTRA --reserve-vram 0.7 > /workspace/comfy.log 2>&1 &
for i in $(seq 1 60); do
  sleep 5
  curl -sf -m 5 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break
done
curl -s -m 30 http://127.0.0.1:8188/object_info | "$PY" -c '
import json, sys
info = json.load(sys.stdin)
for n in ("FaceDetailer", "UltralyticsDetectorProvider"):
    print("[setup]   face-repair " + ("OK  " if n in info else "MISS") + " " + n)
' || log "face-repair: WARNING could not read /object_info after restart"
exit 0
