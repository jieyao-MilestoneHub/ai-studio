#!/usr/bin/env bash
# Make an always-on Linux box you already own -- the Jetson Orin on the desk --
# the persistent side of ai-studio, with ngrok as the public HTTPS front.
#
#   sudo bash deploy/jetson_setup.sh <static-domain>
#
# This is the sibling of vps_setup.sh for a machine that is NOT a fresh VPS:
# no user is created, nothing is cloned, no Caddy, no ufw. The repo stays where
# the caller checked it out, the units run as the caller (SUDO_USER), and the
# tools live in that user's ~/.local/bin, where the non-root installs put them:
#
#   uv        curl -LsSf https://astral.sh/uv/install.sh | sh
#   runpodctl https://github.com/runpod/runpodctl/releases  (linux-arm64)
#   ngrok     https://ngrok.com/download                    (linux-arm64)
#   ffmpeg    a build with `colordetect`, i.e. >= 8.0 -- `ai-studio doctor` checks
#
# ngrok rather than cloudflared because cloudflared's control channel (UDP/TCP
# 7844) is blocked egress on this network; ngrok tunnels over 443. Use a
# reserved static domain: the free ephemeral one changes on every restart, and
# AI_STUDIO_PUBLIC_BASE_URL is baked into reply links, /files media URLs and
# the worker's delivery URLs, so an ephemeral hostname would mean re-pasting
# the LINE webhook URL and restarting both services every time the tunnel
# bounced.
#
# What this does NOT do: write .env, or run `ngrok config add-authtoken`.
# Credentials are yours to place.

set -euo pipefail

DOMAIN="${1:?usage: jetson_setup.sh <static-domain>}"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
APP_USER="${SUDO_USER:?run with sudo from your own account, not as root directly}"
APP_HOME="$(getent passwd "$APP_USER" | cut -d: -f6)"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$APP_HOME/.local/bin"
UNIT_PATH="$BIN:/usr/local/bin:/usr/bin:/bin"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "checking the tools in $BIN"
for tool in uv runpodctl ngrok ffmpeg ffprobe; do
  [ -x "$BIN/$tool" ] || { echo "  missing $BIN/$tool - install it as $APP_USER first"; exit 1; }
done
[ -f "$APP_DIR/.env" ] || { echo "  missing $APP_DIR/.env"; exit 1; }
[ -f "$APP_HOME/.config/ngrok/ngrok.yml" ] \
  || { echo "  ngrok has no authtoken: run 'ngrok config add-authtoken ...' as $APP_USER"; exit 1; }
[ -f "$APP_HOME/.runpod/config.toml" ] \
  || { echo "  runpodctl is not configured: run 'runpodctl config --apiKey ...' as $APP_USER"; exit 1; }
chmod 600 "$APP_DIR/.env"

say "tunnel"
# The tunnel is its own unit so a crash of the web service does not tear the
# public hostname down, and so it comes back on its own after a reboot.
cat > /etc/systemd/system/ai-studio-ngrok.service <<UNIT
[Unit]
Description=ai-studio public HTTPS front (ngrok, static domain)
After=network-online.target
Wants=network-online.target

[Service]
User=${APP_USER}
Environment=HOME=${APP_HOME}
ExecStart=${BIN}/ngrok http --url=${DOMAIN} 127.0.0.1:8000 --log=stdout --log-format=logfmt
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

say "service"
cat > /etc/systemd/system/ai-studio.service <<UNIT
[Unit]
Description=ai-studio LINE webhook, status pages and file delivery
After=network-online.target ai-studio-ngrok.service
Wants=network-online.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=HOME=${APP_HOME}
Environment=PATH=${UNIT_PATH}
# Loopback only: ngrok forwards to it from this same host, and nothing else
# on the LAN has any business reaching the raw port.
ExecStart=${BIN}/uv run ai-studio line serve --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
# LINE requires a 200 within two seconds, so this process must never be cold.
# It is deliberately not socket-activated and not scaled to zero.

[Install]
WantedBy=multi-user.target
UNIT

say "worker"
# The second always-on process: a pod is created by the first request that
# arrives inside business hours, not by a clock. Outside 11:00-13:00
# Asia/Taipei this loop sleeps -- unless a pod is already open, in which case it
# keeps rendering against it (see runtime.session.ensure_pod).
# HOME matters here: runpodctl reads ~/.runpod/config.toml.
cat > /etc/systemd/system/ai-studio-worker.service <<UNIT
[Unit]
Description=ai-studio queue worker: opens the pod on demand and renders
After=network-online.target
Wants=network-online.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=HOME=${APP_HOME}
Environment=PATH=${UNIT_PATH}
ExecStart=${BIN}/uv run ai-studio worker
Restart=always
# Ten seconds, not one. A crash loop here would ask for a pod on every restart,
# and although ensure_pod refuses past AI_STUDIO_MAX_POD_OPENS_PER_DAY, the
# cheapest place to not open a pod is before anything asks.
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

say "window timers"
# Both timers only ever close things. Nothing on a schedule opens a pod; the
# worker does, on demand. --terminate-after on the pod itself is the last
# backstop and closes a pod even if this box dies entirely.
for phase in reap close gc; do
  case "$phase" in
    reap)  cmd="session reap";  when="*:0/1" ;;
    # The explicit "UTC" suffix is load-bearing: systemd reads a bare
    # OnCalendar in the box's *local* zone, and this box runs Asia/Taipei.
    # Without it "20:05" fired at 20:05 Taipei -- prime evening, not 04:05 --
    # and terminated a pod with a render 46 minutes in (observed live
    # 2026-08-27, pod i5s1j69xkcihnn). Same for gc.
    close) cmd="session close"; when="20:05 UTC" ;;
    # Daily disk sweep: prune delivered media and received photos past the
    # retention window (AI_STUDIO_FILES_RETENTION_DAYS). 18:30 UTC = 02:30
    # Asia/Taipei, a quiet hour.
    gc)    cmd="gc";            when="18:30 UTC" ;;
  esac
  cat > /etc/systemd/system/ai-studio-${phase}.service <<UNIT
[Unit]
Description=ai-studio window ${phase}

[Service]
Type=oneshot
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=HOME=${APP_HOME}
Environment=PATH=${UNIT_PATH}
ExecStart=${BIN}/uv run ai-studio ${cmd}
UNIT
  cat > /etc/systemd/system/ai-studio-${phase}.timer <<UNIT
[Unit]
Description=ai-studio window ${phase} timer

[Timer]
# UTC. 20:05 UTC is 04:05 Asia/Taipei: the quietest hour, so the daily hard
# close (a backstop behind the reaper and --terminate-after) lands on an
# idle pod, not on a render.
OnCalendar=${when}
# A missed 'close' firing late is noise; closing is idempotent either way.
Persistent=false

[Install]
WantedBy=timers.target
UNIT
done

say "no sleeping"
# A box that suspends loses webhooks and the window. Mask the sleep targets so
# a lid, an idle timer or a stray key cannot put it under.
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target >/dev/null 2>&1 || true

systemctl daemon-reload
systemctl enable --now ai-studio-ngrok.service >/dev/null 2>&1 || true
systemctl enable --now ai-studio.service >/dev/null 2>&1 || true
systemctl enable --now ai-studio-worker.service >/dev/null 2>&1 || true
for phase in reap close gc; do
  systemctl enable --now ai-studio-${phase}.timer >/dev/null 2>&1 || true
done

say "next steps"
cat <<NEXT
  1. ${APP_DIR}/.env must carry AI_STUDIO_PUBLIC_BASE_URL=https://${DOMAIN}
     (edit, then: systemctl restart ai-studio ai-studio-worker)
  2. curl https://${DOMAIN}/healthz                 -> {"ok":true,...}
  3. LINE console webhook URL: https://${DOMAIN}/callback  -> Verify
  4. systemctl is-active ai-studio-ngrok ai-studio ai-studio-worker -> three active
  5. systemctl list-timers 'ai-studio-*'            -> three timers armed
  6. cd ${APP_DIR} && uv run ai-studio preflight

  Logs: journalctl -u ai-studio -f    journalctl -u ai-studio-worker -f
        journalctl -u ai-studio-ngrok -f
NEXT
