#!/usr/bin/env bash
# Provision a fresh Ubuntu box as the always-on side of videogen.
#
#   curl -fsSL <this file> | bash -s -- <hostname>
# or, on the box:
#   sudo bash deploy/vps_setup.sh vg.example.com
#
# `hostname` is what LINE will call. If you do not own a domain, pass the box's
# own IP with dots turned into dashes plus `.sslip.io`, e.g.
#   203-0-113-7.sslip.io
# sslip.io resolves that to 203.0.113.7 with no account and no DNS to manage,
# and Let's Encrypt will issue a real certificate for it — which matters because
# LINE requires HTTPS from a normally trusted CA and will not accept a bare IP
# or a self-signed cert.
#
# What this does NOT do: write .env. Credentials are yours to place, and a
# provisioning script is the wrong place for a channel access token.

set -euo pipefail

HOSTNAME_ARG="${1:?usage: vps_setup.sh <hostname>}"
APP_USER=videogen
APP_DIR=/srv/ai-studio
REPO="${VIDEOGEN_REPO:-https://github.com/jieyao-MilestoneHub/ai-studio.git}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root"; exit 1; }

say "packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# ffmpeg is here for ffprobe: the provider probes each fetched clip to record its
# real dimensions, duration and whether it actually has an audio track.
apt-get install -y -qq git curl ca-certificates ffmpeg ufw >/dev/null

say "user and checkout"
id -u "$APP_USER" >/dev/null 2>&1 || useradd -r -m -d /home/"$APP_USER" -s /bin/bash "$APP_USER"
mkdir -p "$APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --quiet
else
  # A private repo needs credentials. Easiest: clone it yourself as $APP_USER
  # with a deploy key, or rsync the working tree up. This step is allowed to
  # fail so the rest of the provisioning still completes.
  git clone --quiet "$REPO" "$APP_DIR" || echo "  clone failed - copy the repo to $APP_DIR yourself"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

say "uv"
if ! [ -x /usr/local/bin/uv ]; then
  curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
fi
/usr/local/bin/uv --version

say "dependencies"
sudo -u "$APP_USER" env PATH=/usr/local/bin:/usr/bin:/bin \
  sh -c "cd $APP_DIR && uv sync --extra line" || echo "  uv sync failed - check the checkout"

say "caddy for TLS"
if ! command -v caddy >/dev/null; then
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https >/dev/null
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy >/dev/null
fi

cat > /etc/caddy/Caddyfile <<CADDY
${HOSTNAME_ARG} {
    encode zstd gzip
    # Clips are small (a measured 5s clip is 0.99MB) and delivery is a plain
    # link, so none of LINE's video-message constraints apply here — no range
    # requests, no 200MB ceiling, no poster image.
    reverse_proxy 127.0.0.1:8000
}
CADDY
systemctl enable --now caddy >/dev/null 2>&1 || true
systemctl reload caddy || systemctl restart caddy

say "service"
cat > /etc/systemd/system/videogen.service <<UNIT
[Unit]
Description=videogen LINE webhook, status pages and file delivery
After=network-online.target
Wants=network-online.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/uv run videogen line serve --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
# LINE requires a 200 within two seconds, so this process must never be cold.
# It is deliberately not socket-activated and not scaled to zero.

[Install]
WantedBy=multi-user.target
UNIT

say "window timers"
# One unit per phase. --terminate-after on the pod is still the backstop: it
# guarantees a pod gets closed even if this box dies mid-window.
for phase in open drain reap close; do
  case "$phase" in
    open)  cmd="session open --until 13:00 --tz Asia/Taipei"; when="03:00" ;;
    # The one that actually makes videos. It exits immediately and successfully
    # when no window is open, so firing every 5 minutes all day is a no-op
    # outside the window -- and if a drain dies mid-window the next tick picks
    # the queue back up. systemd will not start a second instance while one is
    # still running, so the long render is not interrupted by the next tick.
    drain) cmd="session drain";                               when="*:0/5" ;;
    reap)  cmd="session reap";                                when="*:0/5" ;;
    close) cmd="session close";                               when="05:00" ;;
  esac
  cat > /etc/systemd/system/videogen-${phase}.service <<UNIT
[Unit]
Description=videogen window ${phase}

[Service]
Type=oneshot
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/uv run videogen ${cmd}
UNIT
  cat > /etc/systemd/system/videogen-${phase}.timer <<UNIT
[Unit]
Description=videogen window ${phase} timer

[Timer]
# UTC. 11:00 Asia/Taipei is 03:00 UTC.
OnCalendar=${when}
# Persistent=false on purpose: a missed 'open' must NOT fire late. Booting a
# GPU pod at 3am because the box was down at 11:00 is exactly the unattended
# spend this project is built to avoid.
Persistent=false

[Install]
WantedBy=timers.target
UNIT
done

systemctl daemon-reload
systemctl enable --now videogen.service >/dev/null 2>&1 || true
for phase in open drain reap close; do
  systemctl enable --now videogen-${phase}.timer >/dev/null 2>&1 || true
done

say "firewall"
ufw --force reset >/dev/null 2>&1 || true
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp >/dev/null
ufw allow 80/tcp >/dev/null   # Let's Encrypt HTTP-01
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null
# 8000 is intentionally not opened: only Caddy on loopback reaches it.

say "next steps"
cat <<NEXT
  1. Put credentials in ${APP_DIR}/.env (chown ${APP_USER}, chmod 600):
       LINE_CHANNEL_SECRET=...
       LINE_CHANNEL_ACCESS_TOKEN=...
       LINE_ALLOWED_GROUP_ID=            <- leave empty to run capture mode
       RUNPOD_API_KEY=...
       VIDEOGEN_PUBLIC_BASE_URL=https://${HOSTNAME_ARG}
       VIDEOGEN_LLM_ENDPOINT_ID=...
  2. systemctl restart videogen
  3. curl https://${HOSTNAME_ARG}/healthz          -> {"ok":true,...}
  4. LINE console webhook URL: https://${HOSTNAME_ARG}/callback  -> Verify
  5. systemctl list-timers 'videogen-*'            -> three timers armed
  Logs: journalctl -u videogen -f
NEXT
