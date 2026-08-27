#!/usr/bin/env bash
# Provision a fresh Ubuntu box as the always-on side of ai-studio.
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
APP_USER=ai-studio
APP_DIR=/srv/ai-studio
REPO="${AI_STUDIO_REPO:-https://github.com/jieyao-MilestoneHub/ai-studio.git}"

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
    # Delivery is a pushed LINE media message, so /files IS a media host and
    # LINE's video-message constraints do apply: HTTPS with a valid cert (this
    # block), and HTTP range requests, which Starlette's FileResponse answers
    # with a 206 and reverse_proxy passes through untouched. Do not put a
    # buffering or transforming handler in front of it.
    reverse_proxy 127.0.0.1:8000
}
CADDY
systemctl enable --now caddy >/dev/null 2>&1 || true
systemctl reload caddy || systemctl restart caddy

say "service"
cat > /etc/systemd/system/ai-studio.service <<UNIT
[Unit]
Description=ai-studio LINE webhook, status pages and file delivery
After=network-online.target
Wants=network-online.target

[Service]
# journalctl -t ai-studio-webhook: without this every unit logs as SYSLOG_IDENTIFIER=uv
# (the ExecStart binary) and only -u can tell them apart.
SyslogIdentifier=ai-studio-webhook
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/uv run ai-studio line serve --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
# LINE requires a 200 within two seconds, so this process must never be cold.
# It is deliberately not socket-activated and not scaled to zero.

[Install]
WantedBy=multi-user.target
UNIT

say "worker"
# The second always-on process. It replaces the old 'open' and 'drain' timers
# outright: a pod is created by the first request that arrives inside business
# hours, not by a clock that fires whether or not anybody asked for anything,
# and work starts within ten seconds instead of at the next five-minute tick.
# Outside 11:00-13:00 Asia/Taipei this loop sleeps and does nothing else.
cat > /etc/systemd/system/ai-studio-worker.service <<UNIT
[Unit]
Description=ai-studio queue worker: opens the pod on demand and renders
After=network-online.target
Wants=network-online.target

[Service]
# journalctl -t ai-studio-worker: without this every unit logs as SYSLOG_IDENTIFIER=uv
# (the ExecStart binary) and only -u can tell them apart.
SyslogIdentifier=ai-studio-worker
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/uv run ai-studio worker
Restart=always
# Ten seconds, not one. A crash loop here would ask for a pod on every restart,
# and although ensure_pod refuses past AI_STUDIO_MAX_POD_OPENS_PER_DAY, the
# cheapest place to not open a pod is before anything asks.
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

say "removing superseded units"
# This script is re-run on boxes it has already provisioned, and it used to
# install four timers. Writing only the two current ones would leave the other
# two ENABLED AND FIRING: ai-studio-open.timer would keep creating a pod at
# 03:00 whether or not anyone had asked for anything, and ai-studio-drain.timer
# would keep claiming jobs out from under the worker, racing it for the same
# queue. An upgrade that leaves the thing it replaced still running is worse
# than no upgrade, so every unit this script has ever installed is removed
# first and only the current set is written back.
for phase in open drain reap close gc; do
  systemctl disable --now "ai-studio-${phase}.timer"   >/dev/null 2>&1 || true
  systemctl disable --now "ai-studio-${phase}.service" >/dev/null 2>&1 || true
  rm -f "/etc/systemd/system/ai-studio-${phase}.timer" \
        "/etc/systemd/system/ai-studio-${phase}.service"
done
systemctl daemon-reload

say "window timers"
# Two timers, and BOTH of them only ever close things. Nothing on a schedule
# opens a pod any more -- that is the point of the worker above, and it means
# every scheduled unit left on this box can only reduce what is billing.
# --terminate-after on the pod itself is still the third and last backstop: it
# closes a pod even if this box dies entirely.
for phase in reap close gc; do
  case "$phase" in
    # Close early once the pod has gone quiet. Also the second line of defence
    # for a worker that died still holding one.
    reap)  cmd="session reap";  when="*:0/1" ;;
    # The hard close at the end of business hours.
    close) cmd="session close"; when="20:05" ;;
    # Daily disk sweep: prune delivered media and received photos past the
    # retention window (AI_STUDIO_FILES_RETENTION_DAYS). 18:30 UTC = 02:30
    # Asia/Taipei, a quiet hour.
    gc)    cmd="gc";            when="18:30" ;;
  esac
  cat > /etc/systemd/system/ai-studio-${phase}.service <<UNIT
[Unit]
Description=ai-studio window ${phase}

[Service]
# journalctl -t ai-studio-${phase}: without this every unit logs as SYSLOG_IDENTIFIER=uv
# (the ExecStart binary) and only -u can tell them apart.
SyslogIdentifier=ai-studio-${phase}
Type=oneshot
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/uv run ai-studio ${cmd}
UNIT
  cat > /etc/systemd/system/ai-studio-${phase}.timer <<UNIT
[Unit]
Description=ai-studio window ${phase} timer

[Timer]
# UTC. 20:05 UTC is 04:05 Asia/Taipei: the quietest hour, so the daily hard
# close (a backstop behind the reaper and --terminate-after) lands on an
# idle pod, not on a render.
OnCalendar=${when}
# Persistent=false on purpose. It mattered more when a timer could open a pod;
# it still matters, because a missed 'close' firing late is noise and one
# firing at boot is confusing. Closing is idempotent either way.
Persistent=false

[Install]
WantedBy=timers.target
UNIT
done

systemctl daemon-reload
systemctl enable --now ai-studio.service >/dev/null 2>&1 || true
systemctl enable --now ai-studio-worker.service >/dev/null 2>&1 || true
for phase in reap close gc; do
  systemctl enable --now ai-studio-${phase}.timer >/dev/null 2>&1 || true
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
       AI_STUDIO_PUBLIC_BASE_URL=https://${HOSTNAME_ARG}
       AI_STUDIO_LLM_ENDPOINT_ID=...
  2. systemctl restart ai-studio ai-studio-worker
  3. curl https://${HOSTNAME_ARG}/healthz          -> {"ok":true,...}
  4. LINE console webhook URL: https://${HOSTNAME_ARG}/callback  -> Verify
  5. systemctl is-active ai-studio ai-studio-worker -> two services active
  6. systemctl list-timers 'ai-studio-*'            -> three timers armed
     Nothing on a timer opens a pod: the worker does, on demand, 11:00-13:00.
  7. uv run ai-studio preflight                     -> the nine Phase 4 checks

  If the worker ever wedges while a pod is open, empty the queue by hand with
  'uv run ai-studio session drain' -- it claims nothing when no pod is up, so
  it is safe to try. That is the only reason drain_window still exists.

  Logs: journalctl -u ai-studio -f    journalctl -u ai-studio-worker -f
NEXT
