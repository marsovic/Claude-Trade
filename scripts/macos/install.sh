#!/bin/bash
# Run the insider bot and its dashboard in the background on macOS (launchd).
#
#   scripts/macos/install.sh              # Trading 212 DEMO account (default)
#   scripts/macos/install.sh sim          # simulator, no broker
#
# Run it again after every `git pull`: it restarts both with the new code (positions and
# journal are kept). No need to activate the virtualenv first.
# Both start now and at every login, and restart if they crash. The Mac is kept awake
# while the bot runs (caffeinate), but a closed laptop lid still puts it to sleep.
# Dashboard: http://127.0.0.1:8051   Logs: ~/Library/Logs/claude-trade/
# Remove with scripts/macos/uninstall.sh
set -euo pipefail

BROKER="${1:-trading212}"
case "$BROKER" in sim|trading212) ;; *) echo "usage: $0 [trading212|sim]"; exit 1 ;; esac
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
STATE="state_insider_${BROKER}"
PORT="${DASHBOARD_PORT:-8051}"
LOGS="$HOME/Library/Logs/claude-trade"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

[ -x "$PY" ] || { echo "No $PY. Create it: cd $REPO && python3 -m venv --clear .venv && .venv/bin/python -m pip install -r requirements.txt"; exit 1; }
# the packages must be in THIS virtualenv: without them the bot cannot even read .env
missing="$("$PY" -c 'import importlib.util as u; print(" ".join(p for m, p in (("dotenv", "python-dotenv"), ("flask", "flask"), ("pandas", "pandas"), ("numpy", "numpy"), ("requests", "requests")) if u.find_spec(m) is None))')"
if [ -n "$missing" ]; then
  echo "Packages missing in $REPO/.venv: $missing"
  echo "Install them:   .venv/bin/python -m pip install -r requirements.txt"
  echo "If that fails, rebuild the environment first:   python3 -m venv --clear .venv"
  exit 1
fi
[ -f "$REPO/.env" ] || { echo "No $REPO/.env: copy .env.example and fill it"; exit 1; }
case "$REPO" in
  "$HOME/Documents"*|"$HOME/Desktop"*|"$HOME/Downloads"*)
    echo "WARNING: $REPO is in a folder macOS protects from background programs."
    echo "Move the project (e.g. to ~/Claude-Trade) or the bot may get 'Operation not permitted'." ;;
esac
# check the configuration once in the foreground, so errors show here and not in a log
"$PY" - "$BROKER" <<'PY' || exit 1
import sys
from invest.config import ConfigError, InvestSettings
try:
    s = InvestSettings.from_env(".env", strategy="insider", broker=sys.argv[1],
                                environment="paper")
except ConfigError as exc:
    sys.exit(f"Configuration error: {exc}")
key = {"massive": s.massive_api_key, "tiingo": s.tiingo_api_key,
       "alpaca": s.alpaca_api_key}.get(s.data_provider, "x")
if not key:
    sys.exit(f"Configuration error: INV_DATA_PROVIDER={s.data_provider} needs its API key")
print("configuration OK")
PY

mkdir -p "$LOGS" "$AGENTS"

write_agent() {  # label, log name, seconds between two restarts, program arguments...
  local label="$1" logname="$2" throttle="$3"; shift 3
  local plist="$AGENTS/$label.plist" args=""
  for a in "$@"; do args="$args    <string>$a</string>
"; done
  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>
$args  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key><string>1</string>
    <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>$throttle</integer>
  <key>StandardOutPath</key><string>$LOGS/$logname.log</string>
  <key>StandardErrorPath</key><string>$LOGS/$logname.log</string>
</dict>
</plist>
PLIST
  # stop the old instance and wait until launchd has let go of it: loading the new one
  # too early fails with "Bootstrap failed: 5: Input/output error"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  for _ in $(seq 1 30); do
    launchctl print "$DOMAIN/$label" >/dev/null 2>&1 || break
    sleep 1
  done
  launchctl bootstrap "$DOMAIN" "$plist" || { sleep 5; launchctl bootstrap "$DOMAIN" "$plist"; }
  echo "started $label"
}

write_agent com.claudetrade.insider-bot bot 60 \
  /usr/bin/caffeinate -i "$PY" -m invest.invest_bot --strategy insider \
  --broker "$BROKER" --environment paper --state-dir "$STATE"
write_agent com.claudetrade.dashboard dashboard 10 \
  "$PY" "$REPO/dashboard/dashboard.py" --state-dir "$STATE" --port "$PORT"

echo
echo "checking that both started..."
sleep 3
"$REPO/scripts/macos/status.sh" || true
echo
echo "Dashboard:  http://127.0.0.1:$PORT   (open it in Safari or Chrome)"
echo "Bot log:    tail -f $LOGS/bot.log"
echo "Status:     scripts/macos/status.sh"
echo "Stop all:   scripts/macos/uninstall.sh"
