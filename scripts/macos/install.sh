#!/bin/bash
# Run the insider bot and its dashboard in the background on macOS (launchd).
#
#   scripts/macos/install.sh              # Trading 212 DEMO account (default)
#   scripts/macos/install.sh sim          # simulator, no broker
#
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

[ -x "$PY" ] || { echo "No $PY. Create it: cd $REPO && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
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

write_agent() {  # label, log name, program arguments...
  local label="$1" logname="$2"; shift 2
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
  <key>ThrottleInterval</key><integer>900</integer>
  <key>StandardOutPath</key><string>$LOGS/$logname.log</string>
  <key>StandardErrorPath</key><string>$LOGS/$logname.log</string>
</dict>
</plist>
PLIST
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$plist"
  echo "started $label"
}

write_agent com.claudetrade.insider-bot bot \
  /usr/bin/caffeinate -i "$PY" -m invest.invest_bot --strategy insider \
  --broker "$BROKER" --environment paper --state-dir "$STATE"
write_agent com.claudetrade.dashboard dashboard \
  "$PY" "$REPO/dashboard/dashboard.py" --state-dir "$STATE" --port "$PORT"

echo
echo "Dashboard:  http://127.0.0.1:$PORT   (open it in Safari or Chrome)"
echo "Bot log:    tail -f $LOGS/bot.log"
echo "Status:     launchctl list | grep claudetrade"
echo "Stop all:   scripts/macos/uninstall.sh"
