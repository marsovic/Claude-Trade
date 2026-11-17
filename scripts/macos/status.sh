#!/bin/bash
# Is the background bot running? Shows both services, the last log lines and the dashboard.
LOGS="$HOME/Library/Logs/claude-trade"
PORT="${DASHBOARD_PORT:-8051}"
echo "== services (first column: process id, '-' = not running; second: last exit code)"
launchctl list | grep claudetrade || echo "none installed: run scripts/macos/install.sh"
echo
echo "== dashboard"
curl -s -m 5 "http://127.0.0.1:$PORT/healthz" && echo "  -> http://127.0.0.1:$PORT" \
  || echo "not answering on port $PORT"
echo
echo "== last lines of the bot log ($LOGS/bot.log)"
tail -n 15 "$LOGS/bot.log" 2>/dev/null || echo "no log yet"
