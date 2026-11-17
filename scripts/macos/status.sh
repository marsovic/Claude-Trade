#!/bin/bash
# Is the background bot running? Shows both services, the dashboard, and the logs that
# explain a failure. Paste the whole output when asking for help.
LOGS="$HOME/Library/Logs/claude-trade"
PORT="${DASHBOARD_PORT:-8051}"

echo "== services"
for label in com.claudetrade.insider-bot com.claudetrade.dashboard; do
  line=$(launchctl list | awk -v l="$label" '$3 == l')
  if [ -z "$line" ]; then
    echo "$label: NOT INSTALLED (run scripts/macos/install.sh)"
  elif [ "$(echo "$line" | awk '{print $1}')" = "-" ]; then
    echo "$label: NOT RUNNING, last exit code $(echo "$line" | awk '{print $2}')"
  else
    echo "$label: running (process $(echo "$line" | awk '{print $1}'))"
  fi
done

echo
echo "== dashboard"
answer=""
for _ in 1 2 3 4 5 6 7 8 9 10; do  # it needs a few seconds after a start
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/healthz"; then answer=yes; break; fi
  sleep 1
done
if [ -n "$answer" ]; then
  echo "answering: http://127.0.0.1:$PORT"
else
  echo "NOT answering on port $PORT"
  echo "-- program listening on that port:"
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || echo "none"
  echo "-- last lines of $LOGS/dashboard.log:"
  tail -n 25 "$LOGS/dashboard.log" 2>/dev/null || echo "no log file"
fi

echo
echo "== last lines of the bot log ($LOGS/bot.log)"
tail -n 15 "$LOGS/bot.log" 2>/dev/null || echo "no log file"
