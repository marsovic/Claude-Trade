#!/bin/bash
# Stop the background bot and dashboard and remove them from launchd.
# The state directory (positions, journal) is kept: install.sh resumes it.
set -uo pipefail
DOMAIN="gui/$(id -u)"
for label in com.claudetrade.insider-bot com.claudetrade.dashboard; do
  launchctl bootout "$DOMAIN/$label" 2>/dev/null && echo "stopped $label"
  rm -f "$HOME/Library/LaunchAgents/$label.plist"
done
echo "Done. With a real broker, open positions and their stops stay at the broker."
