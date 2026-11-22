#!/usr/bin/env bash
set -euo pipefail

CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHENYZ_ROOT"
export PYTHONUNBUFFERED=1

# Saves/exits an existing recorder, then starts SONIC and waits for f / v / m.
# Add --dry-run to show the startup plan without connecting to the robot.
exec python3 "$CHENYZ_ROOT/run.py" deploy-model \
  --task tidy_table_trash --execute "$@"
