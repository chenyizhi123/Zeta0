#!/usr/bin/env bash
set -euo pipefail
CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHENYZ_DATA_PY="$CHENYZ_ROOT/.venv_lerobot/bin/python"
source "$CHENYZ_ROOT/scripts/lerobot_env.sh"
if [[ ! -x "$CHENYZ_DATA_PY" ]]; then
  echo "采集环境缺失，请先运行 bash scripts/install_lerobot.sh" >&2
  exit 1
fi
exec "$CHENYZ_DATA_PY" -u "$CHENYZ_ROOT/scripts/record_lerobot.py" "$@"
