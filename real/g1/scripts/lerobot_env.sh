#!/usr/bin/env bash
# Optional source for direct dataset/training commands in this environment.
if [[ -n "${ZSH_VERSION-}" ]]; then
  CHENYZ_LEROBOT_SOURCE="${(%):-%x}"
else
  CHENYZ_LEROBOT_SOURCE="${BASH_SOURCE[0]}"
fi
CHENYZ_LEROBOT_ROOT="$(cd "$(dirname "$CHENYZ_LEROBOT_SOURCE")/.." && pwd)"
export LD_LIBRARY_PATH="$CHENYZ_LEROBOT_ROOT/.venv_lerobot/lib/ffmpeg:$CHENYZ_LEROBOT_ROOT/.venv_lerobot/lib/python3.12/site-packages/av.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
