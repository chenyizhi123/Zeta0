#!/usr/bin/env bash
set -euo pipefail
CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHENYZ_ROOT"
export PYTHONUNBUFFERED=1
exec python3 "$CHENYZ_ROOT/run.py" deploy-model \
  --task place_tape_on_table --execute \
  --server http://127.0.0.1:8018 \
  --trajectory-dir "$CHENYZ_ROOT/.local/first_frames/place_tape_on_table/episode_000000" \
  "$@"
