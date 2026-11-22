#!/usr/bin/env bash
set -euo pipefail

CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHENYZ_ROOT"
export PYTHONUNBUFFERED=1

# Oct 29 download: PSI0, 30Hz, 50 actions. f uses the matching training
# dataset's episode 0, frame 0 (motion64 + hand12).
exec python3 "$CHENYZ_ROOT/run.py" deploy-model \
  --task place_chair --execute \
  --server http://127.0.0.1:8015 \
  --trajectory-dir "$CHENYZ_ROOT/.local/first_frames/place_chair_20261029/episode_000000" \
  "$@"
