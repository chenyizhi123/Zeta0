#!/usr/bin/env bash
set -euo pipefail
CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHENYZ_ROOT"
export PYTHONUNBUFFERED=1
# Oct 29 download, step20000: PSI0, 30Hz/50 steps, matching demonstration frame.
exec python3 "$CHENYZ_ROOT/run.py" deploy-model \
  --task blow_dry_hair --execute \
  --server http://127.0.0.1:8017 \
  --trajectory-dir "$CHENYZ_ROOT/.local/first_frames/blow_dry_hair_20261029/episode_000000" \
  "$@"
