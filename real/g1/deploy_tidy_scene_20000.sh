#!/usr/bin/env bash
set -euo pipefail
CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$CHENYZ_ROOT/deploy_tidy_scene.sh" \
  --server http://127.0.0.1:8020 \
  --model-paths "$CHENYZ_ROOT/.local/models/tidy_scene_200_step20000/model_paths.json" \
  --trajectory-dir "$CHENYZ_ROOT/.local/first_frames/tidy_scene_200_step20000/episode_000000" \
  "$@"
