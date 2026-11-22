#!/usr/bin/env bash
set -euo pipefail
CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHENYZ_ROOT"
export PYTHONUNBUFFERED=1
CHECK_CUDA=1
for arg in "$@"; do
  case "$arg" in
    --dry-run|--help|-h) CHECK_CUDA=0 ;;
  esac
done
if [[ "$CHECK_CUDA" == 1 ]]; then
  "$CHENYZ_ROOT/.venv_policy/bin/python" "$CHENYZ_ROOT/scripts/check_policy_cuda.py"
fi
exec python3 "$CHENYZ_ROOT/run.py" deploy-model \
  --task tidy_scene --execute \
  --server http://127.0.0.1:8019 \
  --trajectory-dir "$CHENYZ_ROOT/.local/first_frames/tidy_scene/episode_000000" \
  "$@"
