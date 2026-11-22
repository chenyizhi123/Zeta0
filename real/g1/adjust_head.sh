#!/usr/bin/env bash
set -euo pipefail
CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$CHENYZ_ROOT/GR00T-WholeBodyControl/.venv_data_collection/bin/python" "$CHENYZ_ROOT/scripts/adjust_head.py" "$@"
