#!/usr/bin/env bash
# Final real-data recipe: current state, body auxiliary, dual response v5.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/../../.." && pwd -P)
export GPUS=4
export BATCH_SIZE=${BATCH_SIZE:-256}
export GRAD_ACCUMULATION=1
exec bash "$script_dir/finetune-real-sonic.sh" "$@"
