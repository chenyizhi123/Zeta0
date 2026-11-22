#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
: "${CONFIG:?CONFIG must point to one resolved phase config}"
[[ -f "$CONFIG" ]] || { echo "config does not exist: $CONFIG" >&2; exit 2; }
exec sbatch --export=ALL,PROJECT_ROOT="$ROOT",CONFIG="$(realpath "$CONFIG")" \
  "$ROOT/scripts/train/predictive_world/pretrain-5n40g.slurm.sh"
