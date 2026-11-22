#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
: "${CONFIG:?CONFIG must point to a resolved predictive-world JSON config}"
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_BIN=$(command -v -- "$PYTHON_BIN") || { echo "Python executable not found" >&2; exit 2; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
NNODES=${NNODES:-5}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:?NODE_RANK is required}
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

common=(--nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
overrides=()
[[ -n ${MICRO_BATCH_SIZE:-} ]] && overrides+=(--micro-batch "$MICRO_BATCH_SIZE")
[[ -n ${GRADIENT_ACCUMULATION_STEPS:-} ]] && overrides+=(
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS")
[[ -n ${EFFECTIVE_GLOBAL_BATCH:-} ]] && overrides+=(
  --effective-global-batch "$EFFECTIVE_GLOBAL_BATCH")

if [[ ${SKIP_DISTRIBUTED_PREFLIGHT:-0} != 1 ]]; then
  preflight_resume=()
  [[ -n ${RESUME:-} ]] && preflight_resume+=(--allow-existing-output)
  "$PYTHON_BIN" -m torch.distributed.run "${common[@]}" \
    "$ROOT/scripts/train/predictive_world/distributed_preflight.py" \
    --config "$CONFIG" --collective-mib "${COLLECTIVE_MIB:-256}" \
    --require-gpu-name "${REQUIRE_GPU_NAME:-H100}" \
    --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB:-75}" \
    "${preflight_resume[@]}" "${overrides[@]}"
fi

train_args=(--config "$CONFIG" --num-workers "${NUM_WORKERS_PER_RANK:-4}")
train_args+=("${overrides[@]}")
[[ -n ${RESUME:-} ]] && train_args+=(--resume "$RESUME")
[[ ${MEMORY_PREFLIGHT_ONLY:-0} == 1 ]] && train_args+=(--preflight-only
  --preflight-updates "${PREFLIGHT_UPDATES:-3}"
  --minimum-memory-headroom-gib "${MINIMUM_MEMORY_HEADROOM_GIB:-8}")
exec "$PYTHON_BIN" -m torch.distributed.run "${common[@]}" \
  "$ROOT/scripts/train/predictive_world/train.py" "${train_args[@]}"
