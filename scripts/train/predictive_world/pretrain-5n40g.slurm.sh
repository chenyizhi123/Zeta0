#!/usr/bin/env bash
#SBATCH --job-name=pworld-5n40g
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-pworld-%j.out
#SBATCH --error=slurm-pworld-%j.err
set -Eeuo pipefail

ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
[[ -n "$ROOT" ]] || { echo "PROJECT_ROOT or SLURM_SUBMIT_DIR is required" >&2; exit 2; }
ROOT=$(cd -- "$ROOT" && pwd -P)
: "${CONFIG:?export CONFIG before submission}"
master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$master_node" | awk '{print $1; exit}')
MASTER_ADDR=${MASTER_ADDR:-$master_node}
MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}
export ROOT CONFIG MASTER_ADDR MASTER_PORT NNODES=5 NPROC_PER_NODE=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
RUN_OUTPUT=$(${PYTHON_BIN:-python} -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["output_dir"])' "$CONFIG")
if [[ -z ${RESUME:-} && -f "$RUN_OUTPUT/.requeue_ready" ]]; then
  completed_update=$(tr -d '[:space:]' < "$RUN_OUTPUT/.requeue_ready")
  [[ "$completed_update" =~ ^[0-9]+$ ]] || {
    echo "invalid requeue marker: $RUN_OUTPUT/.requeue_ready" >&2
    exit 2
  }
  printf -v checkpoint_name 'checkpoint-%08d' "$completed_update"
  checkpoint_path="$RUN_OUTPUT/$checkpoint_name"
  [[ -f "$checkpoint_path/.complete" && -f "$checkpoint_path/.trainer_complete" ]] || {
    echo "requeue checkpoint is incomplete: $checkpoint_path" >&2
    exit 2
  }
  export RESUME="$checkpoint_path"
  echo "auto-resuming from $RESUME"
fi
srun_pid=
requeue_requested=0
on_walltime() {
  trap - USR1
  requeue_requested=1
  srun --overlap --nodes=5 --ntasks=5 --ntasks-per-node=1 \
    bash -lc 'pkill -USR1 -f "[t]rain/predictive_world/train.py" || true' || true
}
trap on_walltime USR1
set +e
srun --kill-on-bad-exit=1 --nodes=5 --ntasks=5 --ntasks-per-node=1 \
  bash -lc 'export NODE_RANK=$SLURM_PROCID; exec "$ROOT/scripts/train/predictive_world/launch-5n40g.sh"' &
srun_pid=$!
wait "$srun_pid"
status=$?
set -e
if (( requeue_requested == 1 )) && [[ -f "$RUN_OUTPUT/.requeue_ready" ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
