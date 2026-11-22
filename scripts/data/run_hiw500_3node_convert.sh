#!/usr/bin/env bash
# Use as the SAME platform startup command on each of 3 nodes (8 H100/node).
# No Slurm submission, SSH, manual rank assignment or training is performed.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_BIN=$(command -v -- "$PYTHON_BIN") || { echo "Python executable not found" >&2; exit 2; }
RAW_ROOT=${RAW_ROOT:-$ROOT/data/hiw500/raw}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/data/hiw500/converted}
nodes=${PET_NNODES:-${NNODES:-${SLURM_NNODES:-}}}
rank=${PET_NODE_RANK:-${NODE_RANK:-${SLURM_NODEID:-}}}
local_gpus=${PET_NPROC_PER_NODE:-8}
master=${MASTER_ADDR:-}
port=${MASTER_PORT:-29500}
EXPECTED_NODES=3
EXPECTED_GPUS_PER_NODE=8
EXPECTED_CPUS_PER_NODE=160
WORKERS_PER_NODE=${WORKERS_PER_NODE:-128}
COMM_TIMEOUT_SECONDS=${COMM_TIMEOUT_SECONDS:-300}
PROGRESS_EVERY_SECONDS=${PROGRESS_EVERY_SECONDS:-10}
TARGET_SECONDS=3600
DRY_RUN=${DRY_RUN:-0}
[[ "$nodes" == "$EXPECTED_NODES" && "$rank" =~ ^[0-9]+$ && "$local_gpus" == "$EXPECTED_GPUS_PER_NODE" ]] || {
  echo "Requires a 3-node / 8-GPU-per-node platform job with PET_NODE_RANK (or NODE_RANK)." >&2; exit 2;
}
(( rank < nodes )) || { echo "Invalid node rank $rank" >&2; exit 2; }
[[ -n "$master" && "$port" =~ ^[0-9]+$ ]] && (( port > 0 && port < 65536 )) || {
  echo "Platform MASTER_ADDR and valid MASTER_PORT are required." >&2; exit 2;
}
[[ "$WORKERS_PER_NODE" =~ ^[1-9][0-9]*$ && "$COMM_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ && "$PROGRESS_EVERY_SECONDS" =~ ^[1-9][0-9]*$ ]] || exit 2
if [[ "$DRY_RUN" == 1 ]]; then
  echo "nodes=$nodes rank=$rank GPUs_per_node=$local_gpus required_CPUs=$EXPECTED_CPUS_PER_NODE"
  echo "workers_per_node=$WORKERS_PER_NODE codec=mjpeg-copy target_seconds=$TARGET_SECONDS runtime_limit=none progress_every=$PROGRESS_EVERY_SECONDS"
  echo "raw=$RAW_ROOT output=$OUTPUT_ROOT rendezvous=$master:$port"
  exit 0
fi
[[ -x "$PYTHON_BIN" && -d "$RAW_ROOT" ]] || { echo "Missing Python environment or raw data mount" >&2; exit 2; }
command -v ffmpeg >/dev/null
command -v setsid >/dev/null
command -v flock >/dev/null
cpus=$(nproc)
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
  if [[ "$quota" != max ]]; then
    quota_cpus=$((quota / period))
    (( quota_cpus >= cpus )) || cpus=$quota_cpus
  fi
fi
(( cpus >= EXPECTED_CPUS_PER_NODE )) || {
  echo "Actual CPU allocation=$cpus, expected >=$EXPECTED_CPUS_PER_NODE; nproc alone is not a quota." >&2; exit 2;
}
memory_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if [[ -r /sys/fs/cgroup/memory.max && -r /sys/fs/cgroup/memory.current ]]; then
  read -r memory_limit < /sys/fs/cgroup/memory.max
  read -r memory_used < /sys/fs/cgroup/memory.current
  if [[ "$memory_limit" != max ]]; then
    quota_memory_kib=$(((memory_limit - memory_used) / 1024))
    (( quota_memory_kib >= memory_kib )) || memory_kib=$quota_memory_kib
  fi
fi
# Approximate headroom, not a promise of constant memory per variable-length episode.
memory_workers=$(((memory_kib / 1024 / 1024 - 16) / 3))
(( memory_workers >= 1 )) || { echo "Insufficient free memory for conversion" >&2; exit 2; }
(( WORKERS_PER_NODE <= cpus - 16 )) || WORKERS_PER_NODE=$((cpus - 16))
(( WORKERS_PER_NODE <= memory_workers )) || WORKERS_PER_NODE=$memory_workers
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/deps"
stamp=$(date +%Y%m%d-%H%M%S)
log="$OUTPUT_ROOT/logs/node-$rank-$stamp-$$.log"
exec > >(tee -a "$log") 2>&1
echo "bootstrap rank=$rank python=$PYTHON_BIN dependency_mode=offline log=$log"
# Shared lock also catches accidentally launching multiple copies for one rank.
exec 8>"$OUTPUT_ROOT/.launcher-rank-$rank.lock"
flock -n 8 || { echo "A launcher already owns rank $rank in this output directory" >&2; exit 2; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT/scripts/data/hiw_offline_env.sh"
"$PYTHON_BIN" -c 'import torch, torch.distributed, pyarrow, cv2, mcap, mcap_ros2'
"$PYTHON_BIN" - "$EXPECTED_GPUS_PER_NODE" <<'PY'
import sys, torch
n = int(sys.argv[1])
if torch.cuda.device_count() != n:
    raise SystemExit(f"expected {n} visible GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(n)]
if any("H100" not in name for name in names):
    raise SystemExit(f"expected H100 allocation, got {names}")
print("GPU allocation:", names, flush=True)
PY
echo "rank=$rank nodes=$nodes effective_cpus=$cpus workers=$WORKERS_PER_NODE codec=mjpeg-copy"
echo "target=1-hour (not guaranteed); runtime_limit=none; progress_every=${PROGRESS_EVERY_SECONDS}s; log=$log"
conversion_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$conversion_pid"; do
    [[ -n "$pid" ]] || continue
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  # Only these newly created process groups are targeted, never pkill/python.
  for ((attempt=0; attempt<5; attempt++)); do
    local alive=0
    for pid in "$conversion_pid"; do
      [[ -n "$pid" ]] || continue
      kill -0 -- "-$pid" 2>/dev/null && alive=1
    done
    (( alive )) || break
    sleep 1
  done
  for pid in "$conversion_pid"; do
    [[ -n "$pid" ]] || continue
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid "$PYTHON_BIN" -u "$ROOT/scripts/data/convert_hiw500_multinode.py" \
  --raw-root "$RAW_ROOT" --output-root "$OUTPUT_ROOT" \
  --rank "$rank" --nodes "$nodes" --master-addr "$master" --master-port "$port" \
  --workers "$WORKERS_PER_NODE" --codec mjpeg \
  --communication-timeout-seconds "$COMM_TIMEOUT_SECONDS" --target-seconds "$TARGET_SECONDS" \
  --progress-every-seconds "$PROGRESS_EVERY_SECONDS" &
conversion_pid=$!
set +e
wait "$conversion_pid"
status=$?
set -e
exit "$status"
