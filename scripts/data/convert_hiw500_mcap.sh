#!/usr/bin/env bash
# CPU-only, restartable conversion. No training, no raw-data deletion.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_BIN=$(command -v -- "$PYTHON_BIN") || { echo "Python executable not found" >&2; exit 2; }
RAW_ROOT=${RAW_ROOT:-$ROOT/data/hiw500/raw}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/data/hiw500/converted}
cpus=$(nproc)
# nproc reports affinity, not the container's CPU-time quota.
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
  if [[ "$quota" != max ]]; then
    quota_cpus=$((quota / period))
    (( quota_cpus >= 1 )) || quota_cpus=1
    (( quota_cpus >= cpus )) || cpus=$quota_cpus
  fi
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
  read -r quota < /sys/fs/cgroup/cpu/cpu.cfs_quota_us
  read -r period < /sys/fs/cgroup/cpu/cpu.cfs_period_us
  if (( quota > 0 )); then
    quota_cpus=$((quota / period))
    (( quota_cpus >= 1 )) || quota_cpus=1
    (( quota_cpus >= cpus )) || cpus=$quota_cpus
  fi
fi
default_workers=$((cpus * 7 / 8))
(( default_workers >= 1 )) || default_workers=1
WORKERS=${WORKERS:-$default_workers}
ENCODE_THREADS=${ENCODE_THREADS:-1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/deps"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT/scripts/data/hiw_offline_env.sh"
"$PYTHON_BIN" -c 'import mcap, mcap_ros2, av, pyarrow, cv2'
log="$OUTPUT_ROOT/logs/convert-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "Conversion effective_cpus=$cpus workers=$WORKERS encoder_threads=$ENCODE_THREADS log=$log"
"$PYTHON_BIN" -u "$ROOT/scripts/data/convert_hiw500_mcap.py" \
  --raw-root "$RAW_ROOT" --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" --encode-threads "$ENCODE_THREADS" "$@" 2>&1 | tee "$log"
release=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$OUTPUT_ROOT/latest_release.json")
"$PYTHON_BIN" "$ROOT/scripts/data/prepare_hiw500_pretrain_source.py" \
  --release "$release" --output-dir "$release/pretrain_source" 2>&1 | tee -a "$log"
