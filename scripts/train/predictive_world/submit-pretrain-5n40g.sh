#!/usr/bin/env bash
# One-command 5-node/40-H100 predictive-world pretraining entry point.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_BIN=$(command -v -- "$PYTHON_BIN") || { echo "Python executable not found" >&2; exit 2; }
PUBLIC_DATA_ROOT=${PUBLIC_DATA_ROOT:-$ROOT/data}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PUBLIC_DATA_ROOT/predictive_world_artifacts/he_psi0_hiw_temporal}
LEGACY_ARTIFACT_ROOT=${LEGACY_ARTIFACT_ROOT:-$PUBLIC_DATA_ROOT/pretrain_predictor_artifacts/he_psi0_hiw_v4}
: "${QWEN_MODEL:?Set QWEN_MODEL to the matching local pretrained Qwen directory}"
OUTPUT_ROOT=${OUTPUT_ROOT:-$ARTIFACT_ROOT/runs}
SEED=${SEED:-101}
WORLD_UPDATES=${WORLD_UPDATES:-15000}
JOINT_UPDATES=${JOINT_UPDATES:-20000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-256}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
JOINT_MICRO_BATCH_SIZE=${JOINT_MICRO_BATCH_SIZE:-220}
JOINT_GRADIENT_ACCUMULATION_STEPS=${JOINT_GRADIENT_ACCUMULATION_STEPS:-$GRADIENT_ACCUMULATION_STEPS}
OBSERVED_VISUAL_MODE=${OBSERVED_VISUAL_MODE:-current_multiview}
case "$OBSERVED_VISUAL_MODE" in
  history|current_multiview) ;;
  *) echo "invalid OBSERVED_VISUAL_MODE" >&2; exit 2 ;;
esac
EXPECTED_NNODES=${EXPECTED_NNODES:-5}
EXPECTED_GPUS_PER_NODE=${EXPECTED_GPUS_PER_NODE:-8}
NUM_WORKERS_PER_RANK=${NUM_WORKERS_PER_RANK:-4}
FEATURE_CACHE_BATCH_SIZE=${FEATURE_CACHE_BATCH_SIZE:-32}
RELEASE_PROGRESS_EVERY=${RELEASE_PROGRESS_EVERY:-250}
PREFLIGHT_UPDATES=${PREFLIGHT_UPDATES:-3}
MINIMUM_MEMORY_HEADROOM_GIB=${MINIMUM_MEMORY_HEADROOM_GIB:-8}
MINIMUM_SHM_GIB=${MINIMUM_SHM_GIB:-64}
LAUNCH_BACKEND=${LAUNCH_BACKEND:-auto}
DRY_RUN=${DRY_RUN:-0}

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}

[[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2; exit 2; }
for value_name in SEED WORLD_UPDATES JOINT_UPDATES MICRO_BATCH_SIZE \
  GRADIENT_ACCUMULATION_STEPS JOINT_MICRO_BATCH_SIZE JOINT_GRADIENT_ACCUMULATION_STEPS \
  EXPECTED_NNODES EXPECTED_GPUS_PER_NODE \
  NUM_WORKERS_PER_RANK FEATURE_CACHE_BATCH_SIZE RELEASE_PROGRESS_EVERY \
  PREFLIGHT_UPDATES MINIMUM_SHM_GIB; do
  value=${!value_name}
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "$value_name must be an integer" >&2; exit 2; }
done
(( WORLD_UPDATES > 0 && JOINT_UPDATES > 0 && MICRO_BATCH_SIZE > 0 && JOINT_MICRO_BATCH_SIZE > 0 )) || {
  echo "update budgets and per-stage micro batch sizes must be positive" >&2
  exit 2
}
(( GRADIENT_ACCUMULATION_STEPS > 0 && JOINT_GRADIENT_ACCUMULATION_STEPS > 0 \
  && EXPECTED_NNODES > 0 && EXPECTED_GPUS_PER_NODE > 0 )) || {
  echo "topology and gradient accumulation must be positive" >&2
  exit 2
}
(( FEATURE_CACHE_BATCH_SIZE > 0 && RELEASE_PROGRESS_EVERY > 0 )) || {
  echo "feature-cache batch size and release progress interval must be positive" >&2
  exit 2
}
(( PREFLIGHT_UPDATES >= 3 )) || { echo "PREFLIGHT_UPDATES must be at least 3" >&2; exit 2; }
[[ "$MINIMUM_MEMORY_HEADROOM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "MINIMUM_MEMORY_HEADROOM_GIB must be non-negative" >&2
  exit 2
}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }

case "$LAUNCH_BACKEND" in
  auto)
    if [[ -n ${PET_NNODES:-} || -n ${PET_NODE_RANK:-} || -n ${PET_NPROC_PER_NODE:-} ]]; then
      LAUNCH_BACKEND=inspire
    else
      LAUNCH_BACKEND=slurm
    fi
    ;;
  inspire|slurm) ;;
  *) echo "LAUNCH_BACKEND must be auto, inspire or slurm" >&2; exit 2 ;;
esac

if [[ -z ${RUN_ID:-} ]]; then
  # Stable across replacement allocations so rerunning the same one-liner
  # discovers its immutable configs and latest complete checkpoint.
  run_identity="predictive-world-prior-response-fixed-slots-seed${SEED}-b${MICRO_BATCH_SIZE}-a${GRADIENT_ACCUMULATION_STEPS}"
  if [[ "$OBSERVED_VISUAL_MODE" == current_multiview ]]; then
    run_identity+="-current-multiview-v1-checkpoint-fix-v1-early20"
  fi
  RUN_ID=${run_identity//[^A-Za-z0-9._-]/_}
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "RUN_ID must contain only letters, digits, dot, underscore or hyphen" >&2
  exit 2
}

RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
# Retain the completed world run and give changed joint batches their own
# immutable configs, preflight report and checkpoints.
if [[ -z ${JOINT_RUN_ROOT:-} ]]; then
  JOINT_RUN_ROOT=$RUN_ROOT
  if (( JOINT_MICRO_BATCH_SIZE != MICRO_BATCH_SIZE \
    || JOINT_GRADIENT_ACCUMULATION_STEPS != GRADIENT_ACCUMULATION_STEPS )); then
    JOINT_RUN_ROOT=$RUN_ROOT/joint-b${JOINT_MICRO_BATCH_SIZE}-a${JOINT_GRADIENT_ACCUMULATION_STEPS}
  fi
fi
WORLD_PREFLIGHT_CONFIG=$RUN_ROOT/configs/world_preflight.json
WORLD_CONFIG=$RUN_ROOT/configs/world.json
JOINT_PREFLIGHT_CONFIG=$JOINT_RUN_ROOT/configs/joint_preflight.json
JOINT_CONFIG=$JOINT_RUN_ROOT/configs/joint.json

resolve_run() {
  "$PYTHON_BIN" "$ROOT/scripts/train/predictive_world/resolve_two_stage_run.py" \
    --project-root "$ROOT" --artifact-root "$ARTIFACT_ROOT" \
    --qwen-model "$QWEN_MODEL" --run-root "$RUN_ROOT" \
    --joint-run-root "$JOINT_RUN_ROOT" \
    --nodes "$EXPECTED_NNODES" --gpus-per-node "$EXPECTED_GPUS_PER_NODE" \
    --observed-visual-mode "$OBSERVED_VISUAL_MODE" \
    --micro-batch "$MICRO_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --joint-micro-batch "$JOINT_MICRO_BATCH_SIZE" \
    --joint-gradient-accumulation-steps "$JOINT_GRADIENT_ACCUMULATION_STEPS" \
    --world-updates "$WORLD_UPDATES" --joint-updates "$JOINT_UPDATES" --seed "$SEED"
}

preflight_passed() {
  local report=$1
  [[ -f "$report" ]] && "$PYTHON_BIN" - "$report" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "passed" else 1)
PY
}

complete_training() {
  local marker=$1 expected=$2
  [[ -f "$marker" ]] && [[ $(tr -d '[:space:]' < "$marker") == "$expected" ]]
}

latest_resume() {
  local output=$1
  local candidate
  while IFS= read -r candidate; do
    if [[ -f "$candidate/.complete" && -f "$candidate/.trainer_complete" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done < <(find "$output" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-[0-9]*' -print 2>/dev/null | sort -r)
}

if [[ "$LAUNCH_BACKEND" == inspire ]]; then
  : "${PET_NNODES:?PET_NNODES is required for the Inspire backend}"
  : "${PET_NODE_RANK:?PET_NODE_RANK is required for the Inspire backend}"
  : "${PET_NPROC_PER_NODE:?PET_NPROC_PER_NODE is required for the Inspire backend}"
  : "${MASTER_ADDR:?MASTER_ADDR is required for the Inspire backend}"
  base_port=${MASTER_PORT:-29500}
  [[ "$PET_NNODES" =~ ^[1-9][0-9]*$ && "$PET_NODE_RANK" =~ ^[0-9]+$ && \
    "$PET_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid PET topology" >&2
    exit 2
  }
  (( PET_NNODES == EXPECTED_NNODES && PET_NPROC_PER_NODE == EXPECTED_GPUS_PER_NODE \
    && PET_NODE_RANK < PET_NNODES )) || {
    echo "this run requires ${EXPECTED_NNODES} workers x ${EXPECTED_GPUS_PER_NODE} H100; received ${PET_NNODES}x${PET_NPROC_PER_NODE}" >&2
    exit 2
  }
  [[ "$base_port" =~ ^[1-9][0-9]*$ ]] && (( base_port + 50 <= 65535 )) || {
    echo "MASTER_PORT must leave room for launcher ports through $((base_port + 50))" >&2
    exit 2
  }

  sync_stage() {
    local stage=$1 local_status=$2 port_offset=$3
    "$PYTHON_BIN" "$ROOT/scripts/train/predictive_world/platform_stage_sync.py" \
      --master-addr "$MASTER_ADDR" --master-port "$((base_port + port_offset))" \
      --rank "$PET_NODE_RANK" --world-size "$PET_NNODES" \
      --local-status "$local_status" --stage "$stage"
  }

  mkdir -p "$RUN_ROOT/platform-logs"
  exec > >(tee -a "$RUN_ROOT/platform-logs/worker-${PET_NODE_RANK}.log") 2>&1
  echo "backend=inspire run_root=$RUN_ROOT joint_run_root=$JOINT_RUN_ROOT topology=${PET_NNODES}x${PET_NPROC_PER_NODE}"
  echo "training_contract phase=world_model updates=$WORLD_UPDATES micro_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch=$((PET_NNODES * PET_NPROC_PER_NODE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
  echo "training_contract phase=joint updates=$JOINT_UPDATES micro_batch=$JOINT_MICRO_BATCH_SIZE accumulation=$JOINT_GRADIENT_ACCUMULATION_STEPS global_batch=$((PET_NNODES * PET_NPROC_PER_NODE * JOINT_MICRO_BATCH_SIZE * JOINT_GRADIENT_ACCUMULATION_STEPS))"

  stage_status=0
  "$PYTHON_BIN" - "$PET_NPROC_PER_NODE" "$MINIMUM_SHM_GIB" <<'PY' || stage_status=$?
import os
import sys
import torch
expected, minimum_shm = int(sys.argv[1]), float(sys.argv[2])
count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(count)]
memory = [torch.cuda.get_device_properties(index).total_memory / 2**30 for index in range(count)]
stat = os.statvfs("/dev/shm")
shm_gib = stat.f_frsize * stat.f_blocks / 2**30
print(f"local_hardware gpu_count={count} names={names} memory_gib={[round(x, 1) for x in memory]} shm_gib={shm_gib:.1f}")
if count != expected or any("H100" not in name.upper() for name in names):
    raise SystemExit(f"expected {expected} visible H100 GPUs")
if not memory or min(memory) < 75:
    raise SystemExit("minimum GPU memory is below 75 GiB")
if shm_gib < minimum_shm:
    raise SystemExit(f"/dev/shm is {shm_gib:.1f} GiB; require at least {minimum_shm:.1f} GiB")
PY
  sync_stage hardware "$stage_status" 1

  stage_status=0
  if (( PET_NODE_RANK == 0 )); then
    "$PYTHON_BIN" "$ROOT/scripts/data/prepare_predictive_world_release.py" \
      --legacy-artifact-root "$LEGACY_ARTIFACT_ROOT" \
      --output "$ARTIFACT_ROOT" --qwen-model "$QWEN_MODEL" \
      --progress-every "$RELEASE_PROGRESS_EVERY" || stage_status=$?
  fi
  sync_stage prepare-release "$stage_status" 2

  stage_status=0
  set +e
  "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes "$PET_NNODES" --nproc-per-node "$PET_NPROC_PER_NODE" \
    --node-rank "$PET_NODE_RANK" --master-addr "$MASTER_ADDR" \
    --master-port "$((base_port + 3))" \
    "$ROOT/scripts/data/cache_predictive_world_features.py" \
      --distributed --index "$ARTIFACT_ROOT/predictive_world_index.jsonl" \
      --provenance "$ARTIFACT_ROOT/feature_provenance.json" \
      --model "$QWEN_MODEL" --output "$ARTIFACT_ROOT/feature_cache" \
      --legacy-artifact-root "$LEGACY_ARTIFACT_ROOT" \
      --batch-size "$FEATURE_CACHE_BATCH_SIZE"
  stage_status=$?
  set -e
  sync_stage feature-cache "$stage_status" 4

  stage_status=0
  if (( PET_NODE_RANK == 0 )); then
    resolve_run || stage_status=$?
  fi
  sync_stage resolve-configs "$stage_status" 5

  run_stage() {
    local label=$1 config=$2 port_offset=$3 mode=$4 expected_updates=$5
    local output status=0 resume_path=
    output=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["output_dir"])' "$config")
    if [[ "$mode" == preflight ]] && preflight_passed "$output/memory_preflight_report.json"; then
      echo "stage=$label status=reused"
      sync_stage "$label" 0 "$((port_offset + 1))"
      return
    fi
    if [[ "$mode" == train ]] && complete_training "$output/.train_complete" "$expected_updates"; then
      echo "stage=$label status=already-complete"
      sync_stage "$label" 0 "$((port_offset + 1))"
      return
    fi
    if [[ "$mode" == train ]]; then
      resume_path=$(latest_resume "$output")
      [[ -z "$resume_path" ]] || echo "stage=$label auto_resume=$resume_path"
    fi
    set +e
    env -u MICRO_BATCH_SIZE -u GRADIENT_ACCUMULATION_STEPS -u EFFECTIVE_GLOBAL_BATCH \
      CONFIG="$config" NNODES="$PET_NNODES" NPROC_PER_NODE="$PET_NPROC_PER_NODE" \
      NODE_RANK="$PET_NODE_RANK" MASTER_ADDR="$MASTER_ADDR" \
      MASTER_PORT="$((base_port + port_offset))" \
      NUM_WORKERS_PER_RANK="$NUM_WORKERS_PER_RANK" \
      MEMORY_PREFLIGHT_ONLY="$([[ "$mode" == preflight ]] && echo 1 || echo 0)" \
      PREFLIGHT_UPDATES="$PREFLIGHT_UPDATES" \
      MINIMUM_MEMORY_HEADROOM_GIB="$MINIMUM_MEMORY_HEADROOM_GIB" \
      RESUME="$resume_path" \
      bash "$ROOT/scripts/train/predictive_world/launch-5n40g.sh"
    status=$?
    set -e
    sync_stage "$label" "$status" "$((port_offset + 1))"
  }

  run_stage world-preflight "$WORLD_PREFLIGHT_CONFIG" 10 preflight "$WORLD_UPDATES"
  run_stage world-model "$WORLD_CONFIG" 20 train "$WORLD_UPDATES"
  run_stage joint-preflight "$JOINT_PREFLIGHT_CONFIG" 30 preflight "$JOINT_UPDATES"
  run_stage joint "$JOINT_CONFIG" 40 train "$JOINT_UPDATES"
  (( PET_NODE_RANK != 0 )) || echo "training_complete=$JOINT_RUN_ROOT/joint/joint-final"
  exit 0
fi

resolve_run
if [[ "$DRY_RUN" != 1 ]]; then
  command -v sbatch >/dev/null || {
    echo "neither an Inspire PET allocation nor Slurm sbatch was detected" >&2
    echo "use this script as the startup command of a 5-worker x 8-H100 task" >&2
    exit 2
  }
fi

submit_job() {
  local label=$1 dependency=$2 config=$3 memory_preflight=$4 expected_updates=$5
  local output
  output=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["output_dir"])' "$config")
  if [[ "$memory_preflight" == 1 ]] && preflight_passed "$output/memory_preflight_report.json"; then
    echo "stage=$label status=reused"
    SUBMITTED_JOB_ID=$dependency
    return
  fi
  if [[ "$memory_preflight" == 0 ]] && complete_training "$output/.train_complete" "$expected_updates"; then
    echo "stage=$label status=already-complete"
    SUBMITTED_JOB_ID=$dependency
    return
  fi
  local -a dependency_arg=()
  [[ -z "$dependency" ]] || dependency_arg+=(--dependency="afterok:$dependency")
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY_RUN sbatch stage=%s dependency=%s config=%s preflight=%s\n' \
      "$label" "${dependency:-none}" "$config" "$memory_preflight"
    SUBMITTED_JOB_ID="dry-$label"
    return
  fi
  local result
  result=$(env -u MICRO_BATCH_SIZE -u GRADIENT_ACCUMULATION_STEPS -u EFFECTIVE_GLOBAL_BATCH \
    sbatch --parsable --job-name="pw-$label" "${dependency_arg[@]}" \
    --export=ALL,PROJECT_ROOT="$ROOT",CONFIG="$config",RESUME=,MEMORY_PREFLIGHT_ONLY="$memory_preflight",PREFLIGHT_UPDATES="$PREFLIGHT_UPDATES",MINIMUM_MEMORY_HEADROOM_GIB="$MINIMUM_MEMORY_HEADROOM_GIB",NUM_WORKERS_PER_RANK="$NUM_WORKERS_PER_RANK" \
    "$ROOT/scripts/train/predictive_world/pretrain-5n40g.slurm.sh")
  SUBMITTED_JOB_ID=${result%%;*}
  [[ "$SUBMITTED_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "could not parse sbatch job id from: $result" >&2
    exit 2
  }
}

submit_job world-preflight "" "$WORLD_PREFLIGHT_CONFIG" 1 "$WORLD_UPDATES"
world_preflight_job=$SUBMITTED_JOB_ID
submit_job world-model "$world_preflight_job" "$WORLD_CONFIG" 0 "$WORLD_UPDATES"
world_job=$SUBMITTED_JOB_ID
submit_job joint-preflight "$world_job" "$JOINT_PREFLIGHT_CONFIG" 1 "$JOINT_UPDATES"
joint_preflight_job=$SUBMITTED_JOB_ID
submit_job joint "$joint_preflight_job" "$JOINT_CONFIG" 0 "$JOINT_UPDATES"
joint_job=$SUBMITTED_JOB_ID
printf 'run_root=%s\njoint_run_root=%s\nworld_preflight_job=%s\nworld_model_job=%s\njoint_preflight_job=%s\njoint_job=%s\n' \
  "$RUN_ROOT" "$JOINT_RUN_ROOT" "$world_preflight_job" "$world_job" "$joint_preflight_job" "$joint_job"
