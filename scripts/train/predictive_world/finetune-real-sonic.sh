#!/usr/bin/env bash
# Final G1/BrainCo recipe: body auxiliary + FULL+ERT v5, state41/motion76.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash this-script.sh [--prepare-only|--dry-run|--preflight-only]
# BATCH_SIZE is per GPU. All four checkpoints (5k/10k/15k/20k) are retained.
# Request 12 CPU cores per GPU on the execution platform separately.
# Each GPU rank uses 12 training loader workers or 12 cache decoder workers.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/../../.." && pwd -P)
python_bin=${PYTHON_BIN:-$repo_root/.venv-posttrain/bin/python}
dataset=${DATASET:-}
gpus=${GPUS:-4}
batch=${BATCH_SIZE:-256}
run_dir=${OUTPUT_DIR:-}
init_joint=${INIT_JOINT:-$repo_root/models/predictive_world/joint-15000}
source_config=${SOURCE_CONFIG:-$repo_root/posttrain_assets/source/joint-15000.source-config.json}
asset_map=${ASSET_PATH_MAP:-$repo_root/posttrain_assets/source/joint-15000.local-assets.json}
migration=${PROCESSOR_MIGRATION:-$script_dir/place-doll-processor-migration-batch256.local.json}
accumulation=${GRAD_ACCUMULATION:-1}
cache_batch=${CACHE_BATCH_SIZE:-256}
cache_decode_workers=${CACHE_DECODE_WORKERS:-12}
num_workers=${NUM_WORKERS:-12}
prefetch_factor=${PREFETCH_FACTOR:-1}
parquet_cache_capacity=${PARQUET_CACHE_CAPACITY:-2}
video_cache_capacity=${VIDEO_CACHE_CAPACITY:-8}
cache_decode_prefetch=${CACHE_DECODE_PREFETCH_BATCHES:-0}
dataset_format=${DATASET_FORMAT:-real_sonic}
init_mode=${INIT_MODE:-pretrained}
action_horizon=${ACTION_HORIZON:-}
if [[ -z "$action_horizon" ]]; then
  if [[ "$init_mode" == gr00t_n1d7 ]]; then
    action_horizon=40
  elif [[ "$dataset_format" == sonic_lerobot ]]; then
    action_horizon=50
  else
    action_horizon=30
  fi
fi
replacement_qwen=${QWEN_MODEL:-}
gr00t_model=${GR00T_MODEL:-}
updates=${UPDATES:-20000}
rtc_max_delay=${RTC_TRAINING_MAX_DELAY:-15}
mode=${1:-train}
case "$mode" in
  train|--prepare-only|--audit-only|--cache-only|--preflight-only|--dry-run) ;;
  --help|-h)
    echo "Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=256 bash $0 [--prepare-only|--audit-only|--cache-only|--preflight-only|--dry-run]"
    echo "Defaults: 20000 optimizer steps, save every 5000, retain all 4; GPUS=4, BATCH_SIZE=256 per GPU (global 1024), GRAD_ACCUMULATION=1."
    echo "CPU defaults per GPU rank: NUM_WORKERS=12, PREFETCH_FACTOR=1; cache batch 256, CACHE_DECODE_WORKERS=12, CACHE_DECODE_PREFETCH_BATCHES=0 (auto: twice the decoder worker count). CPU core allocation is configured on the execution platform."
    echo "RTC_TRAINING_MAX_DELAY=$rtc_max_delay: uniformly sampled inclusive prefix range 0..$rtc_max_delay on complete action windows."
    echo "Optional: INIT_MODE (pretrained|scratch|gr00t_n1d7), GR00T_MODEL, UPDATES, DATASET, OUTPUT_DIR, INIT_JOINT, PYTHON_BIN, NUM_WORKERS, PREFETCH_FACTOR, PARQUET_CACHE_CAPACITY, VIDEO_CACHE_CAPACITY, CACHE_DECODE_WORKERS, CACHE_DECODE_PREFETCH_BATCHES, CACHE_BATCH_SIZE, RESUME (complete checkpoint directory)."
    echo "Collector v3 additionally requires SNAPSHOT_MAX_AGE_MS: an explicit training/deployment cutoff separate from the unbounded source contract."
    echo "Native SONIC LeRobot exports: DATASET_FORMAT=sonic_lerobot, QWEN_MODEL=<frozen VLM>, ACTION_HORIZON=50 (or 40 for INIT_MODE=gr00t_n1d7); action frequency follows recorded 30Hz rows."
    exit 0 ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
: "${DATASET:?Set DATASET to the extracted real G1/BrainCo LeRobot dataset}"
: "${OUTPUT_DIR:?Set a unique OUTPUT_DIR for this dataset and run}"
if (( $# > 1 )); then echo "Only one optional mode is supported" >&2; exit 2; fi
for value in "$batch" "$gpus" "$accumulation" "$cache_batch" "$prefetch_factor" "$parquet_cache_capacity" "$video_cache_capacity" "$updates"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo "Batch/GPU/accumulation/prefetch/update budgets must be positive integers" >&2; exit 2; fi
done
if [[ ! "$num_workers" =~ ^[0-9]+$ ]]; then echo "NUM_WORKERS must be nonnegative" >&2; exit 2; fi
if [[ ! "$cache_decode_workers" =~ ^[0-9]+$ ]]; then echo "CACHE_DECODE_WORKERS must be nonnegative" >&2; exit 2; fi
if [[ ! "$cache_decode_prefetch" =~ ^[0-9]+$ ]]; then echo "CACHE_DECODE_PREFETCH_BATCHES must be nonnegative" >&2; exit 2; fi
if [[ ! "$rtc_max_delay" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "RTC_TRAINING_MAX_DELAY must be a nonnegative integer" >&2; exit 2; fi
if [[ ! "$action_horizon" =~ ^[1-9][0-9]*$ ]] || (( rtc_max_delay >= action_horizon )); then
  echo "ACTION_HORIZON must exceed the RTC prefix" >&2; exit 2
fi
case "$init_mode" in
  pretrained|scratch|gr00t_n1d7) ;;
  *) echo "Unsupported INIT_MODE=$init_mode" >&2; exit 2 ;;
esac
case "$dataset_format" in
  real_sonic)
    if [[ "$init_mode" == pretrained ]]; then
      [[ "$action_horizon" == 30 && -z "$replacement_qwen" ]] || { echo "Use DATASET_FORMAT=sonic_lerobot or INIT_MODE=scratch|gr00t_n1d7 for VLM/action replacement" >&2; exit 2; }
    else
      : "${QWEN_MODEL:?Set QWEN_MODEL to the frozen Qwen3-VL or exported GR00T N1.7 VLM directory}"
    fi ;;
  sonic_lerobot)
    : "${QWEN_MODEL:?Set QWEN_MODEL to the replacement frozen Psi0/Qwen3-VL or exported GR00T N1.7 VLM directory}" ;;
  *) echo "Unsupported DATASET_FORMAT=$dataset_format" >&2; exit 2 ;;
esac
if [[ "$init_mode" == gr00t_n1d7 ]]; then
  : "${GR00T_MODEL:?Set GR00T_MODEL to the local GR00T N1.7 base directory}"
  [[ "$action_horizon" == 40 ]] || { echo "INIT_MODE=gr00t_n1d7 requires ACTION_HORIZON=40" >&2; exit 2; }
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
cd -- "$repo_root"
required_paths=("$python_bin" "$dataset/meta/info.json")
if [[ "$init_mode" == pretrained ]]; then
  required_paths+=("$source_config" "$asset_map" "$init_joint/.complete")
  if [[ "$dataset_format" == real_sonic ]]; then required_paths+=("$migration"); else required_paths+=("$replacement_qwen/config.json"); fi
else
  required_paths+=("$replacement_qwen/config.json")
  if [[ "$init_mode" == gr00t_n1d7 ]]; then required_paths+=("$gr00t_model/config.json"); fi
fi
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then echo "Required asset missing: $path" >&2; exit 1; fi
done
if [[ ! -x "$python_bin" ]]; then echo "Python is not executable: $python_bin" >&2; exit 1; fi

run() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$mode" != --dry-run ]]; then "$@"; fi
}

# Lock preparation and training together; distinct OUTPUT_DIRs remain independent.
if [[ "$mode" != --dry-run && "$mode" != --audit-only ]]; then
  mkdir -p -- "$run_dir"
  exec 9>"$run_dir/.launcher.lock"
  flock -n 9 || { echo "Another launcher is using $run_dir" >&2; exit 1; }
fi
if [[ "$dataset_format" == real_sonic ]]; then prepare_entry=prepare_real_sonic_task.py; else prepare_entry=prepare_sonic_lerobot_task.py; fi
prepare=("$python_bin" "$script_dir/$prepare_entry"
  --dataset "$dataset" --task "${TASK_ID:-real/place_doll}" --output "$run_dir"
  --gpus "$gpus" --batch "$batch" --accumulation "$accumulation"
  --updates "$updates" --checkpoint-every 5000 --first-checkpoint-update 5000 --checkpoint-retention 4
  --validation-fraction 0 --validation-every 0 --rtc-training-max-delay "$rtc_max_delay"
  --training-variant FULL+ERT
  --current-state-only --fixed-action-grid)
if [[ "$init_mode" == pretrained ]]; then
  prepare+=(--init-joint "$init_joint" --source-config "$source_config" --asset-path-map "$asset_map")
else
  prepare+=(--init-mode "$init_mode" --qwen-model "$replacement_qwen" --action-horizon "$action_horizon")
  if [[ "$init_mode" == gr00t_n1d7 ]]; then
    prepare+=(--gr00t-model "$gr00t_model")
    if [[ "${GR00T_ACTIVATION_CHECKPOINTING:-0}" == 1 ]]; then
      prepare+=(--gr00t-activation-checkpointing)
    fi
  fi
fi
if [[ "${POSTTRAIN_ADAPTATION:-0}" == 1 ]]; then
  prepare+=(--whole-body-target --content-area-pooling)
  if [[ "$init_mode" == pretrained ]]; then
    prepare+=(--latent-calibration-batches "${LATENT_CALIBRATION_BATCHES:-16}")
  fi
elif [[ "${POSTTRAIN_ADAPTATION:-0}" != 0 ]]; then
  echo "POSTTRAIN_ADAPTATION must be 0 or 1" >&2; exit 2
fi
if [[ "$dataset_format" == real_sonic ]]; then
  if [[ "$init_mode" == pretrained ]]; then
    prepare+=(--processor-migration "$migration")
  fi
  prepare+=(--independent-future-components
    --collector-snapshots --recorded-action-projection --real-sample-filter collector_snapshot_v1)
else
  if [[ "$init_mode" == pretrained ]]; then
    prepare+=(--qwen-model "$replacement_qwen" --action-horizon "$action_horizon")
    if [[ -n "${PROCESSOR_MIGRATION:-}" ]]; then
      prepare+=(--processor-migration "$PROCESSOR_MIGRATION")
    fi
  fi
fi
if [[ -n "${SNAPSHOT_MAX_AGE_MS:-}" ]]; then
  prepare+=(--snapshot-max-age-ms "$SNAPSHOT_MAX_AGE_MS")
fi
if [[ -n "${EXCLUDE_EPISODES:-}" ]]; then
  read -r -a excluded_episode_ids <<< "${EXCLUDE_EPISODES//,/ }"
  for episode in "${excluded_episode_ids[@]}"; do
    [[ "$episode" =~ ^[0-9]+$ ]] || { echo "EXCLUDE_EPISODES requires original nonnegative indices" >&2; exit 2; }
    prepare+=(--exclude-episode "$episode")
  done
fi
if [[ "$mode" == --audit-only ]]; then run "${prepare[@]}" --audit-only; exit 0; fi
run "${prepare[@]}"
if [[ "$mode" == --prepare-only ]]; then exit 0; fi

if [[ -n "$replacement_qwen" ]]; then
  qwen_model=$replacement_qwen
else
qwen_model=$("$python_bin" - "$source_config" "$asset_map" <<'PY'
import json
from pathlib import Path
import sys
from psi.data.predictive_world.assets import load_asset_path_map, resolve_asset_path
source = json.loads(Path(sys.argv[1]).read_text())
print(resolve_asset_path(source['model']['qwen_model'], load_asset_path_map(sys.argv[2])))
PY
)
fi
runner=("$python_bin")
if (( gpus > 1 )); then
  runner+=(-m torch.distributed.run --standalone --nproc_per_node "$gpus")
fi
cache=("${runner[@]}" "$repo_root/scripts/data/cache_predictive_world_features.py"
  --index "$run_dir/artifacts/predictive_world_index.jsonl"
  --provenance "$run_dir/artifacts/feature_provenance.json" --model "$qwen_model"
  --output "$run_dir/artifacts/feature_cache" --batch-size "$cache_batch" --decode-workers "$cache_decode_workers"
  --decode-prefetch-batches "$cache_decode_prefetch")
if (( gpus > 1 )); then cache+=(--distributed); fi
run "${cache[@]}"
if [[ "$mode" == --cache-only ]]; then exit 0; fi
if [[ "$mode" != --dry-run && -d "$run_dir/preflight" ]]; then
  mkdir -p -- "$run_dir/preflight-attempts"
  mv -- "$run_dir/preflight" "$run_dir/preflight-attempts/$(date +%s%N)"
fi
run "${runner[@]}" "$script_dir/train.py" --config "$run_dir/preflight.json" --num-workers "$num_workers" --prefetch-factor "$prefetch_factor" \
  --parquet-cache-capacity "$parquet_cache_capacity" --video-cache-capacity "$video_cache_capacity" \
  --micro-batch "$batch" --gradient-accumulation-steps "$accumulation" \
  --effective-global-batch "$((batch * gpus * accumulation))" --preflight-only --preflight-updates 3
if [[ "$mode" == --preflight-only ]]; then exit 0; fi
train=("${runner[@]}" "$script_dir/train.py" --config "$run_dir/config.json" --num-workers "$num_workers" --prefetch-factor "$prefetch_factor" \
  --parquet-cache-capacity "$parquet_cache_capacity" --video-cache-capacity "$video_cache_capacity")
if [[ -n "${RESUME:-}" ]]; then train+=(--resume "$RESUME"); fi
run "${train[@]}"
