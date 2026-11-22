#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
: "${WORLD_MODEL_CONFIG:?WORLD_MODEL_CONFIG must point to the resolved world-model config}"
: "${JOINT_CONFIG:?JOINT_CONFIG must point to the resolved joint-training config}"
[[ -f "$WORLD_MODEL_CONFIG" ]] || { echo "missing WORLD_MODEL_CONFIG: $WORLD_MODEL_CONFIG" >&2; exit 2; }
[[ -f "$JOINT_CONFIG" ]] || { echo "missing JOINT_CONFIG: $JOINT_CONFIG" >&2; exit 2; }

WORLD_MODEL_CONFIG=$(realpath "$WORLD_MODEL_CONFIG")
JOINT_CONFIG=$(realpath "$JOINT_CONFIG")
PYTHON_BIN=${PYTHON_BIN:-python}

"$PYTHON_BIN" - "$WORLD_MODEL_CONFIG" "$JOINT_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

world_path, joint_path = (Path(value).resolve() for value in sys.argv[1:])
world = json.loads(world_path.read_text(encoding="utf-8"))
joint = json.loads(joint_path.read_text(encoding="utf-8"))

if world.get("route") != "predictive_world" or joint.get("route") != "predictive_world":
    raise SystemExit("both configs must use route=predictive_world")
if world.get("training", {}).get("phase") != "world_model":
    raise SystemExit("WORLD_MODEL_CONFIG must use training.phase=world_model")
if joint.get("training", {}).get("phase") != "joint":
    raise SystemExit("JOINT_CONFIG must use training.phase=joint")
if world.get("model") != joint.get("model"):
    raise SystemExit("world-model and joint model configs must match exactly")
if world.get("data") != joint.get("data"):
    raise SystemExit("world-model and joint data configs must match exactly")

world_output = Path(world.get("runtime", {}).get("output_dir", "")).resolve()
joint_output = Path(joint.get("runtime", {}).get("output_dir", "")).resolve()
expected_bundle = world_output / "world-model-final"
configured_bundle = Path(joint.get("training", {}).get("world_model", "")).resolve()
if configured_bundle != expected_bundle:
    raise SystemExit(
        "joint training.world_model must equal " + str(expected_bundle)
    )
if joint_output == world_output or world_output in joint_output.parents:
    raise SystemExit("joint output must be separate from the world-model output")
if joint_output in world_output.parents:
    raise SystemExit("world-model output must be separate from the joint output")
PY

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'validated world_model=%s\nvalidated joint=%s\n' \
    "$WORLD_MODEL_CONFIG" "$JOINT_CONFIG"
  exit 0
fi

world_submission=$(sbatch --parsable \
  --export=ALL,PROJECT_ROOT="$ROOT",CONFIG="$WORLD_MODEL_CONFIG",RESUME=,MEMORY_PREFLIGHT_ONLY=0 \
  "$ROOT/scripts/train/predictive_world/pretrain-5n40g.slurm.sh")
world_job_id=${world_submission%%;*}
[[ "$world_job_id" =~ ^[0-9]+$ ]] || {
  echo "unexpected world-model sbatch response: $world_submission" >&2
  exit 1
}

if ! joint_submission=$(sbatch --parsable --dependency="afterok:$world_job_id" \
  --export=ALL,PROJECT_ROOT="$ROOT",CONFIG="$JOINT_CONFIG",RESUME=,MEMORY_PREFLIGHT_ONLY=0 \
  "$ROOT/scripts/train/predictive_world/pretrain-5n40g.slurm.sh"); then
  echo "world-model job $world_job_id was submitted, but joint submission failed" >&2
  exit 1
fi
joint_job_id=${joint_submission%%;*}
[[ "$joint_job_id" =~ ^[0-9]+$ ]] || {
  echo "world-model job $world_job_id was submitted; unexpected joint sbatch response: $joint_submission" >&2
  exit 1
}

printf 'world_model_job=%s\njoint_job=%s dependency=afterok:%s\n' \
  "$world_job_id" "$joint_job_id" "$world_job_id"
