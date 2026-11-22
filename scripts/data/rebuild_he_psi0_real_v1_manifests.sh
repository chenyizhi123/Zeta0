#!/usr/bin/env bash
# Rebuild the HE + nine Psi0-Real identities, 90/5/5 episode split, leakage
# audit, anchor index and PT_train-only state statistics. Does not cache images
# or launch training.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-python}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$ROOT/data/pretraining-artifacts}
FT_ROOT=${FT_ROOT:-$ROOT/data/humanoidarena}
SEED=${SEED:-101}
: "${SOURCE_SPEC:?Set SOURCE_SPEC to the audited source manifest}"
: "${RUNTIME_SPEC:?Set RUNTIME_SPEC to the audited source manifest}"
PROTOCOL_VERSION=${PROTOCOL_VERSION:-v1}
CANDIDATE_PHASES=${CANDIDATE_PHASES:-1}
VISUAL_HISTORY_SECONDS=${VISUAL_HISTORY_SECONDS:-0.0}
if [[ "$PROTOCOL_VERSION" != v1 && "$(basename -- "$(realpath -m "$ARTIFACT_ROOT")")" == he_psi0_real_v1 ]]; then
  echo "$PROTOCOL_VERSION must use a separate artifact directory; refusing to overwrite V1" >&2
  exit 2
fi
if [[ "$PROTOCOL_VERSION" != v1 && -s "$ARTIFACT_ROOT/pretrain_split_manifest.jsonl" ]]; then
  "$PYTHON_BIN" - "$ARTIFACT_ROOT/pretrain_split_manifest.jsonl" "$PROTOCOL_VERSION" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.loads(next(line for line in handle if line.strip()))
if row.get("format_version") != f"predictor_pretrain_anchor_{sys.argv[2]}":
    label = sys.argv[2].upper()
    raise SystemExit(f"refusing to overwrite an existing non-{label} artifact directory")
PY
fi

[[ -s "$SOURCE_SPEC" && -s "$RUNTIME_SPEC" ]] || { echo "missing source configuration" >&2; exit 2; }
mkdir -p "$ARTIFACT_ROOT/episodes"
cd "$ROOT"

PYTHONPATH=src "$PYTHON_BIN" - "$SOURCE_SPEC" <<'PY'
import json, sys
from pathlib import Path
for row in json.loads(Path(sys.argv[1]).read_text())["sources"]:
    metadata = Path(row["root"]) / ".hfd/repo_metadata.json"
    if not metadata.is_file():
        continue
    actual = json.loads(metadata.read_text()).get("sha")
    if actual and actual != row["revision"]:
        raise SystemExit(
            f"local snapshot revision mismatch for {row['source_id']}: "
            f"configured={row['revision']} local={actual}"
        )
PY

mapfile -t source_rows < <(PYTHONPATH=src "$PYTHON_BIN" - "$SOURCE_SPEC" <<'PY'
import json, sys
from pathlib import Path
for index, row in enumerate(json.loads(Path(sys.argv[1]).read_text())["sources"]):
    print("\t".join((str(index), row["source_id"], row["root"], row["revision"], row["primary_image_key"])))
PY
)

episode_files=()
for record in "${source_rows[@]}"; do
  IFS=$'\t' read -r index source_id source_root revision image_key <<<"$record"
  slug=$(printf '%02d-%s' "$index" "${source_id//\//_}")
  output="$ARTIFACT_ROOT/episodes/$slug.jsonl"
  [[ -s "$source_root/meta/info.json" ]] || { echo "missing source metadata: $source_root" >&2; exit 2; }
  mapfile -t image_args < <("$PYTHON_BIN" - "$SOURCE_SPEC" "$index" <<'PY'
import json, sys
from pathlib import Path
row = json.loads(Path(sys.argv[1]).read_text())["sources"][int(sys.argv[2])]
for key in row.get("input_image_keys", []):
    print("--input-image-key=" + key)
for key, role in row.get("camera_roles", {}).items():
    print("--camera-role=" + key + "=" + role)
for field in row.get("group_fields", []):
    print("--group-field=" + field)
if row.get("action_timestamp_column"):
    print("--action-timestamp-column=" + row["action_timestamp_column"])
PY
)
  PYTHONPATH=src "$PYTHON_BIN" scripts/data/build_lerobot_pretrain_manifest.py \
    --root "$source_root" --source-id "$source_id" --revision "$revision" \
    --primary-image-key "$image_key" --episodes-only --output "$output" \
    --protocol-version "$PROTOCOL_VERSION" "${image_args[@]}"
  episode_files+=("$output")
done

PYTHONPATH=src "$PYTHON_BIN" scripts/data/build_pretrain_registry.py \
  "${episode_files[@]}" --output "$ARTIFACT_ROOT/episode_split_manifest.jsonl" \
  --dataset-freeze "$ARTIFACT_ROOT/episode_freeze.json" --seed "$SEED" \
  --train-fraction 0.9 --val-fraction 0.05
PYTHONPATH=src "$PYTHON_BIN" scripts/data/materialize_pretrain_anchors.py \
  --episode-manifest "$ARTIFACT_ROOT/episode_split_manifest.jsonl" \
  --output "$ARTIFACT_ROOT/anchors_unfrozen.jsonl" --anchor-hz 2 --future-seconds 0.6 \
  --protocol-version "$PROTOCOL_VERSION" --candidate-phases "$CANDIDATE_PHASES" \
  --visual-history-seconds "$VISUAL_HISTORY_SECONDS"
PYTHONPATH=src "$PYTHON_BIN" scripts/data/build_pretrain_registry.py \
  "$ARTIFACT_ROOT/anchors_unfrozen.jsonl" --output "$ARTIFACT_ROOT/pretrain_split_manifest.jsonl" \
  --dataset-freeze "$ARTIFACT_ROOT/dataset_freeze.json" --seed "$SEED" \
  --train-fraction 0.9 --val-fraction 0.05
PYTHONPATH=src "$PYTHON_BIN" scripts/data/build_downstream_exclusion_manifest.py \
  --root "$FT_ROOT" --output "$ARTIFACT_ROOT/downstream_exclusion_manifest.parquet" --seed "$SEED"
PYTHONPATH=src "$PYTHON_BIN" scripts/data/audit_pretrain_exclusion.py \
  --pretrain-manifest "$ARTIFACT_ROOT/pretrain_split_manifest.jsonl" \
  --downstream-exclusion-manifest "$ARTIFACT_ROOT/downstream_exclusion_manifest.parquet" \
  --output "$ARTIFACT_ROOT/cross_stage_overlap_audit.json" --forbidden-root "$FT_ROOT"
PYTHONPATH=src "$PYTHON_BIN" scripts/data/build_pretrain_anchor_index.py \
  --anchor-manifest "$ARTIFACT_ROOT/pretrain_split_manifest.jsonl" \
  --output "$ARTIFACT_ROOT/anchor_index.sqlite"
PYTHONPATH=src "$PYTHON_BIN" scripts/data/fit_pretrain_state_statistics.py \
  --anchor-manifest "$ARTIFACT_ROOT/pretrain_split_manifest.jsonl" \
  --anchor-index "$ARTIFACT_ROOT/anchor_index.sqlite" \
  --dataset-freeze "$ARTIFACT_ROOT/dataset_freeze.json" --source-configs "$RUNTIME_SPEC" \
  --output "$ARTIFACT_ROOT/state_statistics.json"
if [[ "$PROTOCOL_VERSION" == v3 ]]; then
  PYTHONPATH=src "$PYTHON_BIN" scripts/data/fit_pretrain_action_statistics.py \
    --anchor-manifest "$ARTIFACT_ROOT/pretrain_split_manifest.jsonl" \
    --anchor-index "$ARTIFACT_ROOT/anchor_index.sqlite" \
    --dataset-freeze "$ARTIFACT_ROOT/dataset_freeze.json" --source-configs "$RUNTIME_SPEC" \
    --action-protocols "${ACTION_PROTOCOLS_SPEC:?ACTION_PROTOCOLS_SPEC is required for V3}" \
    --output "$ARTIFACT_ROOT/action_statistics.json"
fi

echo "HE + Psi0-Real manifests rebuilt: $ARTIFACT_ROOT"
