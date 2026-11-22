#!/usr/bin/env bash
# Fast, resumable HE + Psi0-Real + HIW V3 manifest/index/statistics rebuild.
# The final anchor JSONL is written once. Independent validation/index and
# state/action statistics passes run concurrently.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-python}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$ROOT/data/pretraining-artifacts}
FT_ROOT=${FT_ROOT:-$ROOT/data/humanoidarena}
: "${SOURCE_SPEC:?Set SOURCE_SPEC to the audited source manifest}"
: "${RUNTIME_SPEC:?Set RUNTIME_SPEC to the audited source manifest}"
: "${ACTION_PROTOCOLS_SPEC:?Set ACTION_PROTOCOLS_SPEC to the audited source manifest}"
PROTOCOL_VERSION=${PROTOCOL_VERSION:-v3}
SEED=${SEED:-101}
CANDIDATE_PHASES=${CANDIDATE_PHASES:-1}
VISUAL_HISTORY_SECONDS=${VISUAL_HISTORY_SECONDS:-0.0}
ANCHOR_HZ=${ANCHOR_HZ:-2}
FUTURE_SECONDS=${FUTURE_SECONDS:-0.6}
ACTION_STEPS=${ACTION_STEPS:-30}
PROGRESS_EVERY_SECONDS=${PROGRESS_EVERY_SECONDS:-10}
PROGRESS_EVERY_ROWS=${PROGRESS_EVERY_ROWS:-100000}

[[ "$PROTOCOL_VERSION" == v3 ]] || { echo "fast HIW rebuild requires PROTOCOL_VERSION=v3" >&2; exit 2; }
for value in "$SOURCE_SPEC" "$RUNTIME_SPEC" "$ACTION_PROTOCOLS_SPEC"; do
  [[ -s "$value" ]] || { echo "missing configuration: $value" >&2; exit 2; }
done
mkdir -p "$ARTIFACT_ROOT/episodes"
cd "$ROOT"

exec 8>"$ARTIFACT_ROOT/.rebuild-fast.lock"
flock -n 8 || { echo "another fast rebuild owns $ARTIFACT_ROOT" >&2; exit 2; }
child_pids=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${child_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${child_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

manifest="$ARTIFACT_ROOT/pretrain_split_manifest.jsonl"
freeze="$ARTIFACT_ROOT/dataset_freeze.json"
audit="$ARTIFACT_ROOT/cross_stage_overlap_audit.json"
index="$ARTIFACT_ROOT/anchor_index.sqlite"
state_statistics="$ARTIFACT_ROOT/state_statistics.json"
action_statistics="$ARTIFACT_ROOT/action_statistics.json"
complete_marker="$ARTIFACT_ROOT/rebuild_complete.json"

contract_args=(
  "$SOURCE_SPEC" "$RUNTIME_SPEC" "$ACTION_PROTOCOLS_SPEC" "$PROTOCOL_VERSION"
  "$SEED" "$CANDIDATE_PHASES" "$VISUAL_HISTORY_SECONDS" "$ANCHOR_HZ"
  "$FUTURE_SECONDS" "$ACTION_STEPS"
)

# A completed marker is created only after every immutable artifact agrees on
# the same manifest identity. Trusting it avoids reparsing a 13-GiB JSONL on
# retries while still rejecting changed configs or incomplete prior runs.
if [[ -s "$complete_marker" && -s "$manifest" && -s "$freeze" && -s "$audit" \
      && -s "$index" && -s "$state_statistics" && -s "$action_statistics" ]]; then
  if "$PYTHON_BIN" - "$complete_marker" "$freeze" "$audit" "$index" \
      "$state_statistics" "$action_statistics" "${contract_args[@]}" <<'PY'
import hashlib, json, sqlite3, sys
from pathlib import Path

marker, freeze_path, audit_path, index_path, state_path, action_path = map(Path, sys.argv[1:7])
source_spec, runtime_spec, action_spec, protocol, seed, phases, history, hz, future, steps = sys.argv[7:]
contract = {
    "source_spec_sha256": hashlib.sha256(Path(source_spec).read_bytes()).hexdigest(),
    "runtime_spec_sha256": hashlib.sha256(Path(runtime_spec).read_bytes()).hexdigest(),
    "action_protocols_sha256": hashlib.sha256(Path(action_spec).read_bytes()).hexdigest(),
    "protocol_version": protocol, "seed": int(seed), "candidate_phases": int(phases),
    "visual_history_seconds": float(history), "anchor_hz": float(hz),
    "future_seconds": float(future), "action_steps": int(steps),
}
saved = json.loads(marker.read_text())
frozen = json.loads(freeze_path.read_text())
audit = json.loads(audit_path.read_text())
state = json.loads(state_path.read_text())
action = json.loads(action_path.read_text())
with sqlite3.connect(index_path) as connection:
    index_meta = dict(connection.execute("SELECT key,value FROM metadata"))
identity = str(frozen.get("manifest_sha256", ""))
valid = (
    bool(identity) and saved.get("contract") == contract
    and saved.get("manifest_sha256") == identity
    and audit.get("passed") is True and not audit.get("findings")
    and audit.get("pt_manifest_sha256") == identity
    and index_meta.get("manifest_sha256") == identity
    and state.get("dataset_manifest_sha256") == identity
    and action.get("dataset_manifest_sha256") == identity
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "fast_rebuild status=reused artifact_root=$ARTIFACT_ROOT"
    exit 0
  fi
fi

stage_started=$SECONDS
echo "fast_rebuild stage=episode_manifests status=started"
"$PYTHON_BIN" - "$SOURCE_SPEC" <<'PY'
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

mapfile -t source_rows < <("$PYTHON_BIN" - "$SOURCE_SPEC" <<'PY'
import json, sys
from pathlib import Path
for index, row in enumerate(json.loads(Path(sys.argv[1]).read_text())["sources"]):
    print("\t".join((str(index), row["source_id"], row["root"], row["revision"], row["primary_image_key"])))
PY
)

episode_files=()
episode_pids=()
for record in "${source_rows[@]}"; do
  IFS=$'\t' read -r source_index source_id source_root revision image_key <<<"$record"
  slug=$(printf '%02d-%s' "$source_index" "${source_id//\//_}")
  output="$ARTIFACT_ROOT/episodes/$slug.jsonl"
  episode_files+=("$output")
  (
    [[ -s "$source_root/meta/info.json" ]] || { echo "missing source metadata: $source_root" >&2; exit 2; }
    mapfile -t image_args < <("$PYTHON_BIN" - "$SOURCE_SPEC" "$source_index" <<'PY'
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
  ) &
  episode_pids+=("$!")
  child_pids+=("$!")
done
episode_status=0
for pid in "${episode_pids[@]}"; do
  wait "$pid" || episode_status=$?
done
(( episode_status == 0 )) || { echo "episode manifest worker failed: $episode_status" >&2; exit "$episode_status"; }

PYTHONPATH=src "$PYTHON_BIN" scripts/data/build_pretrain_registry.py \
  "${episode_files[@]}" --output "$ARTIFACT_ROOT/episode_split_manifest.jsonl" \
  --dataset-freeze "$ARTIFACT_ROOT/episode_freeze.json" --seed "$SEED" \
  --train-fraction 0.9 --val-fraction 0.05
echo "fast_rebuild stage=episode_manifests status=passed elapsed_seconds=$((SECONDS-stage_started))"

# This small independent inventory can overlap the expensive anchor materialization.
PYTHONPATH=src "$PYTHON_BIN" scripts/data/build_downstream_exclusion_manifest.py \
  --root "$FT_ROOT" --output "$ARTIFACT_ROOT/downstream_exclusion_manifest.parquet" --seed "$SEED" &
downstream_pid=$!
child_pids+=("$downstream_pid")

stage_started=$SECONDS
echo "fast_rebuild stage=anchors status=started output=$manifest"
PYTHONPATH=src "$PYTHON_BIN" scripts/data/materialize_pretrain_anchors.py \
  --episode-manifest "$ARTIFACT_ROOT/episode_split_manifest.jsonl" \
  --episode-freeze "$ARTIFACT_ROOT/episode_freeze.json" \
  --output "$manifest" --dataset-freeze "$freeze" \
  --anchor-hz "$ANCHOR_HZ" --future-seconds "$FUTURE_SECONDS" \
  --protocol-version "$PROTOCOL_VERSION" --candidate-phases "$CANDIDATE_PHASES" \
  --visual-history-seconds "$VISUAL_HISTORY_SECONDS" --action-steps "$ACTION_STEPS" \
  --progress-every-seconds "$PROGRESS_EVERY_SECONDS"
wait "$downstream_pid"
echo "fast_rebuild stage=anchors status=passed elapsed_seconds=$((SECONDS-stage_started))"

stage_started=$SECONDS
echo "fast_rebuild stage=audit_and_index status=started mode=fused_single_scan"
PYTHONPATH=src "$PYTHON_BIN" scripts/data/build_pretrain_anchor_index.py \
  --anchor-manifest "$manifest" --output "$index" \
  --progress-every-rows "$PROGRESS_EVERY_ROWS" \
  --audit-downstream-exclusion-manifest "$ARTIFACT_ROOT/downstream_exclusion_manifest.parquet" \
  --audit-output "$audit" --forbidden-root "$FT_ROOT"
echo "fast_rebuild stage=audit_and_index status=passed elapsed_seconds=$((SECONDS-stage_started))"

stage_started=$SECONDS
echo "fast_rebuild stage=state_and_action_statistics status=started mode=parallel"
PYTHONPATH=src "$PYTHON_BIN" scripts/data/fit_pretrain_state_statistics.py \
  --anchor-manifest "$manifest" --anchor-index "$index" --trust-frozen-index \
  --dataset-freeze "$freeze" --source-configs "$RUNTIME_SPEC" \
  --output "$state_statistics" &
state_pid=$!
child_pids+=("$state_pid")
PYTHONPATH=src "$PYTHON_BIN" scripts/data/fit_pretrain_action_statistics.py \
  --anchor-manifest "$manifest" --anchor-index "$index" --trust-frozen-index \
  --dataset-freeze "$freeze" --source-configs "$RUNTIME_SPEC" \
  --action-protocols "$ACTION_PROTOCOLS_SPEC" --output "$action_statistics" &
action_pid=$!
child_pids+=("$action_pid")
(
  heartbeat_started=$SECONDS
  while kill -0 "$state_pid" 2>/dev/null || kill -0 "$action_pid" 2>/dev/null; do
    sleep "$PROGRESS_EVERY_SECONDS"
    if kill -0 "$state_pid" 2>/dev/null || kill -0 "$action_pid" 2>/dev/null; then
      echo "fast_rebuild stage=state_and_action_statistics status=running elapsed_seconds=$((SECONDS-heartbeat_started))"
    fi
  done
) &
statistics_heartbeat_pid=$!
child_pids+=("$statistics_heartbeat_pid")
parallel_status=0
wait "$state_pid" || parallel_status=$?
wait "$action_pid" || parallel_status=$?
kill "$statistics_heartbeat_pid" 2>/dev/null || true
wait "$statistics_heartbeat_pid" || true
(( parallel_status == 0 )) || { echo "parallel statistics failed: $parallel_status" >&2; exit "$parallel_status"; }
echo "fast_rebuild stage=state_and_action_statistics status=passed elapsed_seconds=$((SECONDS-stage_started))"

"$PYTHON_BIN" - "$complete_marker" "$freeze" "${contract_args[@]}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
marker, freeze_path = map(Path, sys.argv[1:3])
source_spec, runtime_spec, action_spec, protocol, seed, phases, history, hz, future, steps = sys.argv[3:]
contract = {
    "source_spec_sha256": hashlib.sha256(Path(source_spec).read_bytes()).hexdigest(),
    "runtime_spec_sha256": hashlib.sha256(Path(runtime_spec).read_bytes()).hexdigest(),
    "action_protocols_sha256": hashlib.sha256(Path(action_spec).read_bytes()).hexdigest(),
    "protocol_version": protocol, "seed": int(seed), "candidate_phases": int(phases),
    "visual_history_seconds": float(history), "anchor_hz": float(hz),
    "future_seconds": float(future), "action_steps": int(steps),
}
payload = {
    "format_version": "predictor_pretrain_fast_rebuild_v1",
    "contract": contract,
    "manifest_sha256": json.loads(freeze_path.read_text())["manifest_sha256"],
}
temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(marker)
PY
echo "fast_rebuild status=passed artifact_root=$ARTIFACT_ROOT"
