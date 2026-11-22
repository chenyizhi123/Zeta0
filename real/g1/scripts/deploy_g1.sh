#!/usr/bin/env bash
set -euo pipefail
CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$CHENYZ_ROOT/config/hand-hardware.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    hardware = json.load(source)
if hardware.get('deployment_driver_compatible') is not True or hardware.get('service_verified') is not True:
    print('Deployment blocked: BrainCo Revo2 services have not been verified after hand driver adaptation.', file=sys.stderr)
    sys.exit(2)
PY
CHENYZ_SONIC_ROOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sonic_root"])' "$CHENYZ_ROOT/config/local.json")"
CHENYZ_INTERFACE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["robot_interface"])' "$CHENYZ_ROOT/config/local.json")"
source "$CHENYZ_SONIC_ROOT/local_env.sh"
cd "$CHENYZ_SONIC_ROOT/gear_sonic_deploy"
if [[ "${1:-}" != "--help" && "${1:-}" != "-h" ]]; then
    python3 "$CHENYZ_ROOT/scripts/capture_body_runtime.py" \
        --pid "$$" --deployment-root "$PWD" --output "$CHENYZ_ROOT/logs/sonic-body-runtime.json"
fi
exec target/release/g1_deploy_onnx_ref "$CHENYZ_INTERFACE" \
    policy/release/model_decoder.onnx reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 --disable-dex3 \
    --logs-dir "$CHENYZ_ROOT/logs/deploy" "$@"
