#!/usr/bin/env bash
set -euo pipefail
: "${SONIC_ROOT:?Set SONIC_ROOT to the existing GR00T-WholeBodyControl checkout}"
if [[ ! -f "$SONIC_ROOT/gear_sonic_deploy/deploy.sh" ]]; then
    echo "Missing existing SONIC deployment launcher: $SONIC_ROOT/gear_sonic_deploy/deploy.sh" >&2
    exit 1
fi
for argument in "$@"; do
    if [[ "$argument" == --velocity-fallback ]]; then
        sonic_launcher_supports_fallback=false
        while IFS= read -r sonic_launcher_line; do
            if [[ "$sonic_launcher_line" =~ ^[[:space:]]*--velocity-fallback\) ]]; then
                sonic_launcher_supports_fallback=true
                break
            fi
        done < "$SONIC_ROOT/gear_sonic_deploy/deploy.sh"
        if [[ "$sonic_launcher_supports_fallback" != true ]]; then
            echo "SONIC deploy.sh does not forward --velocity-fallback; apply real/patches/sonic_velocity_fallback_ack_v1.patch to the SONIC checkout first." >&2
            exit 1
        fi
    fi
done
exec bash "$SONIC_ROOT/gear_sonic_deploy/deploy.sh" --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 "$@"
