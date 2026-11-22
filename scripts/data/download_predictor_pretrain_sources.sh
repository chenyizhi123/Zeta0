#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?usage: download_predictor_pretrain_sources.sh OUTPUT_ROOT}
MAX_WORKERS=${HF_DOWNLOAD_MAX_WORKERS:-16}
mkdir -p "$ROOT/hiw500_lerobot" "$ROOT/humanoid_everyday" "$ROOT/psi0_real"

# Resolved immutable snapshots.  Unset the cluster proxy because these hosts
# are reachable directly here and the configured proxy truncates large files.
download_hiw() {
  until env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
    hf download BitRobot/HIW-500-LeRobot --repo-type dataset \
    --revision 2d32431d3be3b0621c462759b02f7b378e53c2cc \
    --max-workers "$MAX_WORKERS" \
    --local-dir "$ROOT/hiw500_lerobot"; do
    echo "HIW download interrupted; resuming in 10 seconds" >&2
    sleep 10
  done
}
download_he() {
  until env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
    hf download USC-GVL/humanoid-everyday --repo-type dataset \
    --revision 71f6210e91c2a7e8b8e4d47db0147444a1a0640e \
    --max-workers "$MAX_WORKERS" \
    --local-dir "$ROOT/humanoid_everyday"; do
    echo "Humanoid Everyday download interrupted; resuming in 10 seconds" >&2
    sleep 10
  done
}
download_psi0_real() {
  until env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
    hf download USC-PSI-Lab/psi-data --repo-type dataset \
    --revision 92da00fe58b90ecfaaa30a096ff6b9b9090feb26 \
    --max-workers "$MAX_WORKERS" \
    --include 'real/*' --local-dir "$ROOT/psi0_real"; do
    echo "Psi0 real-data download interrupted; resuming in 10 seconds" >&2
    sleep 10
  done
}
download_hiw &
HIW_PID=$!
download_he &
HE_PID=$!
download_psi0_real &
PSI0_REAL_PID=$!
wait "$HIW_PID"
wait "$HE_PID"
wait "$PSI0_REAL_PID"
