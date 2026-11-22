#!/usr/bin/env bash
# HE + nine Psi0-Real + the immutable 21,447-episode HIW release.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
export ARTIFACT_ROOT=${ARTIFACT_ROOT:-$ROOT/data/pretraining-artifacts}
: "${SOURCE_SPEC:?Set SOURCE_SPEC to the audited source manifest}"
: "${RUNTIME_SPEC:?Set RUNTIME_SPEC to the audited source manifest}"
: "${ACTION_PROTOCOLS_SPEC:?Set ACTION_PROTOCOLS_SPEC to the audited source manifest}"
export PROTOCOL_VERSION=v3
export CANDIDATE_PHASES=1
export VISUAL_HISTORY_SECONDS=0.0
exec bash "$ROOT/scripts/data/rebuild_he_psi0_hiw_v3_fast.sh"
