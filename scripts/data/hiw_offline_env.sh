#!/usr/bin/env bash
# Source this helper. Compute-node bootstrap must never access a package index.
: "${PYTHON_BIN:?PYTHON_BIN is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
HIW_WHEELHOUSE=${HIW_WHEELHOUSE:-$OUTPUT_ROOT/offline-wheels}
_hiw_python_tag=$("$PYTHON_BIN" -c 'import sys; print(sys.implementation.cache_tag)')
_hiw_dependency_dir="$OUTPUT_ROOT/deps/$_hiw_python_tag"
mkdir -p "$_hiw_dependency_dir"
export PYTHONPATH="$_hiw_dependency_dir${PYTHONPATH:+:$PYTHONPATH}"
_hiw_check_dependencies() {
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import lz4.frame
import zstandard
expected = {"mcap": "1.4.0", "mcap-ros2-support": "0.5.7",
            "lz4": "4.4.5", "zstandard": "0.25.0"}
for name, version in expected.items():
    actual = metadata.version(name)
    if actual != version:
        raise RuntimeError(f"{name}: expected {version}, found {actual}")
payload = b"HIW offline compression smoke"
assert lz4.frame.decompress(lz4.frame.compress(payload)) == payload
assert zstandard.ZstdDecompressor().decompress(zstandard.ZstdCompressor().compress(payload)) == payload
print("HIW offline dependencies OK: " + str(expected), flush=True)
PY
}
(
  flock -x 9 || exit $?
  if ! _hiw_check_dependencies >/dev/null 2>&1; then
    [[ -d "$HIW_WHEELHOUSE" ]] || {
      echo "Missing shared offline wheels: $HIW_WHEELHOUSE; no network fallback is permitted." >&2
      exit 2
    }
    "$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || {
      echo "HIW bootstrap requires pip in PYTHON_BIN; prepare the data-bootstrap dependency group before going offline. No network fallback is permitted." >&2
      exit 2
    }
    echo "Installing HIW dependencies OFFLINE: python=$_hiw_python_tag wheels=$HIW_WHEELHOUSE"
    PIP_CONFIG_FILE=/dev/null PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
      "$PYTHON_BIN" -m pip install --no-index --no-deps --only-binary=:all: \
      --find-links "$HIW_WHEELHOUSE" --upgrade --target "$_hiw_dependency_dir" \
      'mcap==1.4.0' 'mcap-ros2-support==0.5.7' 'lz4==4.4.5' 'zstandard==0.25.0' || exit $?
  fi
  _hiw_check_dependencies
) 9>"$OUTPUT_ROOT/deps/.install-$_hiw_python_tag.lock"
