#!/usr/bin/env bash
# Dedicated capture environment; leaves SONIC's control environments untouched.
set -euo pipefail
CHENYZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CHENYZ_ROOT"
CHENYZ_LEROBOT_COMMIT=7e241bd630a3719a56157a497ce5d08f244784f1
if ! command -v uv >/dev/null; then
  echo "需要 uv：请先按 https://docs.astral.sh/uv/getting-started/installation/ 安装。" >&2
  exit 1
fi
if [[ ! -d external_dependencies/lerobot/.git ]]; then
  git clone --branch v0.6.1 --depth 1 https://github.com/huggingface/lerobot.git external_dependencies/lerobot
fi
if [[ "$(git -C external_dependencies/lerobot rev-parse HEAD)" != "$CHENYZ_LEROBOT_COMMIT" ]] ||
   [[ -n "$(git -C external_dependencies/lerobot status --porcelain)" ]]; then
  echo "官方 LeRobot 源码需为未修改的 v0.6.1 ($CHENYZ_LEROBOT_COMMIT)；请检查 external_dependencies/lerobot。" >&2
  exit 1
fi
if [[ ! -x .venv_lerobot/bin/python ]]; then
  uv venv --python 3.12 .venv_lerobot
fi
uv pip sync --python .venv_lerobot/bin/python scripts/requirements-lerobot.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
# PyAV bundles FFmpeg 7. Expose its libraries to torchcodec without changing
# the system FFmpeg or installing a second ABI into SONIC's environments.
uv run --no-project --python .venv_lerobot/bin/python python - <<'PY'
from pathlib import Path
import re
import sysconfig

packages=Path(sysconfig.get_paths()['purelib'])
libs=packages/'av.libs'
aliases=Path('.venv_lerobot/lib/ffmpeg')
aliases.mkdir(exist_ok=True)
for source in libs.glob('lib*.so.*'):
    if not source.name.startswith(('libav','libsw')):
        continue
    name=re.sub(r'-[0-9a-f]+(?=\.so)','',source.name)
    name=name.split('.so.')[0]+'.so.'+name.split('.so.')[1].split('.')[0]
    target=aliases/name
    if target.is_symlink():target.unlink()
    target.symlink_to(source)
PY
source scripts/lerobot_env.sh
uv run --no-project --python .venv_lerobot/bin/python python - <<'PY'
import importlib.metadata
import sys
import pinocchio
import torchcodec
from lerobot.configs.video import rgb_encoder_defaults
from lerobot_robot_sonic import SonicRobotConfig, SonicActionsConfig
import deepdiff

assert importlib.metadata.version('lerobot')=='0.6.1'
assert sys.version_info[:2]==(3,12)
encoder=rgb_encoder_defaults()
print('官方 LeRobot 0.6.1 就绪：Python 3.12，AV1 / CRF 30 / GOP 2；SONIC 环境保持独立。')
PY
