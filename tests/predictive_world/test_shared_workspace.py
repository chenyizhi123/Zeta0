"""The shared site file resolves relative to itself and preserves user choices."""
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def site_values(tmp_path, overrides=None):
    site = tmp_path / 'workspace.local.env'
    shutil.copy2(ROOT / 'workspace.env.example', site)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('ZETA0_') and key != 'PYTHON_BIN'}
    env.update(overrides or {})
    result = subprocess.check_output([
        'bash', '-c',
        'source "$1"; printf "%s\\n" "$ZETA0_ROOT" "$PYTHON_BIN" '
        '"$ZETA0_INIT_JOINT" "$ZETA0_POSTTRAIN_ASSET_ROOT" '
        '"$ZETA0_OUTPUT_ROOT"',
        'site-check', str(site),
    ], cwd='/', env=env, text=True)
    return result.splitlines()


def test_site_paths_are_local_to_shared_checkout(tmp_path):
    values = site_values(tmp_path)
    assert values == [
        str(tmp_path), str(tmp_path / '.venv-posttrain/bin/python'),
        str(tmp_path / 'models/predictive_world/joint-15000'),
        str(tmp_path / 'posttrain_assets'),
        str(tmp_path / 'posttrain_runs/joint'),
    ]


def test_explicit_site_overrides_are_not_overwritten(tmp_path):
    values = site_values(tmp_path, {
        'PYTHON_BIN': '/allocated/python', 'ZETA0_INIT_JOINT': '/reviewed/joint',
        'ZETA0_POSTTRAIN_ASSET_ROOT': '/shared/assets',
        'ZETA0_OUTPUT_ROOT': '/allocated/output',
    })
    assert values[1:] == ['/allocated/python', '/reviewed/joint', '/shared/assets', '/allocated/output']
