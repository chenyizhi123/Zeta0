"""Check real Git acquisition, public dependency boundaries and the runnable demo."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.fetch_sources import acquire
from scripts.setup_g1_dependencies import install, safe_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def public_source(tmp_path):
    origin = tmp_path / 'upstream'
    origin.mkdir()
    subprocess.run(['git', 'init', '-q', str(origin)], check=True)
    (origin / 'sdk').mkdir()
    (origin / 'sdk/header.h').write_bytes(b'public SDK source')
    (origin / 'LICENSE').write_text('Synthetic test license')
    subprocess.run(['git', '-C', str(origin), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(origin), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', '-c', 'core.hooksPath=/dev/null',
                    'commit', '-qm', 'fixture'], check=True)
    commit = subprocess.check_output(['git', '-C', str(origin), 'rev-parse', 'HEAD'], text=True).strip()
    root = tmp_path / 'release'
    root.mkdir()
    return root, {'url': origin.as_uri(), 'commit': commit, 'path': 'third_party/upstream'}


def test_fetch_plan_then_exact_checkout_and_preserve_changes(public_source):
    root, entry = public_source
    assert acquire(root, entry)['status'] == 'planned'
    assert not list(root.iterdir())
    assert acquire(root, entry, apply=True)['status'] == 'acquired and verified'
    assert acquire(root, entry, apply=True)['status'] == 'already verified'
    local = root / entry['path'] / 'sdk/header.h'
    local.write_text('local changes')
    with pytest.raises(ValueError, match='local changes'):
        acquire(root, entry, apply=True)
    assert local.read_text() == 'local changes'


def test_public_install_preflights_all_files_and_never_overwrites(public_source):
    root, entry = public_source
    acquire(root, entry, apply=True)
    (root / 'third_party/sources.toml').write_text(
        '[WBC]\n' + '\n'.join(f'{k} = {json.dumps(v)}' for k, v in entry.items()) + '\n')
    g1 = root / 'real/g1'
    g1.mkdir(parents=True)
    target = 'GR00T-WholeBodyControl/external_dependencies/sdk/header.h'
    files = [{'path': target, 'overlay': 'header.h',
              'sha256': hashlib.sha256(b'public SDK source').hexdigest()}]
    (g1 / 'header.h').write_bytes(b'public SDK source')
    (g1 / 'PUBLIC_DEPENDENCIES.json').write_text(json.dumps({'source': 'WBC', 'files': files}))
    assert install(root)['written_files'] == 0
    assert not (g1 / target).exists()
    assert install(root, apply=True)['written_files'] == 1
    assert install(root, apply=True)['written_files'] == 0
    (g1 / target).write_text('my changes')
    with pytest.raises(FileExistsError):
        install(root, apply=True)
    assert (g1 / target).read_text() == 'my changes'


def test_fetch_failure_cleans_temporary_checkout(public_source):
    root, entry = public_source
    with pytest.raises(RuntimeError):
        acquire(root, {**entry, 'commit': '0' * 40}, apply=True)
    assert not (root / entry['path']).exists()
    assert not list((root / 'third_party').iterdir())


def test_dependency_paths_reject_symlinks_and_traversal(tmp_path):
    for name in ('../outside', '/outside', 'sdk/../../outside'):
        with pytest.raises(ValueError):
            safe_path(tmp_path, name)
    (tmp_path / 'sdk').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_path(tmp_path, 'sdk/header.h')


def test_quickstart_runs_from_another_directory_and_preserves_existing_output(tmp_path):
    output = tmp_path / 'demo'
    command = [sys.executable, str(ROOT / 'scripts/examples/quickstart.py'), '--output', str(output)]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, timeout=60)
    report = json.loads((output / 'summary.json').read_text())
    assert report['completed_updates'] == 8 and report['resumed_at'] == 4
    assert report['final_mse'] < report['initial_mse']
    assert report['robot_commands_sent'] is False
    before = (output / 'predictor.safetensors').read_bytes()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, timeout=60)
    assert result.returncode != 0
    assert (output / 'predictor.safetensors').read_bytes() == before
