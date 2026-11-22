"""Exercise local integration boundaries without hardware, downloads or GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts.check_g1_sources import verify
from scripts.import_g1_local import restore
from scripts.export_source import source_files

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / 'real/g1'


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_and_runtime_identities():
    report = verify()
    assert not report['findings']
    assert len(report['profiles']) == 5
    assert {p['files'] for p in report['profiles']} == {46}


def test_bundled_runtime_retains_its_preprocessing_reference():
    require_runtime = load(G1 / 'scripts/local_vla_compat.py').require_preprocessing_runtime
    manifest = json.loads((G1 / 'SOURCE_MANIFEST.json').read_text())
    profiles = [entry for entry in manifest['files']
                if entry['path'].startswith('runtime_profiles/')
                and entry['path'].endswith('/COMPATIBILITY.json')]
    assert len(profiles) == 5
    for index, entry in enumerate(profiles):
        kit = (G1 / entry['path']).parent
        require_runtime(kit, entry['sha256'])
        require_runtime(kit, entry['source_sha256'])
        with pytest.raises(ValueError, match='different runtime'):
            require_runtime(kit, profiles[index - 1]['source_sha256'])


def test_launcher_help_works_before_local_configuration(tmp_path):
    script = tmp_path / 'run.py'
    script.write_bytes((G1 / 'run.py').read_bytes())
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'deploy-model' in result.stdout and 'teleop' in result.stdout
    assert not (tmp_path / 'config').exists()


def test_configure_never_overwrites_and_does_not_copy_calibration(tmp_path):
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/local.example.json').write_bytes((G1 / 'config/local.example.json').read_bytes())
    configure = load(G1 / 'configure_local_paths.py').configure
    destination = configure(tmp_path)
    value = json.loads(destination.read_text())
    assert value['sonic_root'] == str(tmp_path / 'GR00T-WholeBodyControl')
    assert value['robot_host'] == 'REPLACE_ROBOT_HOST'
    assert not (tmp_path / 'config/hand-hardware.json').exists()
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        configure(tmp_path)
    assert destination.read_bytes() == before


def make_archive(tmp_path, name='external_dependencies/sdk/header.h'):
    root = tmp_path / 'g1'
    root.mkdir()
    archive = tmp_path / 'source.zip'
    payload = b'reviewed local SDK source'
    with zipfile.ZipFile(archive, 'w') as source:
        source.writestr('g1_collection_deployment/' + name, payload)
    manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'local_dependencies': [{'path': name, 'sha256': hashlib.sha256(payload).hexdigest()}]}
    (root / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest))
    return root, archive, payload, manifest


def test_import_is_read_only_by_default_and_idempotent(tmp_path):
    root, archive, payload, _ = make_archive(tmp_path)
    assert restore(archive, root) == {'missing_files': 1, 'written_files': 0}
    assert list(root.iterdir()) == [root / 'SOURCE_MANIFEST.json']
    assert restore(archive, root, apply=True)['written_files'] == 1
    assert (root / 'external_dependencies/sdk/header.h').read_bytes() == payload
    assert restore(archive, root, apply=True)['written_files'] == 0


def test_import_refuses_mismatched_archive_without_writes(tmp_path):
    root, archive, _, _ = make_archive(tmp_path)
    archive.write_bytes(archive.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Archive SHA256'):
        restore(archive, root, apply=True)
    assert not (root / 'external_dependencies').exists()


@pytest.mark.parametrize('name', ['../outside', '/outside', 'scripts/source.py',
                                  'external_dependencies/../../outside'])
def test_import_cannot_write_outside_allowlist(tmp_path, name):
    root, archive, _, _ = make_archive(tmp_path, name)
    with pytest.raises(ValueError, match='path'):
        restore(archive, root, apply=True)
    assert len(list(root.iterdir())) == 1


def test_import_refuses_symlink_and_modified_dependency(tmp_path):
    root, archive, _, _ = make_archive(tmp_path)
    (root / 'external_dependencies').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        restore(archive, root, apply=True)
    (root / 'external_dependencies').unlink()
    restore(archive, root, apply=True)
    file = root / 'external_dependencies/sdk/header.h'
    file.write_text('local changes')
    with pytest.raises(FileExistsError):
        restore(archive, root, apply=True)
    assert file.read_text() == 'local changes'


def test_teleop_dry_run_needs_no_local_config_or_robot_modules():
    result = subprocess.run([sys.executable, str(G1 / 'scripts/start_sonic.py'), '--dry-run'],
                            cwd='/tmp', capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout)['robot_commands_sent'] is False


def test_each_profile_loads_verified_transport_in_an_isolated_process():
    for profile in ('legacy', 'clean_table_trash', 'place_chair_20261029',
                    'blow_dry_hair_20261029', 'tidy_scene'):
        code = ('import online_runtime; online_runtime.import_runtime(); '
                'import psi; from pathlib import Path; '
                'assert Path(psi.__file__).is_relative_to(online_runtime.KIT); '
                'assert "torch" not in __import__("sys").modules')
        result = subprocess.run([sys.executable, '-I', '-c',
                                 'import sys; sys.path.insert(0, ' + repr(str(G1 / 'scripts')) + '); ' + code],
                                env={**os.environ, 'CHENYZ_POLICY_PROFILE': profile},
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr


def test_episode_success_cannot_override_a_timing_gap():
    quality = load(G1 / 'scripts/episode_quality.py')
    lifecycle = quality.EpisodeLifecycle(task_success=True)
    lifecycle.frame(0, 1.0, 1 / 30)
    lifecycle.frame(1, 1.2, 1 / 30)
    lifecycle.end('save')
    result = quality.build_episode_quality(lifecycle, True, {'runtime_verified': True})
    assert result['storage_valid'] is True
    assert result['training_eligible'] is False
    assert 'time_gap_present' in result['quality_exclusions']


def test_missing_body_capture_entrypoint_is_restored(tmp_path):
    names = ('target/release/g1_deploy_onnx_ref', 'policy/release/model_decoder.onnx',
             'policy/release/model_encoder.onnx', 'policy/release/observation_config.yaml',
             'planner/target_vel/V2/planner_sonic.onnx')
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic fixture')
    output = tmp_path / 'identity.json'
    subprocess.run([sys.executable, str(G1 / 'scripts/capture_body_runtime.py'),
                    '--pid', '123', '--deployment-root', str(tmp_path), '--output', str(output)],
                   check=True, timeout=10)
    manifest = json.loads(output.read_text())
    assert manifest['pid'] == 123
    assert set(manifest['artifacts']) == set(names)


def test_every_g1_source_is_in_the_release_selection():
    manifest = json.loads((G1 / 'SOURCE_MANIFEST.json').read_text())
    selected = {path.as_posix() for path in source_files(ROOT)}
    assert all('real/g1/' + entry['path'] in selected for entry in manifest['files'])


def test_restored_local_dependencies_and_private_settings_stay_ignored():
    names = (
        'real/g1/.local/models/model.py',
        'real/g1/config/local.json',
        'real/g1/config/hand-hardware.json',
        'real/g1/external_dependencies/sdk/header.h',
        'real/g1/GR00T-WholeBodyControl/external_dependencies/sdk/header.h',
        'real/g1/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/sdk/header.h',
        'real/g1/GR00T-WholeBodyControl/gear_sonic/data/robot/mesh.stl',
    )
    result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'],
                            input='\n'.join(names) + '\n', cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert set(result.stdout.splitlines()) == set(names)
