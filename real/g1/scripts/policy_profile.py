"""Select an isolated runtime before any psi modules are imported."""
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
def _kit(name):
    local = ROOT / '.local/zeta0' / name
    return local if local.is_dir() else ROOT / 'runtime_profiles' / name


LEGACY_KIT = _kit('zeta0_deploy_offline_kit_20261024')
CLEAN_KIT = _kit('clean_table_runtime_local_v1')
CHAIR_KIT = _kit('place_chair_runtime_20261029')
TASKS = ('flatten_box', 'tidy_table_trash', 'place_chair', 'clean_table_trash', 'blow_dry_hair', 'place_tape_on_table', 'tidy_scene')


def runtime_kit(profile=None):
    profile = profile or os.environ.get('CHENYZ_POLICY_PROFILE', 'legacy')
    kits = {'legacy': LEGACY_KIT, 'clean_table_trash': CLEAN_KIT, 'place_chair_20261029': CHAIR_KIT}
    kits['blow_dry_hair_20261029'] = _kit('blow_dry_hair_runtime_20261029')
    kits['tidy_scene'] = _kit('tidy_scene_runtime_local_v1')
    if profile not in kits:
        raise ValueError('Unknown local policy profile: ' + profile)
    return kits[profile]


def model_paths(task, override=None):
    if not isinstance(task, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', task):
        raise ValueError('Task must be a simple name containing letters, numbers, underscores or hyphens')
    custom = task not in TASKS
    if custom and override is None:
        raise ValueError('A custom task requires --model-paths with an explicit runtime_profile')
    defaults = {
        'clean_table_trash': '.local/models/clean_table_trash/model_paths.json',
        'place_chair': '.local/models/place_chair/model_paths.json',
        'blow_dry_hair': '.local/models/blow_dry_hair/model_paths.json',
        'place_tape_on_table': '.local/models/place_tape_on_table/model_paths.json',
        'tidy_scene': '.local/models/tidy_scene/model_paths.json',
    }
    default = ROOT / defaults.get(task, '.local/models/zeta0/model_paths.json')
    path = Path(override or default).expanduser().resolve()
    paths = json.loads(path.read_text())
    if not isinstance(paths, dict) or not isinstance(paths.get('tasks'), dict) or task not in paths['tasks']:
        raise ValueError('Model-path configuration does not contain the selected task: ' + task)
    if custom:
        if paths.get('format') != 'local_zeta0_model_paths.v1' or not paths.get('runtime_profile'):
            raise ValueError('Custom tasks require format=local_zeta0_model_paths.v1 and an explicit runtime_profile')
        entry = paths['tasks'][task]
        if not isinstance(entry, dict) or not isinstance(entry.get('task'), str) or not entry['task'].strip():
            raise ValueError('Custom task requires a nonempty instruction')
        if (entry.get('action_hz'), entry.get('action_horizon')) not in {(30, 50), (50, 30)}:
            raise ValueError('Custom task requires a supported, explicit action_hz/action_horizon')
        for target, key in ((paths, 'qwen_model'), (entry, 'weights_dir')):
            value = target.get(key)
            if not isinstance(value, str) or not value.strip() or 'REPLACE_' in value:
                raise ValueError('Custom task requires a real path for ' + key)
            target[key] = str((path.parent / Path(value).expanduser()).resolve())
    profile = paths.get('runtime_profile', 'clean_table_trash' if task == 'clean_table_trash' else 'legacy')
    previous = os.environ.get('CHENYZ_POLICY_PROFILE', 'legacy')
    if previous != profile and ('online_runtime' in sys.modules or 'psi' in sys.modules):
        raise RuntimeError('Use a separate process to switch policy runtimes')
    kit = runtime_kit(profile)  # Reject unknown profiles before changing process state.
    if custom:
        entry = paths['tasks'][task]
        manifest = json.loads((Path(entry['weights_dir']) / 'inference_manifest.json').read_text())
        compatibility = json.loads((kit / 'COMPATIBILITY.json').read_text())
        temporal = manifest.get('temporal_abi', {})
        if (manifest.get('task') != entry['task'] or any(
                temporal.get(key) != entry[key] or compatibility.get(key) != entry[key]
                for key in ('action_hz', 'action_horizon'))):
            raise ValueError('Custom task instruction/action grid disagrees with weights or selected runtime')
    os.environ['CHENYZ_POLICY_PROFILE'] = profile
    return paths
