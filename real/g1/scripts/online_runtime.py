"""Load the verified unified or legacy runtime without importing model/GPU dependencies."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
from policy_profile import runtime_kit

KIT = runtime_kit()


def import_runtime():
    compatibility = json.loads((KIT / 'COMPATIBILITY.json').read_text())
    if compatibility.get('runtime_mode') == 'repository_src':
        runtime_root = (ROOT.parents[1] / 'src').resolve()
        expected_entry = runtime_root / 'psi/deploy/real_sonic_runtime.py'
        if not expected_entry.is_file():
            raise ValueError('Unified repository runtime is missing: ' + str(expected_entry))
    else:
        expected = compatibility['runtime_files_sha256']
        for relative, digest in expected.items():
            path = KIT / relative
            if not path.resolve().is_relative_to(KIT.resolve()):
                raise ValueError('Runtime source path escaped the verified kit')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Runtime source checksum mismatch: ' + relative)
        runtime_root = (KIT / 'runtime').resolve()
    for name, module in list(sys.modules.items()):
        if name == 'psi' or name.startswith('psi.'):
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(runtime_root):
                raise RuntimeError('Another model runtime is already imported: ' + name)
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    return runtime_root
