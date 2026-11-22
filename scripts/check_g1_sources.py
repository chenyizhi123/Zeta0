"""Verify the G1 source overlay and all five runtime identities without hardware."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1] / 'real/g1'


def checked_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('Unsafe manifest path')
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Unsafe source path')
    return path


def verify(root=ROOT):
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    findings = []
    for record in manifest['files']:
        path = checked_path(root, record['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            findings.append({'path': record['path'], 'kind': 'source_hash'})
    profiles = []
    for path in sorted((root / 'runtime_profiles').glob('*/COMPATIBILITY.json')):
        expected = json.loads(path.read_text())['runtime_files_sha256']
        for name, sha in expected.items():
            source = checked_path(path.parent, name)
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != sha:
                findings.append({'path': str(source.relative_to(root)), 'kind': 'runtime_hash'})
        profiles.append({'profile': path.parent.name, 'files': len(expected)})
    if len(profiles) != 5:
        findings.append({'kind': 'profile_count'})
    return {'source_files': len(manifest['files']), 'profiles': profiles, 'findings': findings}


if __name__ == '__main__':
    report = verify()
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report['findings']))
