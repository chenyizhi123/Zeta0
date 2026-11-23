"""Verify the G1 source overlay, unified runtime manifest, and legacy archive index."""
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
        compat = json.loads(path.read_text())
        if compat.get('runtime_mode') == 'repository_src':
            profiles.append({
                'profile': path.parent.name,
                'runtime_mode': 'repository_src',
                'supported_hosts': compat.get('supported_hosts', []),
            })
            continue
        expected = compat['runtime_files_sha256']
        for name, sha in expected.items():
            source = checked_path(path.parent, name)
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != sha:
                findings.append({'path': str(source.relative_to(root)), 'kind': 'runtime_hash'})
        profiles.append({'profile': path.parent.name, 'files': len(expected)})
    if not any(p['profile'] == 'unified' and p.get('runtime_mode') == 'repository_src' for p in profiles):
        findings.append({'kind': 'missing_unified_profile'})
    archive_path = root / 'runtime_profiles/LEGACY_ARCHIVE.json'
    if not archive_path.is_file():
        findings.append({'kind': 'missing_legacy_archive_index'})
    else:
        archive = json.loads(archive_path.read_text())
        if len(archive.get('profiles', [])) != 5:
            findings.append({'kind': 'legacy_archive_profile_count'})
    return {'source_files': len(manifest['files']), 'profiles': profiles, 'findings': findings}


if __name__ == '__main__':
    report = verify()
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report['findings']))
