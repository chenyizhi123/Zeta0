"""Install verified public G1 SDK/geometry files; no private ZIP is needed.

The source checkout must match third_party/sources.toml. LFS objects are fetched
only for manifest-listed robot geometry and verified against original SHA256.
Vendor executables and model weights are never installed by this command.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

try:
    from scripts.check_source_pins import check_source, tomllib
except ModuleNotFoundError:
    from check_source_pins import check_source, tomllib

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ('GR00T-WholeBodyControl/external_dependencies/',
            'GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/',
            'GR00T-WholeBodyControl/gear_sonic/data/')
LIMIT = 10_000_000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('Unsafe dependency path')
    result = root / name
    if not result.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in (result, *result.parents)):
        raise ValueError('Unsafe dependency path or symlink')
    return result


def read_public(record, g1, source, entry):
    name = record['path']
    if record.get('overlay'):
        data = safe_path(g1, record['overlay']).read_bytes()
    else:
        relative = name.removeprefix('GR00T-WholeBodyControl/')
        data = safe_path(source, relative).read_bytes()
        if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            if b'oid sha256:' + record['sha256'].encode() not in data:
                raise ValueError('LFS identity differs from the supplied source')
            # Build the public URL from the pinned, credential-free GitHub origin.
            prefix = 'https://github.com/'
            if not entry['url'].startswith(prefix):
                raise ValueError('LFS downloads require a public GitHub source')
            repo = entry['url'][len(prefix):].removesuffix('.git')
            url = f"https://media.githubusercontent.com/media/{repo}/{entry['commit']}/{quote(relative)}"
            with urlopen(url, timeout=60) as response:
                data = response.read(LIMIT + 1)
    if len(data) > LIMIT or sha(data) != record['sha256']:
        raise ValueError('Dependency checksum mismatch: ' + name)
    return data


def install(root=ROOT, *, apply=False):
    g1 = root / 'real/g1'
    manifest = json.loads((g1 / 'PUBLIC_DEPENDENCIES.json').read_text())
    with (root / 'third_party/sources.toml').open('rb') as handle:
        entry = tomllib.load(handle)[manifest['source']]
    errors = check_source(root, entry)
    if errors:
        raise ValueError('Run fetch_sources.py first: ' + '; '.join(errors))
    source = root / entry['path']
    missing = []
    for record in manifest['files']:
        if not record['path'].startswith(PREFIXES):
            raise ValueError('Dependency outside the G1 install boundary')
        target = safe_path(g1, record['path'])
        if target.exists():
            if not target.is_file() or sha(target.read_bytes()) != record['sha256']:
                raise FileExistsError('Refusing to replace local dependency: ' + record['path'])
        else:
            missing.append(record)
    if apply:
        # Finish all identity/network checks before writing any destination.
        with ThreadPoolExecutor(max_workers=4) as pool:
            contents = list(pool.map(lambda r: read_public(r, g1, source, entry), missing))
        for record, data in zip(missing, contents):
            target = safe_path(g1, record['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(data)
    return {'missing_files': len(missing), 'written_files': len(missing) if apply else 0,
            'requires_private_archive': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(install(apply=args.apply), indent=2))
