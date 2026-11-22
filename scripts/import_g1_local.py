"""Restore excluded G1 SDK/assets from the exact supplied archive, without execution.

This is a local compatibility import, not a redistribution license grant.
Only entries in SOURCE_MANIFEST.json/local_dependencies can be written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1] / 'real/g1'
PREFIXES = (
    'external_dependencies/',
    'GR00T-WholeBodyControl/external_dependencies/',
    'GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/',
    'GR00T-WholeBodyControl/gear_sonic/data/',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restore(archive: Path, root: Path = ROOT, *, apply=False):
    root = root.resolve(strict=True)
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    if digest(archive.read_bytes()) != manifest['archive_sha256']:
        raise ValueError('Archive SHA256 differs from the reviewed source')
    planned = []
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        # Preflight every member before any writes. Never extractall.
        for record in manifest['local_dependencies']:
            name = record['path']
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                    or not name.startswith(PREFIXES)):
                raise ValueError('Invalid local dependency path')
            destination = root / name
            if (not destination.resolve().is_relative_to(root)
                    or any(p.is_symlink() for p in (destination, *destination.parents))):
                raise ValueError('Local dependency symlink is not allowed')
            info = source.getinfo('g1_collection_deployment/' + name)
            if stat.S_ISLNK(info.external_attr >> 16) or info.file_size > 10_000_000:
                raise ValueError('Invalid archive member type or size')
            data = source.read(info)
            if digest(data) != record['sha256']:
                raise ValueError('Local dependency hash mismatch: ' + name)
            if destination.exists():
                if not destination.is_file() or digest(destination.read_bytes()) != record['sha256']:
                    raise FileExistsError('Refusing to overwrite local dependency: ' + name)
            else:
                planned.append((destination, data))
    if apply:
        for destination, data in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as output:
                output.write(data)
    return {'missing_files': len(planned), 'written_files': len(planned) if apply else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--apply', action='store_true', help='Write verified missing files; default is read-only')
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, apply=args.apply), indent=2))


if __name__ == '__main__':
    main()
