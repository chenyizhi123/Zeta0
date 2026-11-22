"""Acquire exact optional public sources; default prints a plan without writes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

try:
    from scripts.check_source_pins import check_source, tomllib
except ModuleNotFoundError:
    from check_source_pins import check_source, tomllib

ROOT = Path(__file__).resolve().parents[1]


def acquire(root, entry, *, apply=False):
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts or str(relative) == '.':
        raise ValueError('Invalid source path')
    destination = root / relative
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError('Source path cannot traverse symlinks')
    if not re.fullmatch(r'[0-9a-f]{40}', entry['commit']):
        raise ValueError('A full source commit is required')
    if destination.exists():
        errors = check_source(root, entry)
        if errors:
            raise ValueError('Existing checkout rejected: ' + '; '.join(errors))
        return {'path': str(relative), 'status': 'already verified'}
    if not apply:
        return {'path': str(relative), 'status': 'planned', 'commit': entry['commit']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.source-fetch-', dir=destination.parent))
    env = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1', 'GIT_TERMINAL_PROMPT': '0'}
    try:
        commands = [
            ['clone', '--filter=blob:none', '--no-checkout', '--no-tags', entry['url'], str(temporary)],
            ['-C', str(temporary), 'fetch', '--depth=1', 'origin', entry['commit']],
        ]
        if entry.get('sparse'):
            commands.extend([
                ['-C', str(temporary), 'sparse-checkout', 'init', '--cone'],
                ['-C', str(temporary), 'sparse-checkout', 'set', *entry['sparse']],
            ])
        commands.append(['-C', str(temporary), 'checkout', '--detach', '-q', 'FETCH_HEAD'])
        for args in commands:
            result = subprocess.run(['git', *args], capture_output=True, env=env, timeout=600)
            if result.returncode:
                raise RuntimeError('Source acquisition failed; existing checkouts were preserved')
        # Verify under the temporary name before publishing it.
        errors = check_source(root, {**entry, 'path': str(temporary.relative_to(root))})
        if errors:
            raise ValueError('; '.join(errors))
        if destination.exists():
            raise FileExistsError('Source destination appeared during fetch')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'path': str(relative), 'status': 'acquired and verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='+')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'third_party/sources.toml').open('rb') as source:
        sources = tomllib.load(source)
    if set(args.names) - sources.keys():
        parser.error('Unknown source name')
    for name in args.names:
        print(json.dumps({'source': name, **acquire(ROOT, sources[name], apply=args.apply)}))


if __name__ == '__main__':
    main()
