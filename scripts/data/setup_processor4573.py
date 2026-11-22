#!/usr/bin/env python3
"""Create a new diagnostic-only Transformers overlay, preserving the base venv."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--uv', default='uv')
    args = parser.parse_args()
    # Preserve the venv executable path: resolving its symlink would select the
    # system interpreter's site-packages instead of the requested base venv.
    base_python = args.base_python.absolute()
    target = args.output.absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'refusing to modify an existing environment: {target}')
    probe = 'import json, sys, sysconfig; print(json.dumps([sys.prefix, sys.base_prefix, sysconfig.get_path("purelib")]))'
    prefix, base_prefix, packages = json.loads(subprocess.check_output(
        [str(base_python), '-c', probe], text=True
    ))
    if prefix == base_prefix:
        raise ValueError('base-python must identify an existing virtual environment')
    subprocess.run([args.uv, 'venv', '--python', str(base_python), str(target)], check=True)
    python = target/'bin/python'
    # Install before exposing inherited packages so no installer can uninstall
    # or replace anything in the shared base environment.
    lock = Path(__file__).with_name('processor4573.requirements.txt')
    subprocess.run([args.uv, 'pip', 'install', '--python', str(python), '--no-deps',
                    '--require-hashes', '-r', str(lock)], check=True)
    _, _, new_packages = json.loads(subprocess.check_output([str(python), '-c', probe], text=True))
    with (Path(new_packages)/'zz_training_readonly.pth').open('x') as handle:
        handle.write(packages+'\n')
    print(f'Created diagnostic overlay: {python}; read-only dependency base: {packages}')


if __name__ == '__main__':
    main()
