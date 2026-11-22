"""Build a reproducible source tarball with portable ownership and timestamps."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

try:
    from scripts.export_source import export_source
except ModuleNotFoundError:
    from export_source import export_source


def package_source(root: Path, output: Path):
    root = root.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Archive output must be a new file')
    if output.resolve().is_relative_to(root):
        raise ValueError('Archive must be outside the repository')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='zeta0-package-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'source'
        manifest = export_source(root, staged)
        archive_path = Path(temporary) / 'source.tar.gz'
        with archive_path.open('wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                    for path in sorted(p for p in staged.rglob('*') if p.is_file()):
                        data = path.read_bytes()
                        member = tarfile.TarInfo('zeta0-source/' + path.relative_to(staged).as_posix())
                        member.size = len(data)
                        member.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                        member.uid = member.gid = 0
                        member.uname = member.gname = ''
                        member.mtime = 0
                        archive.addfile(member, io.BytesIO(data))
        # Publish only a completed archive; do not replace a concurrent output.
        os.link(archive_path, output)
    return {'source_files': len(manifest['files']), 'archive': str(output),
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_source(Path(__file__).resolve().parents[1], args.output), indent=2))
