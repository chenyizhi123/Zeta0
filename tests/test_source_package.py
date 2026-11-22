"""Check portable tar ownership, immutable output and byte reproducibility."""
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts.package_source import package_source


def test_portable_reproducible_source_archive_and_manifest(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root / '.gitignore').write_text('.local/\n')
    (root / 'run.sh').write_text('#!/bin/sh\nexit 0\n')
    (root / 'run.sh').chmod(0o755)
    (root / '.local').mkdir()
    (root / '.local/private.txt').write_text('local input')
    first = tmp_path / 'first.tar.gz'
    report = package_source(root, first)
    assert report['source_files'] == 2
    with tarfile.open(first) as archive:
        members = archive.getmembers()
        assert {m.name for m in members} == {
            'zeta0-source/.gitignore', 'zeta0-source/run.sh', 'zeta0-source/source_manifest.json'}
        assert all(m.uid == m.gid == m.mtime == 0 and not m.uname and not m.gname for m in members)
        assert archive.getmember('zeta0-source/run.sh').mode == 0o755
        assert archive.extractfile('zeta0-source/run.sh').read() == (root / 'run.sh').read_bytes()
    os.utime(root / 'run.sh', (123456, 123456))
    second = tmp_path / 'second.tar.gz'
    assert package_source(root, second)['sha256'] == report['sha256']
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        package_source(root, first)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(ValueError):
        package_source(root, root / 'nested.tar.gz')
