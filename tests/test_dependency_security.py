"""Real filesystem regressions for the locked cache/download dependencies."""

import os
from pathlib import Path
import tempfile
import zipfile

import pytest
import requests.utils
from filelock import Timeout, UnixFileLock


@pytest.mark.skipif(os.name != "posix", reason="Unix lock implementation")
def test_lock_symlink_does_not_truncate_target(tmp_path):
    target = tmp_path / "sentinel"
    target.write_bytes(b"existing cache contents")
    lock_path = tmp_path / "cache.lock"
    lock_path.symlink_to(target)

    try:
        with UnixFileLock(lock_path, timeout=0):
            pass
    except (OSError, Timeout):
        pass

    assert target.read_bytes() == b"existing cache contents"


def test_zip_helper_does_not_trust_preexisting_temp_file(tmp_path, monkeypatch):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/fixture.pem", b"legitimate archive contents")
    extraction_dir = tmp_path / "extraction"
    extraction_dir.mkdir()
    stale = extraction_dir / "fixture.pem"
    stale.write_bytes(b"untrusted preexisting contents")
    monkeypatch.setattr(tempfile, "tempdir", str(extraction_dir))

    extracted = requests.utils.extract_zipped_paths(
        str(archive / "nested" / "fixture.pem")
    )

    assert Path(extracted).read_bytes() == b"legitimate archive contents"
    assert stale.read_bytes() == b"untrusted preexisting contents"
