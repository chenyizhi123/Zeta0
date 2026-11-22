"""Training provenance must not depend on the launcher's working directory."""
from pathlib import Path
import runpy
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/train/predictive_world/train.py'


def test_git_revision_outside_repository(tmp_path, monkeypatch):
    entrypoint = runpy.run_path(str(SCRIPT))
    expected = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=SCRIPT.parents[3], text=True
    ).strip()
    monkeypatch.chdir(tmp_path)
    assert entrypoint['_git_revision']() == expected


def test_missing_repository_is_not_silently_accepted(tmp_path):
    entrypoint = runpy.run_path(str(SCRIPT))
    revision = entrypoint['_git_revision']
    revision.__globals__['__file__'] = str(tmp_path / 'scripts/train/predictive_world/train.py')
    with pytest.raises(subprocess.CalledProcessError):
        revision()
