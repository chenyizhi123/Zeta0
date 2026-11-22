"""Exercise default discovery and checkout imports with the real pytest runner."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_subprocess_uses_same_checkout_from_other_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import psi; print(psi.__file__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert Path(result.stdout.strip()).resolve() == ROOT / "src/psi/__init__.py"


def test_default_pytest_stays_in_tests_and_uses_local_source(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    package = tmp_path / "src/psi"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "CHECKOUT = 'local-test-checkout'\n", encoding="utf-8"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_local.py").write_text(
        "import psi\ndef test_import():\n    assert psi.CHECKOUT == 'local-test-checkout'\n",
        encoding="utf-8",
    )
    for directory in ("scripts", "baselines", "src/vendor"):
        path = tmp_path / directory
        path.mkdir(parents=True)
        (path / "test_do_not_collect.py").write_text(
            "raise RuntimeError('out-of-scope collection')\n",
            encoding="utf-8",
        )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
