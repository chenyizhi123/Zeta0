import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WHEELS = Path(os.environ.get("HIW_WHEELHOUSE", "/nonexistent/hiw-offline-wheels"))


def test_missing_pip_has_actionable_offline_error(tmp_path):
    environment = tmp_path / "without-pip"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
        timeout=30,
    )
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    result = subprocess.run(
        ["bash", "-uc", "source scripts/data/hiw_offline_env.sh"],
        cwd=ROOT,
        env=dict(
            os.environ,
            PYTHON_BIN=str(environment / "bin/python"),
            OUTPUT_ROOT=str(tmp_path / "output"),
            HIW_WHEELHOUSE=str(wheels),
            PYTHONPATH="",
            PYTHONNOUSERSITE="1",
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "data-bootstrap" in result.stderr
    assert "No network fallback" in result.stderr
    assert "Installing HIW dependencies" not in result.stdout


def test_launchers_have_no_online_install_fallback():
    for name in ("run_hiw500_3node_convert.sh", "convert_hiw500_mcap.sh"):
        script = (ROOT / "scripts/data" / name).read_text()
        assert 'source "$ROOT/scripts/data/hiw_offline_env.sh"' in script
        assert "--index-url" not in script
        assert "pypi.org" not in script
    helper = (ROOT / "scripts/data/hiw_offline_env.sh").read_text()
    assert "--no-index" in helper and "--find-links" in helper
    assert "from mcap.reader import make_reader" in helper
    assert "lz4.frame.decompress" in helper and "ZstdDecompressor" in helper


@pytest.mark.skipif(not os.environ.get("HIW_WHEELHOUSE") or not WHEELS.is_dir(),
                    reason="set HIW_WHEELHOUSE to an explicitly prepared offline wheelhouse")
def test_three_cold_bootstraps_share_offline_install_lock(tmp_path):
    env = dict(
        os.environ,
        PYTHON_BIN=sys.executable,
        OUTPUT_ROOT=str(tmp_path),
        HIW_WHEELHOUSE=str(WHEELS),
        PYTHONPATH="",
        PYTHONNOUSERSITE="1",
        HTTP_PROXY="http://127.0.0.1:9",
        HTTPS_PROXY="http://127.0.0.1:9",
        PIP_INDEX_URL="http://127.0.0.1:9",
    )
    command = ["bash", "-euc", "source scripts/data/hiw_offline_env.sh"]
    children = [
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(3)
    ]
    outputs = []
    try:
        for child in children:
            output, _ = child.communicate(timeout=60)
            assert child.returncode == 0, output
            assert "HIW offline dependencies OK" in output
            assert "Looking in indexes" not in output
            outputs.append(output)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
    # Already-correct base environments need no install at all.
    assert (
        sum(output.count("Installing HIW dependencies OFFLINE") for output in outputs)
        <= 1
    )
