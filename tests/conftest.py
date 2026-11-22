"""Keep subprocess-based tests on the same checkout as the pytest process."""

import os
import signal
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def run_isolated_process():
    """Run a POSIX test command and clean up its process group, including on timeout.

    Descendants must not detach into another session. This is test isolation,
    not a sandbox for untrusted programs or a production job supervisor.
    """
    def run(args, *, timeout, cwd=None, env=None):
        if os.name != "posix":
            raise RuntimeError("Process-group cleanup requires POSIX")
        with subprocess.Popen(
            args, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
            finally:
                # The leader may already have exited while a compiler remains.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    return run


@pytest.fixture(scope="session", autouse=True)
def checkout_subprocess_pythonpath():
    source = str(Path(__file__).resolve().parents[1] / "src")
    original = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = source + (os.pathsep + original if original else "")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original
