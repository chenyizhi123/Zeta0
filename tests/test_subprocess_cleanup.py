"""Use real child processes to verify test-runner process-group cleanup."""

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def test_output_and_failure_status_are_preserved(run_isolated_process):
    result = run_isolated_process(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"],
        timeout=5,
    )
    assert result.returncode == 7
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


@pytest.mark.parametrize("timeout_parent", [False, True])
def test_descendant_is_stopped_even_if_leader_exits(
    tmp_path, run_isolated_process, timeout_parent
):
    import fcntl

    lock_path = tmp_path / "child.lock"
    ready_path = tmp_path / "ready"
    child = """
import fcntl, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1], 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    pathlib.Path(sys.argv[2]).touch()
    time.sleep(30)
"""
    parent = """
import pathlib, subprocess, sys, time
subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2], sys.argv[3]],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while not pathlib.Path(sys.argv[3]).exists():
    time.sleep(0.01)
print('child ready', flush=True)
if sys.argv[4] == 'True':
    time.sleep(30)
"""
    args = [sys.executable, "-c", parent, child, str(lock_path), str(ready_path), str(timeout_parent)]
    if timeout_parent:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            run_isolated_process(args, timeout=3)
        assert b"child ready" in caught.value.output
    else:
        result = run_isolated_process(args, timeout=3)
        assert result.returncode == 0
        assert result.stdout == "child ready\n"
    assert ready_path.exists(), "child must have held the lock before cleanup"
    # A killed child may briefly be a zombie; lock release proves it cannot run.
    with lock_path.open("a") as lock:
        deadline = time.monotonic() + 2
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    pytest.fail("descendant survived process-group cleanup")
                time.sleep(0.01)
