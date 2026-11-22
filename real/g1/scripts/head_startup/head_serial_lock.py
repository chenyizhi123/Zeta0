"""Coordinate short monitor reads with explicit head startup/adjustment."""
from contextlib import contextmanager
import fcntl
import subprocess
import time


@contextmanager
def exclusive_head_bus(device, root, timeout=2):
    with (root/'serial.lock').open('r') as lock:
        deadline=time.monotonic()+timeout
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>=deadline:
                    raise RuntimeError('Head serial is in use by another head operation')
                time.sleep(.02)
        if subprocess.run(['fuser',str(device.resolve())],capture_output=True,timeout=1).returncode!=1:
            raise RuntimeError('Head adapter already in use or owner check failed')
        yield
