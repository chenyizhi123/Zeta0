"""Bound spawned CPU test workers and retain their stacks on a hang."""
import faulthandler
import time


def _run_worker(rank, worker, args, timeout):
    # Leave the watchdog active through interpreter shutdown as well as the test.
    faulthandler.dump_traceback_later(timeout, exit=True)
    worker(rank, *args)


def spawn_with_timeout(worker, args, *, nprocs=2, timeout=90):
    import torch.multiprocessing as mp

    context = mp.spawn(_run_worker, args=(worker, args, timeout), nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout + 15
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Spawned workers did not exit within {timeout + 15}s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
