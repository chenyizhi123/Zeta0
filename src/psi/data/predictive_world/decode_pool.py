"""Node-local, demand-driven video decoding shared by independent GPU jobs.

Only CPU workers live here. Clients retain sample ordering and cache commits;
the service round-robins their pending requests into one bounded process pool.
The private Unix socket and authentication key are scoped to the current user.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
from itertools import count
import multiprocessing
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from psi.data.predictive_world.video import decode_frame_job, initialize_decode_worker


def default_decode_pool_path() -> str:
    # A separate allocation can explicitly supply its own pool path. Single- and
    # multi-view launchers on the same node share this default CPU budget.
    return str(Path('/tmp') / f'zeta0-decode-{os.getuid()}' / 'pool.sock')


def _decoder_identity() -> str:
    """A shared service must run the client's exact decoder code/environment."""
    digest = hashlib.sha256(os.path.realpath(sys.executable).encode())
    digest.update(str(Path(sys.prefix).resolve()).encode())
    for path in (Path(__file__), Path(__file__).with_name('video.py')):
        digest.update(path.read_bytes())
    for package in ('torch', 'torchcodec', 'numpy', 'opencv-python', 'opencv-python-headless'):
        try:
            value = version(package)
        except PackageNotFoundError:
            value = 'absent'
        digest.update(f'{package}:{value}'.encode())
    return digest.hexdigest()


def _private_paths(address: str):
    path = Path(address).expanduser().absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_uid != os.getuid() or path.parent.stat().st_mode & 0o077:
        raise ValueError('decode pool socket needs a private directory owned by the current user')
    key_path = path.with_suffix('.key')
    try:
        fd = os.open(key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(os.urandom(32))
    return str(path), key_path.read_bytes()


class SharedDecodeClient:
    """Future-based client; at most ``max_pending`` decoded batches can be in flight."""

    def __init__(self, address: str, workers: int, max_pending: int, *, idle_seconds: float = 60):
        if workers <= 0 or max_pending <= 0:
            raise ValueError('shared decoder workers and pending capacity must be positive')
        path = Path(address).expanduser().absolute()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Serialize auto-start and key creation across independently started jobs.
        with path.with_suffix('.start.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            address, key = _private_paths(str(path))
            try:
                connection = Client(address, family='AF_UNIX', authkey=key)
            except (FileNotFoundError, ConnectionRefusedError):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                process = subprocess.Popen([
                    sys.executable, '-m', 'psi.data.predictive_world.decode_pool',
                    '--socket', address, '--workers', str(workers), '--idle-seconds', str(idle_seconds),
                ], env=env, stdin=subprocess.DEVNULL, start_new_session=True)
                deadline = time.monotonic() + 120
                while True:
                    try:
                        connection = Client(address, family='AF_UNIX', authkey=key)
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('shared CPU decode service failed to start')
                        time.sleep(0.05)
            connection.send(('hello', workers, max_pending, _decoder_identity()))
            if not connection.poll(120):
                connection.close()
                raise TimeoutError('shared CPU decode service did not acknowledge the client')
            reply = connection.recv()
            if reply[0] != 'ready':
                connection.close()
                raise ValueError(reply[1])
        self.connection = connection
        self.server_pid = reply[1]
        self.max_pending = max_pending
        self.pending: dict[int, Future] = {}
        self.lock = threading.Lock()
        self.next_id = 0
        self.closed = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def submit(self, function, *args) -> Future:
        if function is not decode_frame_job:
            raise ValueError('shared decode service accepts only frame decoding requests')
        with self.lock:
            if self.closed:
                raise RuntimeError('shared decoder connection is closed')
            if len(self.pending) >= self.max_pending:
                raise RuntimeError('shared decoder prefetch bound exceeded')
            request_id = self.next_id
            self.next_id += 1
            future = Future()
            self.pending[request_id] = future
            try:
                self.connection.send(('decode', request_id, args))
            except Exception:
                del self.pending[request_id]
                raise
            return future

    def _read(self):
        failure = RuntimeError('shared CPU decode service disconnected')
        try:
            while True:
                request_id, success, payload = self.connection.recv()
                with self.lock:
                    future = self.pending.pop(request_id, None)
                if future is not None and not future.cancelled():
                    if success:
                        future.set_result(payload)
                    else:
                        future.set_exception(RuntimeError(payload))
        except (EOFError, OSError, TypeError) as error:
            failure = RuntimeError(f'shared CPU decode service disconnected: {error}')
        finally:
            with self.lock:
                self.closed = True
                pending, self.pending = self.pending, {}
            for future in pending.values():
                if not future.done():
                    future.set_exception(failure)

    def shutdown(self, wait=True, cancel_futures=False):
        with self.lock:
            pending = list(self.pending.values())
        if wait and not cancel_futures:
            for future in pending:
                future.result(timeout=600)
        with self.lock:
            if not self.closed:
                try:
                    self.connection.send(('close',))
                except (OSError, EOFError):
                    pass
                self.closed = True
            if cancel_futures:
                for future in self.pending.values():
                    future.cancel()
        self.reader.join(timeout=5)
        self.connection.close()


def serve(address: str, workers: int, idle_seconds: float):
    if workers <= 0 or idle_seconds <= 0:
        raise ValueError('workers and idle interval must be positive')
    address, key = _private_paths(address)
    # Server lifetime lock also covers idle shutdown versus a new auto-start.
    with Path(address).with_suffix('.server.lock').open('a') as server_lock:
        fcntl.flock(server_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        Path(address).unlink(missing_ok=True)
        listener = Listener(address, family='AF_UNIX', authkey=key)
        condition = threading.Condition()
        clients = {}
        client_ids = count()
        rotation = deque()
        active = 0
        stopping = False
        last_activity = time.monotonic()
        decoder_identity = _decoder_identity()
        pool = multiprocessing.get_context('spawn').Pool(workers, initializer=initialize_decode_worker)
        last_activity = time.monotonic()

        def complete(client_id, request_id, success, result):
            nonlocal active, last_activity
            with condition:
                active -= 1
                last_activity = time.monotonic()
                client = clients.get(client_id)
                if client is not None:
                    client['responses'].put((request_id, success, result))
                condition.notify_all()

        def send_results(connection, state):
            try:
                while True:
                    result = state['responses'].get()
                    if result is None:
                        return
                    with condition:
                        state['outstanding'] -= 1
                    connection.send(result)
            except (OSError, EOFError):
                pass

        def handle(connection):
            nonlocal last_activity
            # IDs must not be reused while jobs from a disconnected client are
            # still running, or their results could be routed to a new client.
            client_id = next(client_ids)
            state = None
            try:
                hello = connection.recv()
                if len(hello) != 4 or hello[0] != 'hello' or hello[1] != workers or hello[2] <= 0:
                    connection.send(('error', f'shared decode pool is configured for {workers} total workers'))
                    return
                if hello[3] != decoder_identity:
                    connection.send(('error', 'shared decoder source/environment differs; use a separate --decode-pool'))
                    return
                state = dict(pending=deque(), responses=queue.Queue(), outstanding=0, limit=hello[2])
                with condition:
                    clients[client_id] = state
                    last_activity = time.monotonic()
                    condition.notify_all()
                connection.send(('ready', os.getpid()))
                threading.Thread(target=send_results, args=(connection, state), daemon=True).start()
                while True:
                    message = connection.recv()
                    if message[0] == 'close':
                        break
                    if message[0] != 'decode':
                        raise ValueError('unknown decode request')
                    _, request_id, args = message
                    with condition:
                        if state['outstanding'] >= state['limit']:
                            raise ValueError('client exceeded decode prefetch capacity')
                        state['outstanding'] += 1
                        if not state['pending']:
                            rotation.append(client_id)
                        state['pending'].append((request_id, args))
                        last_activity = time.monotonic()
                        condition.notify_all()
            except (OSError, EOFError, ValueError):
                pass
            finally:
                with condition:
                    clients.pop(client_id, None)
                    last_activity = time.monotonic()
                    if state is not None:
                        state['responses'].put(None)
                    condition.notify_all()
                connection.close()

        def accept():
            while not stopping:
                try:
                    connection = listener.accept()
                except (OSError, EOFError):
                    return
                threading.Thread(target=handle, args=(connection,), daemon=True).start()

        threading.Thread(target=accept, daemon=True).start()
        print(f'shared CPU decoder ready: pid={os.getpid()} workers={workers} socket={address}', flush=True)
        try:
            with condition:
                while True:
                    if not clients and not active and time.monotonic() - last_activity >= idle_seconds:
                        break
                    if active >= workers or not rotation:
                        condition.wait(timeout=0.5)
                        continue
                    client_id = rotation.popleft()
                    state = clients.get(client_id)
                    if state is None or not state['pending']:
                        continue
                    request_id, args = state['pending'].popleft()
                    if state['pending']:
                        rotation.append(client_id)
                    active += 1
                    pool.apply_async(decode_frame_job, args,
                        callback=lambda result, c=client_id, r=request_id: complete(c, r, True, result),
                        error_callback=lambda error, c=client_id, r=request_id: complete(c, r, False, repr(error)))
        finally:
            stopping = True
            try:
                listener.close()
            except FileNotFoundError:
                pass
            pool.terminate()
            pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--idle-seconds', type=float, default=60)
    args = parser.parse_args()
    serve(args.socket, args.workers, args.idle_seconds)
