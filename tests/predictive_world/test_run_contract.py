import faulthandler
import json
from datetime import timedelta
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from psi.models.predictive_world.engine import publish_run_contract


def _publish_worker(rank, root, invalid):
    root = Path(root)
    with (root / f'worker-{rank}.trace').open('w') as trace:
        faulthandler.dump_traceback_later(30, repeat=True, file=trace)
        try:
            _publish_on_rank(rank, root, invalid)
        finally:
            faulthandler.cancel_dump_traceback_later()


def _publish_on_rank(rank, root, invalid):
    dist.init_process_group(
        'gloo', init_method=f'file://{root / "rendezvous"}',
        rank=rank, world_size=2, timeout=timedelta(seconds=20),
    )
    try:
        try:
            publish_run_contract(root / 'run_contract.json', {'seed': 101}, device=torch.device('cpu'))
        except RuntimeError as error:
            assert invalid
            assert 'publication failed' in str(error)
            (root / f'failed-{rank}').write_text(str(error))
        else:
            assert not invalid
            assert json.loads((root / 'run_contract.json').read_text()) == {'seed': 101}
            (root / f'passed-{rank}').touch()
    finally:
        dist.destroy_process_group()


def _join_workers(context, root, timeout=60):
    deadline = time.monotonic() + timeout
    try:
        while not context.join(timeout=min(1, max(0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                traces = '\n'.join(p.read_text() for p in sorted(root.glob('worker-*.trace')))
                raise TimeoutError(f'Publication workers did not exit within {timeout}s\n{traces}')
    finally:
        # Process-group timeouts do not bound spawn's join or worker teardown.
        # Reap even a stuck child so this test cannot strand the entire suite.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def _stalled_worker(rank):
    time.sleep(60)


def test_worker_deadline_reaps_a_real_stalled_child(tmp_path):
    context = mp.spawn(_stalled_worker, nprocs=1, join=False)
    with pytest.raises(TimeoutError, match='did not exit'):
        _join_workers(context, tmp_path, timeout=0.5)
    assert all(not process.is_alive() for process in context.processes)


@pytest.mark.parametrize('existing', [None, '{"seed": 101}', '{"seed": 102}', ''])
def test_publication_synchronizes_success_and_failure_across_ranks(tmp_path, existing):
    path = tmp_path / 'run_contract.json'
    if existing is not None:
        path.write_text(existing)
    invalid = existing in ('{"seed": 102}', '')
    context = mp.spawn(_publish_worker, args=(str(tmp_path), invalid), nprocs=2, join=False)
    _join_workers(context, tmp_path)
    assert len(list(tmp_path.glob('failed-*' if invalid else 'passed-*'))) == 2
    if existing is not None:
        assert path.read_text() == existing
    assert not list(tmp_path.glob('.run_contract.json.*'))


def test_contract_is_complete_before_it_becomes_visible(tmp_path, monkeypatch):
    import psi.models.predictive_world.engine as engine

    link = engine.os.link
    path = tmp_path / 'run_contract.json'

    def inspect_publication(source, destination):
        assert not path.exists()
        assert json.loads(Path(source).read_text()) == {'seed': 101}
        link(source, destination)

    monkeypatch.setattr(engine.os, 'link', inspect_publication)
    publish_run_contract(path, {'seed': 101}, device=torch.device('cpu'))
    assert json.loads(path.read_text()) == {'seed': 101}
