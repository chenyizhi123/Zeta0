"""Real disk round-trip checks for explicitly saved diagnostic bundles."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from psi.data.predictive_world.cache import file_sha256
from psi.models.predictive_world.checkpoint import load_predictive_world_bundle, read_predictive_world_bundle
from psi.models.predictive_world.engine import load_engine_state, _rng_state
from psi.models.predictive_world.training import module_state_sha256


def state_tree_sha256(value):
    """Typed content digest, without buffering a second complete optimizer."""
    digest = hashlib.sha256()
    def visit(item):
        digest.update(type(item).__name__.encode() + b'\0')
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode())
        digest.update(b'\0end\0')
    visit(value)
    return digest.hexdigest()


def verify_diagnostic_roundtrip(path, *, modules, optimizer, scheduler,
                                config_identity, completed_updates, next_occurrence, device):
    """Clear in-memory state, reload real artifacts, and compare exact contents.

    No additional optimizer update is performed. This tests serialization and
    state restoration, not equivalence of an uninterrupted future trajectory.
    """
    sidecar = read_predictive_world_bundle(path, allow_diagnostic=True)
    if sidecar['metadata'].get('artifact_use') != 'diagnostic_only':
        raise ValueError('round-trip mutation is restricted to diagnostic checkpoints')
    before_modules = {name: module_state_sha256(module) for name, module in modules.items()}
    before_optimizer = state_tree_sha256(optimizer.state_dict())
    before_scheduler = state_tree_sha256(scheduler.state_dict())
    before_rng = state_tree_sha256(_rng_state())
    with torch.no_grad():
        for module in modules.values():
            for parameter in module.parameters():
                parameter.zero_()
    optimizer.state.clear()
    scheduler.last_epoch = -1
    load_predictive_world_bundle(path, phase=sidecar['phase'], modules=modules,
                                 expected_metadata=sidecar['metadata'], allow_diagnostic=True)
    restored_updates, restored_occurrence = load_engine_state(
        path, optimizer=optimizer, scheduler=scheduler, config_identity=config_identity, device=device)
    checks = {
        'model_tensors_equal': before_modules == {name: module_state_sha256(module) for name, module in modules.items()},
        'optimizer_state_equal': before_optimizer == state_tree_sha256(optimizer.state_dict()),
        'scheduler_state_equal': before_scheduler == state_tree_sha256(scheduler.state_dict()),
        'rng_state_equal': before_rng == state_tree_sha256(_rng_state()),
        'update_cursor_equal': restored_updates == completed_updates == sidecar['completed_updates'],
        'data_cursor_equal': restored_occurrence == next_occurrence,
    }
    if not all(checks.values()):
        raise ValueError(f'diagnostic checkpoint round trip failed: {checks}')
    return {'format': 'zeta0.diagnostic_checkpoint_roundtrip.v1', 'status': 'passed',
            **checks, 'completed_updates': restored_updates, 'next_occurrence': restored_occurrence,
            'module_sha256': before_modules, 'weights_sha256': sidecar['weights_sha256'],
            'optimizer_state_sha256': before_optimizer,
            'trainer_file_sha256': file_sha256(Path(path)/'trainer_state_rank0000.pt'),
            'extra_optimizer_updates': 0, 'production_accepted': False}
