import pytest
import torch

from psi.models.predictive_world.checkpoint import (
    PHASE_MODULES, save_predictive_world_bundle, complete_predictive_world_bundle,
    read_predictive_world_bundle,
)
from psi.models.predictive_world.checkpoint_diagnostic import verify_diagnostic_roundtrip, state_tree_sha256
from psi.models.predictive_world.engine import save_engine_state
from test_checkpoint import _joint_metadata


def test_real_diagnostic_save_clear_reload_restores_weights_optimizer_scheduler_and_rng(tmp_path):
    modules = {name: torch.nn.Linear(2, 2) for name in sorted(PHASE_MODULES['joint'])}
    parameters = [p for module in modules.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    for _ in range(3):
        optimizer.zero_grad()
        sum(p.square().sum() for p in parameters).backward()
        optimizer.step()
        scheduler.step()
    metadata = {**_joint_metadata(), 'artifact_use': 'diagnostic_only', 'production_accepted': False}
    path = tmp_path/'checkpoint'
    save_predictive_world_bundle(path, phase='joint', modules=modules,
                                 completed_updates=3, metadata=metadata, publish_complete=False)
    save_engine_state(path, optimizer=optimizer, scheduler=scheduler, completed_updates=3,
                      next_occurrence=6, config_identity='test-config')
    complete_predictive_world_bundle(path)
    with pytest.raises(ValueError, match='diagnostic-only'):
        read_predictive_world_bundle(path)
    result = verify_diagnostic_roundtrip(
        path, modules=modules, optimizer=optimizer, scheduler=scheduler,
        config_identity='test-config', completed_updates=3, next_occurrence=6, device=torch.device('cpu'))
    assert result['status'] == 'passed'
    assert result['optimizer_state_equal'] and result['model_tensors_equal'] and result['rng_state_equal']
    assert result['extra_optimizer_updates'] == 0


def test_state_tree_digest_covers_values_shapes_types_and_order():
    value = {'state': torch.arange(4).float(), 'lr': .001}
    baseline = state_tree_sha256(value)
    assert baseline == state_tree_sha256(dict(reversed(list(value.items()))))
    assert baseline != state_tree_sha256({**value, 'lr': .002})
    assert baseline != state_tree_sha256({**value, 'state': value['state'].reshape(2, 2)})
    assert baseline != state_tree_sha256({**value, 'state': value['state'].double()})
