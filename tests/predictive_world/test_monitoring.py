import random

import numpy as np
import pytest
import torch
from torch import nn

from psi.models.predictive_world.engine import PhaseLoss
from psi.models.predictive_world.monitoring import (
    evaluate_joint, isolated_evaluation, training_log_due,
)


def test_adjacent_logs_capture_both_modes_without_changing_legacy_cadence():
    assert [i for i in range(1, 21) if training_log_due(i, 10, False)] == [1, 10, 20]
    assert [i for i in range(1, 21) if training_log_due(i, 10, True)] == [1, 2, 9, 10, 19, 20]


def test_evaluation_restores_rng_and_mixed_module_modes_on_failure():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    model[0].eval()
    torch_state = torch.get_rng_state().clone()
    py_state, np_state = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError):
        with isolated_evaluation(model, seed=123):
            assert not model.training
            random.random(), np.random.rand(), torch.rand(2)
            raise RuntimeError("failure")
    assert model.training and not model[0].training and model[1].training
    assert random.getstate() == py_state
    np.testing.assert_equal(np.random.get_state(), np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)


class MetricModel(nn.Module):
    def forward(self, batch):
        values = batch["values"]
        valid = values > 0
        return PhaseLoss(values, valid, {
            "loss": values[valid].mean(), "inactive": values.new_tensor(0),
            "random": torch.rand(()),
        }, metric_weights={
            "loss": valid.sum(), "inactive": values.new_tensor(0), "random": valid.sum(),
        })


def test_validation_is_reproducible_and_uses_metric_denominators():
    rows = [[dict(source_id=source, action_protocol_id="p", sample_id=str(i),
                  split="validation", value=value)
             for i, (source, value) in enumerate([("a", 1.), ("a", 0.), ("b", 5.)])]]
    def evaluate():
        return evaluate_joint(
            MetricModel(), rows=iter(rows), device=torch.device("cpu"), seed=8, update=0,
            collator=lambda group: {"values": torch.tensor([r["value"] for r in group])},
        )
    first, second = evaluate(), evaluate()
    assert first == second
    assert first["sample_count"] == 3
    overall = [g for g in first["groups"] if g["source_id"] == "__all__"]
    assert len(overall) == 2
    for group in overall:
        assert group["metrics"]["loss"] == 3
        assert group["metric_weights"]["loss"] == 2
        assert "inactive" not in group["metrics"]
    rows[0][0]["split"] = "train"
    with pytest.raises(ValueError, match="held-out"):
        evaluate()


def _distributed_validation_worker(rank, rendezvous, output):
    import json
    from datetime import timedelta
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        rows = [[dict(source_id=f'source{rank}', action_protocol_id='p', sample_id=f'{rank}-{i}',
                      split='validation', value=float(1 + rank * 4)) for i in range(rank + 1)]]
        report = evaluate_joint(
            MetricModel(), rows=iter(rows), device=torch.device('cpu'), seed=8, update=0,
            collator=lambda group: {'values': torch.tensor([r['value'] for r in group])},
        )
        if rank == 0:
            from pathlib import Path
            Path(output).write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


def test_distributed_validation_merges_different_sources_and_sample_counts(tmp_path):
    import json
    from ddp_test_utils import spawn_with_timeout
    output = tmp_path / 'report.json'
    spawn_with_timeout(_distributed_validation_worker,
                       args=(f'file://{tmp_path / "rendezvous"}', str(output)))
    report = json.loads(output.read_text())
    assert report['sample_count'] == 3
    assert {g['source_id'] for g in report['groups']} == {'source0', 'source1', '__all__'}
    for group in report['groups']:
        if group['source_id'] == '__all__':
            assert group['metrics']['loss'] == pytest.approx(11 / 3)
            assert group['metric_weights']['loss'] == 3
