import torch
import pytest
from torch import nn

from psi.models.predictive_world.engine import (
    EngineConfig,
    PhaseLoss,
    _clip_gradients_,
    _read_only_fp64_grad_norm,
    run_phase_training,
)


def test_engine_uses_one_update_denominator_across_microbatches():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    batches = iter([
        (torch.tensor([[1.0], [100.0]]), torch.tensor([[2.0], [0.0]]), torch.tensor([True, False])),
        (torch.tensor([[1.0]]), torch.tensor([[4.0]]), torch.tensor([True])),
    ])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def validity(batch):
        return batch[2], None

    def forward(module, batch):
        x, target, valid = batch
        prediction = module(x).squeeze(-1)
        per_sample = (prediction - target.squeeze(-1)).square()
        return PhaseLoss(per_sample, valid, {"mse": per_sample[valid].mean()})

    run_phase_training(
        model=model,
        batches=batches,
        optimizer=optimizer,
        scheduler=None,
        config=EngineConfig(
            phase="world_model", max_updates=1, accumulation_steps=2,
            max_grad_norm=100.0, log_every=1, checkpoint_every=1,
        ),
        validity_fn=validity,
        forward_fn=forward,
        device=torch.device("cpu"),
    )
    # Mean of (w-2)^2 and (w-4)^2 has gradient -6 at w=0.
    torch.testing.assert_close(model.weight, torch.tensor([[0.6]]))


def test_engine_aggregates_metrics_by_global_numerator_and_denominator():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    batches = iter([
        (torch.ones(1, 1), torch.tensor([2.0]), torch.tensor([True])),
        (torch.ones(2, 1), torch.tensor([4.0, 6.0]), torch.tensor([True, True])),
    ])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    observed = []

    def forward(module, batch):
        x, target, valid = batch
        per_sample = (module(x).squeeze(-1) - target).square()
        return PhaseLoss(
            per_sample,
            valid,
            {"mse": per_sample[valid].mean()},
            metric_weights={"mse": valid.sum()},
        )

    run_phase_training(
        model=model,
        batches=batches,
        optimizer=optimizer,
        scheduler=None,
        config=EngineConfig(
            phase="world_model", max_updates=1, accumulation_steps=2,
            max_grad_norm=100.0, log_every=1, checkpoint_every=1,
        ),
        validity_fn=lambda batch: (batch[2], None),
        forward_fn=forward,
        device=torch.device("cpu"),
        on_update=lambda _update, metrics: observed.append(metrics),
    )
    assert observed[0]["mse"] == pytest.approx((4.0 + 16.0 + 36.0) / 3.0)
    assert observed[0]["global_valid_weight"] == 3.0
    assert 0 <= observed[0]['perf/input_wait_seconds'] <= observed[0]['perf/step_seconds']
    assert 0 <= observed[0]['perf/input_wait_fraction'] <= 1
    assert observed[0]['perf/rank_samples_per_second'] * observed[0]['perf/step_seconds'] == pytest.approx(3)


def test_fp64_norm_does_not_overflow_and_clipping_happens_after_measurement():
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.full_like(parameter, 1e30)
    before = parameter.grad.clone()
    norm = _read_only_fp64_grad_norm([parameter], device=torch.device("cpu"))
    assert torch.isfinite(norm)
    assert torch.equal(parameter.grad, before)
    _clip_gradients_([parameter], norm=norm, maximum=1.0)
    clipped = _read_only_fp64_grad_norm([parameter], device=torch.device("cpu"))
    assert float(clipped) == pytest.approx(1.0, rel=1e-6)


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value * 0.0

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf"))


def test_engine_rejects_nonfinite_gradient_before_clip_or_step():
    model = nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def forward(module, _batch):
        value = _FiniteForwardInfiniteBackward.apply(module.weight).reshape(1)
        return PhaseLoss(value, torch.tensor([True]), {"finite_forward": value.sum()})

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        run_phase_training(
            model=model,
            batches=iter([torch.zeros(1)]),
            optimizer=optimizer,
            scheduler=None,
            config=EngineConfig(
                phase="policy", max_updates=1, accumulation_steps=1,
                max_grad_norm=1.0, log_every=1, checkpoint_every=1,
            ),
            validity_fn=lambda _batch: (torch.tensor([True]), None),
            forward_fn=forward,
            device=torch.device("cpu"),
        )
    torch.testing.assert_close(model.weight, before)


@pytest.mark.parametrize('start,stop_at,expected', [
    (0, None, [20, 30, 40]), (20, None, [30, 40]),
    (0, 20, [20]), (0, 40, [20, 30, 40]),
])
def test_early_checkpoint_preserves_cadence_and_resume(tmp_path, monkeypatch, start, stop_at, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from psi.models.predictive_world.engine import save_engine_state, load_engine_state
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    observed = []
    current = [start]
    valid = torch.tensor([True])

    def forward(module, batch):
        loss = module(batch).reshape(1).square()
        return PhaseLoss(loss, valid, {})

    def checkpoint(update):
        directory = tmp_path / str(update)
        directory.mkdir()  # Reject duplicate callbacks for immutable checkpoints.
        save_engine_state(directory, optimizer=optimizer, scheduler=None,
                          completed_updates=update, next_occurrence=update,
                          config_identity='early-checkpoint-test')
        restored = torch.optim.AdamW(model.parameters(), lr=0.01)
        assert load_engine_state(directory, optimizer=restored, scheduler=None,
                                 config_identity='early-checkpoint-test',
                                 device=torch.device('cpu')) == (update, update)
        observed.append(update)

    run_phase_training(
        model=model, batches=iter([torch.ones(1, 1)] * 40),
        optimizer=optimizer, scheduler=None,
        config=EngineConfig(phase='world_model', max_updates=40, accumulation_steps=1,
                            max_grad_norm=1.0, log_every=1, checkpoint_every=30,
                            first_checkpoint_update=20),
        validity_fn=lambda _: (valid, None), forward_fn=forward,
        device=torch.device('cpu'), start_update=start,
        on_update=lambda update, _: current.__setitem__(0, update),
        on_checkpoint=checkpoint,
        should_stop=lambda: current[0] == stop_at,
    )
    assert observed == expected


def test_step_end_hook_runs_between_updates_and_checkpoints_with_custom_logging():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    events = []
    def forward(module, batch):
        loss = module(batch).square().flatten()
        return PhaseLoss(loss, torch.ones_like(loss, dtype=torch.bool), {'mse': loss.mean()})
    run_phase_training(
        model=model, batches=iter([torch.ones(1, 1)] * 3), optimizer=optimizer,
        scheduler=None, config=EngineConfig(phase='joint', max_updates=5,
        accumulation_steps=1, max_grad_norm=1, log_every=10, checkpoint_every=5),
        validity_fn=lambda batch: (torch.ones(1, dtype=torch.bool), None),
        forward_fn=forward, device=torch.device('cpu'), start_update=2,
        log_due=lambda update: update in {3, 4},
        on_update=lambda update, metrics: events.append(('log', update)),
        on_step_end=lambda update: events.append(('validation', update)),
        on_checkpoint=lambda update: events.append(('checkpoint', update)),
    )
    assert events == [('log', 3), ('validation', 3), ('log', 4), ('validation', 4),
                      ('validation', 5), ('checkpoint', 5)]
