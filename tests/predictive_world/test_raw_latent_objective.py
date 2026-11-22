"""Raw prediction coordinates across forecast, readers and ordinary action learning."""
from dataclasses import replace

import pytest
import torch

from test_phases import _batch, _components
from psi.data.predictive_world.objective import masked_per_sample_mse
from psi.models.predictive_world.action_expert import clone_independent_expert
from psi.models.predictive_world.phases import (
    ForecastObjective,
    JointObjective,
    PolicyObjective,
    RepresentationObjective,
    WorldModelObjective,
)
from psi.models.predictive_world.representation import EMAFutureTrajectoryEncoder


def _unused_statistics(normalizer, monkeypatch):
    """An unfitted legacy buffer cannot participate in a raw-space computation."""
    with torch.no_grad():
        normalizer.mean.fill_(17)
        normalizer.std.fill_(3)
        normalizer.fitted.fill_(False)

    def forbidden(*args, **kwargs):
        pytest.fail("raw prediction used dataset latent statistics")

    for name in ("fit", "update_ema", "whiten", "unwhiten"):
        monkeypatch.setattr(normalizer, name, forbidden)
    return {name: value.clone() for name, value in normalizer.state_dict().items()}


@pytest.mark.parametrize("mode", ["prior", "world"])
def test_raw_forecast_is_masked_raw_mse_without_fitted_statistics(monkeypatch, mode):
    encoder, _, _, adapters, history, transition, normalizer = _components()
    _unused_statistics(normalizer, monkeypatch)
    objective = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, latent_prediction_space="raw",
    )
    batch = _batch()
    batch.update(forecast_mode=mode, _action_prefix_lengths=torch.zeros(2, dtype=torch.long))
    target = torch.arange(2 * 3 * 8 * 4).reshape(2, 3, 8, 4).float() / 10
    batch["_z_target_raw"] = target.requires_grad_()
    batch["group_valid"][0, 2] = False
    batch["world_action_group_valid"] = torch.tensor([[True, False, True], [True, True, True]])
    eligible = batch["group_valid"].clone()
    if mode == "world":
        eligible &= batch["world_action_group_valid"].long().cumprod(dim=1).bool()
    # The real F head starts at zero, so the raw target sets the exact expected MSE.
    expected, valid = masked_per_sample_mse(
        torch.zeros_like(target), target, eligible[:, :, None, None].expand_as(target),
    )
    result = objective(batch)
    torch.testing.assert_close(result.per_sample, expected)
    torch.testing.assert_close(result.metrics[f"{mode}_prediction_mse"], expected[valid].mean())
    result.per_sample.mean().backward()
    assert target.grad is None
    assert transition.output_projection[-1].weight.grad.abs().sum() > 0


def test_readout_both_responses_and_action_policy_receive_same_raw_prediction(monkeypatch):
    encoder, readout, teacher, adapters, history, transition, normalizer = _components()
    _unused_statistics(normalizer, monkeypatch)
    student = clone_independent_expert(teacher)
    with torch.no_grad():
        transition.output_projection[-1].bias.fill_(0.75)
    forecast = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, latent_prediction_space="raw",
        readout=readout, readout_weight=0.25,
        control_teacher=teacher, control_reader="both", control_weight=1,
        freeze_reference=False,
    )
    policy = PolicyObjective(
        student=student, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, latent_prediction_space="raw", freeze_world=False,
    )
    seen = {name: [] for name in ("readout", "teacher", "student")}
    handles = []
    for name, module in (("readout", readout), ("teacher", teacher), ("student", student)):
        def capture(module, args, kwargs, name=name):
            seen[name].append(kwargs["z_raw"].detach().clone())

        handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
    original_velocity = student.velocity

    def capture_policy(*args, **kwargs):
        seen.setdefault("policy", []).append(kwargs["z_raw"].detach().clone())
        return original_velocity(*args, **kwargs)

    monkeypatch.setattr(student, "velocity", capture_policy)
    batch = _batch()
    batch["_action_prefix_lengths"] = torch.zeros(2, dtype=torch.long)
    try:
        result = forecast(batch, control_student=student)
        result.per_sample.mean().backward()
        assert all(p.grad is None for p in encoder.parameters())
        assert all(p.grad is None for p in readout.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for p in student.parameters())
        assert transition.output_projection[-1].weight.grad.abs().sum() > 0
        policy(batch)
    finally:
        for handle in handles:
            handle.remove()
    # T's first response sees the observed EMA future; every prediction reader sees raw F.
    for name, values in seen.items():
        selected = values[1:] if name == "teacher" else values
        assert selected, name
        for value in selected:
            torch.testing.assert_close(value, torch.full_like(value, 0.75))
    assert result.metrics["prior_control_teacher_response_weight"].item() == 0.5
    assert result.metrics["prior_control_student_response_weight"].item() == 0.5


@pytest.mark.parametrize("warmup,ramp,expected", [(0, 0, [1., 1., 1.]), (1, 2, [0., 0.5, 1.])])
def test_raw_joint_training_updates_ema_without_calibration_and_retains_optional_forecast_schedule(
    monkeypatch, warmup, ramp, expected,
):
    encoder, readout, teacher, adapters, history, transition, normalizer = _components()
    original_statistics = _unused_statistics(normalizer, monkeypatch)
    encoder = EMAFutureTrajectoryEncoder(encoder, decay=0.5)
    student = clone_independent_expert(teacher)
    representation = RepresentationObjective(encoder, readout, teacher)
    forecast = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, latent_prediction_space="raw", freeze_reference=False,
        control_teacher=teacher, control_reader="both", control_weight=1,
    )
    world = WorldModelObjective(
        representation, forecast, normalizer,
        forecast_warmup_updates=warmup, forecast_ramp_updates=ramp,
        require_complete_initial_calibration=True,
    )
    policy = PolicyObjective(
        student=student, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, latent_prediction_space="raw", freeze_world=False,
    )
    model = JointObjective(world, policy).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _batch()
    batch["_action_prefix_lengths"] = torch.zeros(2, dtype=torch.long)
    # Raw mode needs no complete first batch for fitting per-coordinate statistics.
    batch["group_valid"][:, -1] = False
    batch["world_action_group_valid"] = torch.ones(2, 3, dtype=torch.bool)
    initial_target = next(encoder.target_encoder.parameters()).detach().clone()
    for update, weight in enumerate(expected):
        batch.update(_optimizer_update=update, forecast_mode="prior" if update % 2 == 0 else "world")
        optimizer.zero_grad(set_to_none=True)
        result = model(batch)
        assert result.metrics["forecast_objective_weight"].item() == weight
        assert result.metrics["normalizer_calibration_active"].item() == 0
        assert torch.isfinite(result.metrics["target_latent_rms"])
        result.per_sample.mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert next(encoder.online_encoder.parameters()).grad is not None
        assert readout.body_output[-1].weight.grad is not None
        optimizer.step()
    assert int(encoder.last_target_update) == 2
    assert int(encoder.last_normalizer_update) == -1
    assert not encoder.normalizer_frozen
    assert not torch.equal(initial_target, next(encoder.target_encoder.parameters()))
    for name, value in original_statistics.items():
        torch.testing.assert_close(normalizer.state_dict()[name], value, rtol=0, atol=0)


def test_factory_threads_raw_space_to_forecast_and_policy(tmp_path, monkeypatch):
    from test_scratch_joint import build_small_scratch, scratch_config

    config = scratch_config(tmp_path)
    config = replace(config, training=replace(
        config.training, latent_prediction_space="raw", normalizer_calibration_updates=0,
        forecast_warmup_updates=0, forecast_ramp_updates=0,
    ))
    config.validate()
    built, batch = build_small_scratch(monkeypatch, config)
    normalizer = built.reference_modules["normalizer"]
    _unused_statistics(normalizer, monkeypatch)
    result = built.model(batch)
    assert result.metrics["forecast_objective_weight"].item() == 1
    result.per_sample.mean().backward()
    assert not normalizer.fitted

