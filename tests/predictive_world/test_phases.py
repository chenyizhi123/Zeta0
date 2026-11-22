import copy
import os

import pytest
import torch

from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.data.predictive_world.contracts import NativeActionBatch, ObservedHistory
from psi.data.predictive_world.objective import masked_per_sample_mse
from psi.models.predictive_world.action_expert import (
    TemporalNativeActionExpert,
    clone_independent_expert,
    committed_action_prefix_lengths,
    make_committed_action_prefix,
)
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.latent_transition import LatentNormalizer, TemporalLatentTransition
from psi.models.predictive_world.phases import (
    ForecastObjective,
    JointObjective,
    PolicyObjective,
    PhaseModelWithObservation,
    RepresentationObjective,
    WorldModelObjective,
    _readout_losses,
)
from psi.models.predictive_world.policy import ObservationFeatures
from psi.models.predictive_world.representation import (
    EMAFutureTrajectoryEncoder,
    FutureTrajectoryEncoder,
    WorldReadout,
)


def _expert(action_steps=4):
    return TemporalNativeActionExpert(
        {"toy": {"dim": 2, "mean": [1.0, -1.0], "std": [2.0, 4.0]}},
        {"state": 3}, qwen_dim=6, context_tokens=3, latent_dim=4,
        latent_groups=3, latent_slots=8, state_history_count=5, action_steps=action_steps,
        hidden_dim=24, num_blocks=1, num_heads=4, ff_dim=48,
    )


def _components(action_steps=4):
    encoder = FutureTrajectoryEncoder(
        visual_dim=6, measurement_dim=4, latent_shape=(3, 8, 4), hidden_dim=16,
        num_blocks=1, num_heads=4, ff_dim=32,
    )
    readout = WorldReadout(
        visual_dim=6, measurement_dim=4, latent_shape=(3, 8, 4), max_scene_patches=5,
        body_points_per_group=2, hidden_dim=16, num_blocks=1, num_heads=4, ff_dim=32,
    )
    teacher = _expert(action_steps=action_steps)
    adapters = StateAdapterBank({"state": 3})
    history = PredictiveHistoryEncoder(
        qwen_dim=6, visual_dim=6, state_dim=64, state_history_count=5,
        output_tokens=3, context_tokens=4, hidden_dim=16, num_blocks=1,
        num_heads=4, ff_dim=32,
    )
    flow = TemporalLatentTransition(
        {"toy": {"dim": 2, "mean": [1.0, -1.0], "std": [2.0, 4.0]}},
        context_dim=16, latent_shape=(3, 8, 4), hidden_dim=16,
        num_blocks=1, num_heads=4, ff_dim=32,
    )
    normalizer = LatentNormalizer((3, 8, 4))
    normalizer.fit(torch.randn(4, 3, 8, 4), torch.ones(4, 3, dtype=torch.bool))
    return encoder, readout, teacher, adapters, history, flow, normalizer


def _batch(action_steps=4):
    batch = 2
    observed = ObservedHistory(
        qwen_inputs={}, raw_states=torch.randn(batch, 5, 3),
        state_valid=torch.ones(batch, 5, 3, dtype=torch.bool),
        state_times=torch.linspace(-0.4, 0, 5)[None].expand(batch, -1),
        state_schema_ids=("state",) * batch,
        action_protocol_ids=("toy",) * batch,
        action_native_dims=torch.full((batch,), 2),
        action_normalization_spaces=("native",) * batch,
    )
    actions = NativeActionBatch(
        values=torch.randn(batch, action_steps, 2), valid=torch.ones(batch, action_steps, 2, dtype=torch.bool),
        times=torch.arange(action_steps).float()[None].expand(batch, -1) / 50,
        protocol_ids=("toy",) * batch, native_dims=torch.full((batch,), 2),
        normalization_spaces=("native",) * batch,
    )
    return {
        "observed": observed,
        "features": ObservationFeatures(
            qwen_context=torch.randn(batch, 5, 6),
            context_valid=torch.ones(batch, 5, dtype=torch.bool),
            current_visual=torch.randn(batch, 3, 6),
        ),
        "actions": actions,
        "anchor_patches": torch.randn(batch, 3, 6),
        "anchor_patch_valid": torch.ones(batch, 3, dtype=torch.bool),
        "current_measurements": torch.randn(batch, 4),
        "current_measurement_valid": torch.ones(batch, 4, dtype=torch.bool),
        "future_patches": torch.randn(batch, 3, 5, 6),
        "future_patch_valid": torch.ones(batch, 3, 5, dtype=torch.bool),
        "future_scene_times": torch.tensor([[0.2, 0.4, 0.6]]).expand(batch, -1),
        "future_measurements": torch.randn(batch, 3, 2, 4),
        "future_measurement_valid": torch.ones(batch, 3, 2, 4, dtype=torch.bool),
        "future_measurement_times": torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]
        ).expand(batch, -1, -1),
        "group_valid": torch.ones(batch, 3, dtype=torch.bool),
        "action_sigma": torch.full((batch,), 0.5),
        "action_noise": torch.randn(batch, action_steps, 2),
    }


def test_representation_forecast_and_policy_objectives_obey_ownership():
    encoder, readout, teacher, adapters, history, flow, normalizer = _components()
    batch = _batch()
    representation = RepresentationObjective(encoder, readout, teacher)
    result = representation(batch)
    assert "teacher_action_mse" in result.metrics
    assert "latent_token_std" in result.metrics
    assert "latent_regularizer" in result.metrics
    result.per_sample.mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    for module in (encoder, readout, teacher):
        module.zero_grad(set_to_none=True)

    forecast = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history, latent_transition=flow,
    )
    result = forecast(batch)
    result.per_sample.mean().backward()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in flow.parameters())

    student = clone_independent_expert(teacher)
    policy = PolicyObjective(
        student=student, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history, latent_transition=flow,
    )
    result = policy(batch)
    result.per_sample.mean().backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


@pytest.mark.parametrize('maximum', [0, 6, 13])
def test_phase_wrapper_caps_real_teacher_student_and_prior_prefixes(monkeypatch, maximum):
    torch.manual_seed(47)
    encoder, readout, teacher, adapters, history, transition, normalizer = _components(action_steps=30)
    student = clone_independent_expert(teacher)
    representation = RepresentationObjective(encoder, readout, teacher)
    forecast = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history, latent_transition=transition,
        control_teacher=teacher, control_weight=1.0, freeze_reference=False,
    )
    world = WorldModelObjective(representation, forecast, normalizer)
    policy = PolicyObjective(
        student=student, normalizer=normalizer, history_state_adapters=adapters,
        history_encoder=history, latent_transition=transition, freeze_world=False,
    )
    batch = _batch(action_steps=30)

    batch['world_action_group_valid'] = torch.ones(2, 3, dtype=torch.bool)

    class CachedObservation(torch.nn.Module):
        # Only replace frozen image extraction; all training/sampling paths are real.
        def forward(self, observed):
            return batch['features']

    wrapped = PhaseModelWithObservation(
        CachedObservation(), JointObjective(world, policy), rtc_training_max_delay=maximum,
    )
    prefixes = {'teacher': [], 'student': [], 'prior': []}
    for name, expert in (('teacher', teacher), ('student', student)):
        original = expert.velocity

        def record_velocity(*args, _original=original, _name=name, **kwargs):
            prefixes[_name].extend((kwargs['sigma_a'] == 0).sum(dim=-1).tolist())
            return _original(*args, **kwargs)

        monkeypatch.setattr(expert, 'velocity', record_velocity)

    def record_prior(module, args, kwargs):
        if kwargs.get('mode') == 'prior':
            prefix = kwargs.get('action_condition')
            prefixes['prior'].extend([0, 0] if prefix is None else committed_action_prefix_lengths(prefix).tolist())

    handle = transition.register_forward_pre_hook(record_prior, with_kwargs=True)
    try:
        for mode in ('prior', 'world'):
            batch['forecast_mode'] = mode
            # An input batch cannot override the config-owned cap.
            batch['_rtc_training_max_delay'] = 29
            result = wrapped(batch)
            result.validate()
            result.per_sample[result.valid].mean().backward()
            assert all(torch.isfinite(p.grad).all() for p in wrapped.parameters() if p.grad is not None)
            wrapped.zero_grad(set_to_none=True)
    finally:
        handle.remove()
    for name, values in prefixes.items():
        assert values and min(values) >= 0 and max(values) <= maximum, (name, values)
        if maximum:
            assert max(values) > 0, (name, values)
    assert batch['_rtc_training_max_delay'] == 29  # caller input remains unchanged


@pytest.mark.parametrize('provided', ['lengths', 'committed'])
def test_training_cap_rejects_explicit_long_prefixes(provided):
    from psi.models.predictive_world.phases import _committed_prefix
    from dataclasses import replace
    batch = _batch(action_steps=30)
    batch['_rtc_training_max_delay'] = 6
    lengths = torch.tensor([0, 7])
    if provided == 'lengths':
        batch['action_prefix_lengths'] = lengths
    else:
        batch['observed'] = replace(batch['observed'], committed_actions=make_committed_action_prefix(batch['actions'], lengths))
    with pytest.raises(ValueError, match='exceeds rtc_training_max_delay'):
        _committed_prefix(batch, batch['observed'], batch['actions'])


def test_representation_masks_body_targets_for_groups_without_visual_latents():
    encoder, readout, teacher, _, _, _, _ = _components()
    batch = _batch()
    batch["group_valid"][:, -1] = False
    batch["future_patch_valid"][:, -1] = False
    batch["future_measurements"][:, -1] = 1e6
    batch["_action_prefix_lengths"] = torch.zeros(2, dtype=torch.long)
    objective = RepresentationObjective(encoder, readout, teacher)
    result = objective(batch)
    changed = copy.deepcopy(batch)
    changed["future_measurements"][:, -1] = -1e6
    changed_result = objective(changed)
    torch.testing.assert_close(result.per_sample, changed_result.per_sample)


def test_world_model_uses_checkpointed_ema_target_and_frozen_normalizer_stage():
    encoder, readout, teacher, adapters, history, flow, _ = _components()
    encoder = EMAFutureTrajectoryEncoder(encoder, decay=0.5)
    with torch.no_grad():
        next(encoder.online_encoder.parameters()).add_(0.1)
    normalizer = LatentNormalizer((3, 8, 4))
    representation = RepresentationObjective(encoder, readout, teacher)
    forecast = ForecastObjective(
        reference_encoder=encoder,
        normalizer=normalizer,
        history_state_adapters=adapters,
        history_encoder=history,
        latent_transition=flow,
        freeze_reference=False,
    )
    objective = WorldModelObjective(
        representation,
        forecast,
        normalizer,
        normalizer_calibration_updates=1,
        forecast_warmup_updates=1,
        forecast_ramp_updates=2,
    )
    batch = _batch()
    batch.update({"forecast_mode": "prior", "_optimizer_update": 0})
    result = objective(batch)
    assert result.metrics["forecast_objective_weight"].item() == 0
    assert bool(encoder.normalizer_frozen)
    assert int(encoder.last_target_update) == 0
    assert int(encoder.last_normalizer_update) == 0
    assert result.metrics["online_target_latent_mse"].item() > 0
    assert torch.isfinite(result.metrics["online_target_latent_cosine"])
    with torch.no_grad():
        target = encoder(**{
            name: batch[name]
            for name in (
                "anchor_patches", "anchor_patch_valid", "current_measurements",
                "current_measurement_valid", "future_patches", "future_patch_valid",
                "future_scene_times", "future_measurements", "future_measurement_valid",
                "future_measurement_times", "group_valid",
            )
        })
    torch.testing.assert_close(normalizer.mean, target.mean(dim=0))
    frozen_mean = normalizer.mean.clone()
    batch["_optimizer_update"] = 1
    result = objective(batch)
    assert result.metrics["forecast_objective_weight"].item() == 0.5
    torch.testing.assert_close(normalizer.mean, frozen_mean)


def test_policy_uses_deployable_prediction_with_committed_prefix():
    encoder, _, teacher, adapters, history, flow, normalizer = _components()
    student = clone_independent_expert(teacher)
    policy = PolicyObjective(
        student=student,
        normalizer=normalizer,
        history_state_adapters=adapters,
        history_encoder=history,
        latent_transition=flow,
    )
    batch = _batch()
    batch["action_prefix_lengths"] = torch.tensor([1, 2])
    result = policy(batch)
    assert result.valid.all()
    assert result.metrics["policy_origin_predicted"].item() == 1


def test_predicted_action_loss_refines_world_dynamics_without_training_future_encoder():
    encoder, _, teacher, adapters, history, flow, normalizer = _components()
    student = clone_independent_expert(teacher)
    policy = PolicyObjective(
        student=student,
        normalizer=normalizer,
        history_state_adapters=adapters,
        history_encoder=history,
        latent_transition=flow,
        freeze_world=False,
    )
    batch = _batch()
    result = policy(batch)
    result.per_sample[result.valid].mean().backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert any(parameter.grad is not None for parameter in history.parameters())
    assert any(parameter.grad is not None for parameter in flow.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_joint_world_model_trains_representation_and_dynamics_together():
    encoder, readout, teacher, adapters, history, flow, normalizer = _components()
    representation = RepresentationObjective(encoder, readout, teacher)
    forecast = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history, latent_transition=flow,
        freeze_reference=False,
    )
    objective = WorldModelObjective(representation, forecast, normalizer)
    batch = _batch()
    batch["forecast_mode"] = "prior"
    result = objective(batch)
    result.per_sample.mean().backward()
    assert result.valid.all()
    assert "world_model_objective" in result.metrics
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in flow.parameters())


def test_joint_continuation_keeps_both_losses_and_advances_ema_clock():
    online, readout, teacher, adapters, history, flow, normalizer = _components()
    encoder = EMAFutureTrajectoryEncoder(online, decay=0.5)
    representation = RepresentationObjective(encoder, readout, teacher)
    forecast = ForecastObjective(
        reference_encoder=encoder,
        normalizer=normalizer,
        history_state_adapters=adapters,
        history_encoder=history,
        latent_transition=flow,
        freeze_reference=False,
    )
    world = WorldModelObjective(
        representation,
        forecast,
        normalizer,
        optimizer_update_offset=15_000,
    )
    student = clone_independent_expert(teacher)
    policy = PolicyObjective(
        student=student,
        normalizer=normalizer,
        history_state_adapters=adapters,
        history_encoder=history,
        latent_transition=flow,
        freeze_world=False,
    )
    objective = JointObjective(world, policy)
    batch = _batch()
    batch.update({
        "forecast_mode": "prior",
        "_optimizer_update": 0,
    })
    result = objective(batch)
    result.validate()
    result.per_sample[result.valid].mean().backward()
    assert int(encoder.last_target_update) == 15_000
    assert result.metrics["absolute_optimizer_update"].item() == 15_000
    assert "joint_world_component" in result.metrics
    assert "joint_action_component" in result.metrics
    assert any(parameter.grad is not None for parameter in online.parameters())
    assert any(parameter.grad is not None for parameter in flow.parameters())
    assert any(parameter.grad is not None for parameter in student.parameters())


def test_policy_prediction_never_consumes_future_labels_or_availability():
    _, _, teacher, adapters, history, transition, normalizer = _components()
    policy = PolicyObjective(
        student=clone_independent_expert(teacher), normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, freeze_world=False,
    )
    batch = _batch()
    batch["_optimizer_update"] = 99
    batch["_action_prefix_lengths"] = torch.zeros(2, dtype=torch.long)
    result = policy(batch)
    causal = {k: v for k, v in batch.items() if not k.startswith("future_")}
    causal["group_valid"] = torch.tensor([[True, False, False]]).expand(2, -1)
    changed = policy(causal)
    torch.testing.assert_close(result.per_sample, changed.per_sample)
    assert changed.valid.all()
    assert "student_action_mse" in result.metrics


def test_control_response_trains_predictor_without_altering_teacher_gradients():
    torch.manual_seed(12)
    encoder, readout, teacher, adapters, history, transition, normalizer = _components()
    torch.nn.init.normal_(transition.output_projection[-1].weight, std=0.05)
    batch = _batch()
    batch["_action_prefix_lengths"] = torch.tensor([1, 2])
    representation = RepresentationObjective(encoder, readout, teacher)
    rep = representation(batch).per_sample.mean()
    teacher_parameters = list(teacher.parameters())
    expected = torch.autograd.grad(rep, teacher_parameters, retain_graph=True, allow_unused=True)
    captured = []
    handle = teacher.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(dict(kwargs)), with_kwargs=True
    )
    objective = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, prior_weight=0, world_weight=0,
        control_teacher=teacher, control_weight=1, freeze_reference=False,
    )
    result = objective(batch)
    handle.remove()
    assert len(captured) == 2
    for key in ("normalized_action_samples", "sigma_a", "action_valid", "latent_group_valid"):
        torch.testing.assert_close(captured[0][key], captured[1][key])
    assert captured[0]["sigma_a"][0, 0] == 0
    assert captured[0]["sigma_a"][1, :2].eq(0).all()
    assert not captured[0]["z_raw"].requires_grad
    assert captured[1]["z_raw"].requires_grad
    assert result.metrics["prior_control_response_mse"] > 0
    result.per_sample.mean().backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    for module in (history, transition):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    rep.backward()
    for parameter, gradient in zip(teacher_parameters, expected):
        assert parameter.requires_grad
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient)


def test_executor_reader_uses_shared_flow_and_updates_predictor_only():
    torch.manual_seed(620)
    encoder, _, teacher, adapters, history, transition, normalizer = _components()
    student = clone_independent_expert(teacher)
    with torch.no_grad():
        student.adapters["toy"].output[-1].bias.add_(0.4)
        torch.nn.init.normal_(transition.output_projection[-1].weight, std=0.05)
    batch = _batch()
    batch["_action_prefix_lengths"] = torch.tensor([1, 2])
    batch["group_valid"][0, 1] = False
    batch["group_valid"][1] = False
    calls = {"teacher": [], "student": []}
    teacher_hook = teacher.register_forward_hook(
        lambda module, args, kwargs, output: calls["teacher"].append((kwargs, output)),
        with_kwargs=True,
    )
    student_hook = student.register_forward_hook(
        lambda module, args, kwargs, output: calls["student"].append((kwargs, output)),
        with_kwargs=True,
    )
    objective = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, prior_weight=0, world_weight=0,
        control_teacher=teacher,
        control_reader="student",
        control_weight=1, freeze_reference=False,
    )
    assert not {id(parameter) for parameter in objective.parameters()} & {
        id(parameter) for parameter in student.parameters()
    }
    result = objective(batch, control_student=student)
    teacher_hook.remove()
    student_hook.remove()
    assert len(calls["teacher"]) == len(calls["student"]) == 1
    oracle_args, oracle_velocity = calls["teacher"][0]
    reader_args, reader_velocity = calls["student"][0]
    for key in ("normalized_action_samples", "sigma_a", "action_valid", "latent_group_valid"):
        torch.testing.assert_close(oracle_args[key], reader_args[key])
    assert not oracle_velocity.requires_grad
    assert reader_velocity.requires_grad
    assert not oracle_args["z_raw"].requires_grad
    assert reader_args["z_raw"].requires_grad
    _, action_valid = teacher.normalize_actions(batch["actions"])
    response_mask = action_valid.clone()
    response_mask[0, :1] = False
    response_mask[1, :2] = False
    response_mask &= batch["group_valid"].any(dim=1)[:, None, None]
    expected, valid = masked_per_sample_mse(reader_velocity, oracle_velocity, response_mask)
    torch.testing.assert_close(result.per_sample, expected)
    assert torch.equal(result.valid, valid)
    assert result.valid.tolist() == [True, False]
    result.per_sample[result.valid].mean().backward()
    assert all(parameter.grad is None for parameter in (list(teacher.parameters())
                                                       + list(student.parameters())
                                                       + list(encoder.parameters())))
    for module in (history, transition):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in module.parameters())
    policy = PolicyObjective(
        student=student, normalizer=normalizer, history_state_adapters=adapters,
        history_encoder=history, latent_transition=transition,
    )
    policy(batch).per_sample.mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in student.parameters())


def test_joint_executor_reader_uses_its_own_student_after_model_copy():
    torch.manual_seed(621)
    encoder, readout, teacher, adapters, history, transition, normalizer = _components()
    student = clone_independent_expert(teacher)
    world = WorldModelObjective(
        RepresentationObjective(encoder, readout, teacher),
        ForecastObjective(
            reference_encoder=encoder, normalizer=normalizer,
            history_state_adapters=adapters, history_encoder=history,
            latent_transition=transition, control_teacher=teacher,
            control_reader="student", control_weight=1, freeze_reference=False,
        ), normalizer,
    )
    policy = PolicyObjective(
        student=student, normalizer=normalizer, history_state_adapters=adapters,
        history_encoder=history, latent_transition=transition, freeze_world=False,
    )
    original = JointObjective(world, policy)
    batch = _batch()
    batch["_action_prefix_lengths"] = torch.zeros(2, dtype=torch.long)
    original(batch)  # Populate the one-time expert contract check before copying.
    copied = copy.deepcopy(original)
    assert copied.policy.student is not original.policy.student
    original_calls, copied_calls = [], []
    first = original.policy.student.register_forward_hook(lambda *args: original_calls.append(True))
    second = copied.policy.student.register_forward_hook(lambda *args: copied_calls.append(True))
    copied(batch)
    first.remove()
    second.remove()
    assert original_calls == []
    assert len(copied_calls) == 1  # Auxiliary calls the module; policy calls velocity directly.


def test_control_response_is_zero_for_matching_latents_and_masks_missing_rows():
    encoder, _, teacher, adapters, history, transition, normalizer = _components()
    batch = _batch()
    batch["_action_prefix_lengths"] = torch.zeros(2, dtype=torch.long)
    # Zero-initialized head predicts exactly the normalizer mean.
    batch["_z_target_raw"] = normalizer.mean[None].expand(2, -1, -1, -1).clone()
    batch["group_valid"][1] = False
    batch["_z_target_raw"][1] = float("nan")
    objective = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, prior_weight=0, world_weight=0,
        control_teacher=teacher, control_weight=1, freeze_reference=False,
    )
    result = objective(batch)
    assert result.valid.tolist() == [True, False]
    assert torch.isfinite(result.per_sample).all()
    torch.testing.assert_close(result.per_sample, torch.zeros_like(result.per_sample), atol=1e-8, rtol=0)
    assert result.metric_weights["prior_control_response_mse"].item() == 1


@pytest.mark.parametrize("mode,scope,weight,calls", [
    ("prior", "prior", 1.0, 2),
    ("world", "prior", 1.0, 0),
    ("world", "both", 1.0, 2),
    ("prior", "prior", 0.0, 0),
    ("world", "both", 0.0, 0),
])
def test_control_scope_skips_unneeded_teacher_forwards(mode, scope, weight, calls):
    encoder, _, teacher, adapters, history, transition, normalizer = _components()
    batch = _batch()
    batch["forecast_mode"] = mode
    batch["world_action_group_valid"] = torch.ones(2, 3, dtype=torch.bool)
    captured = []
    handle = teacher.register_forward_hook(lambda *args: captured.append(True))
    objective = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, control_teacher=teacher,
        control_weight=weight, control_scope=scope, freeze_reference=False,
    )
    result = objective(batch)
    handle.remove()
    assert len(captured) == calls
    result.validate()
    assert result.metrics[f"{mode}_control_response_active"].item() == bool(calls)
    assert (f"{mode}_control_response_mse" in result.metrics) == bool(calls)
    assert result.metrics[f"{mode}_control_response_weight"].item() == (weight if calls else 0)
    result.per_sample.mean().backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_forecast_readout_preserves_representation_gradients_but_trains_only_forecaster():
    torch.manual_seed(23)
    encoder, readout, teacher, adapters, history, transition, normalizer = _components()
    torch.nn.init.normal_(transition.output_projection[-1].weight, std=0.05)
    batch = _batch()
    representation = RepresentationObjective(encoder, readout, teacher)
    rep_loss = representation(batch).per_sample.mean()
    parameters = list(readout.parameters())
    expected = torch.autograd.grad(rep_loss, parameters, retain_graph=True, allow_unused=True)
    forecast = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, prior_weight=0, world_weight=0,
        readout=readout, readout_weight=0.25, freeze_reference=False,
    )
    result = forecast(batch)
    result.per_sample.mean().backward()
    assert all(p.requires_grad and p.grad is None for p in parameters)
    assert all(p.grad is None for p in encoder.parameters())
    for module in (history, transition):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    rep_loss.backward()
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in expected)
    for parameter, gradient in zip(parameters, expected):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient)


def test_readout_delta_masks_unmatched_and_missing_slots_without_removing_body_loss():
    class ZeroReadout(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.zeros_like(batch["future_patches"]), torch.zeros_like(batch["future_measurements"])

    batch = _batch()
    batch["future_patch_valid"][:] = False
    batch["future_measurements"] = torch.tensor([0., 100., 1., 200., 2., 300.]).view(1, 3, 2, 1).expand(2, 3, 2, 4).clone()
    batch["future_measurement_pair_valid"] = torch.tensor([[True, False], [True, False]])
    batch["future_measurement_valid"][1, 1, 0] = False
    batch["future_measurements"][1, 1, 0] = float("nan")
    decoded = _readout_losses(ZeroReadout(), batch, torch.zeros(2, 3, 8, 4), batch["group_valid"])
    assert decoded[1][1].tolist() == [True, True]
    temporal_loss, temporal_valid = decoded[2]
    assert temporal_valid.tolist() == [True, False]
    torch.testing.assert_close(temporal_loss, torch.tensor([1., 0.]))


def test_metric_denominators_follow_available_targets():
    encoder, readout, teacher, adapters, history, transition, normalizer = _components()
    batch = _batch()
    batch["_action_prefix_lengths"] = torch.zeros(2, dtype=torch.long)
    batch["group_valid"][1, 1:] = False
    batch["future_measurement_valid"][1] = False
    batch["actions"].valid[1] = False
    representation = RepresentationObjective(encoder, readout, teacher)
    result = representation(batch)
    result.validate()
    assert result.valid.tolist() == [True, True]
    for name, count in {
        "scene_mse": 2, "body_mse": 1, "teacher_action_mse": 1,
        "temporal_delta_mse": 1, "latent_token_std": 4,
    }.items():
        assert result.metric_weights[name].item() == count

    forecast = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=transition, readout=readout, readout_weight=0.25,
        control_teacher=teacher, control_weight=1.0, freeze_reference=False,
    )
    result = forecast(batch)
    result.validate()
    assert result.valid.tolist() == [True, True]
    for name, count in {
        "prediction_mse": 2, "prediction_mse_group_0": 2,
        "prediction_mse_group_1": 1, "prediction_mse_group_2": 1,
        "scene_mse": 2, "body_mse": 1, "temporal_delta_mse": 1,
        "control_response_mse": 1,
    }.items():
        assert result.metric_weights[f"prior_{name}"].item() == count


def test_cpu_world_and_joint_optimizer_steps():
    _world_and_joint_optimizer_steps("cpu")


@pytest.mark.skipif(os.getenv("PW_CUDA_SMOKE") != "1", reason="explicit local GPU smoke only")
def test_cuda_bf16_world_and_joint_optimizer_steps():
    assert torch.cuda.is_available()
    _world_and_joint_optimizer_steps("cuda")


def _world_and_joint_optimizer_steps(device):
    # Tiny synthetic dimensions: validates the complete trainable chain, not
    # production throughput or the capacity of an H100 batch of 256.
    with torch.device(device):
        encoder, readout, teacher, adapters, history, transition, normalizer = _components()
        encoder = EMAFutureTrajectoryEncoder(encoder, decay=0.99)
        representation = RepresentationObjective(encoder, readout, teacher)
        forecast = ForecastObjective(
            reference_encoder=encoder, normalizer=normalizer,
            history_state_adapters=adapters, history_encoder=history,
            latent_transition=transition, readout=readout, readout_weight=0.25,
            control_teacher=teacher, control_weight=1.0,
            freeze_reference=False,
        )
        world = WorldModelObjective(representation, forecast, normalizer)
        student = clone_independent_expert(teacher)
        policy = PolicyObjective(
            student=student, normalizer=normalizer,
            history_state_adapters=adapters, history_encoder=history,
            latent_transition=transition, freeze_world=False,
        )
        joint = JointObjective(world, policy)
        optimizer = torch.optim.AdamW(joint.parameters(), lr=1e-3)
        batch = _batch()
        batch["world_action_group_valid"] = torch.ones(2, 3, dtype=torch.bool)
        for update, mode in enumerate(("prior", "world", "prior")):
            optimizer.zero_grad(set_to_none=True)
            batch.update(forecast_mode=mode, _optimizer_update=update)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                result = (world if update < 2 else joint)(batch)
                loss = result.per_sample[result.valid].mean()
            assert torch.isfinite(loss)
            result.validate()
            assert (f"{mode}_control_response_mse" in result.metrics) == (mode == "prior")
            loss.backward()
            gradients = [p.grad for p in joint.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            if update == 2:
                for module in (history, transition, student):
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
            optimizer.step()
        assert encoder.last_target_update.item() == 2
        assert all(p.grad is None for p in encoder.target_encoder.parameters())
        if device == "cuda":
            torch.cuda.synchronize()


def test_world_evaluation_uses_online_representation_without_mutating_ema():
    from psi.models.predictive_world.monitoring import isolated_evaluation
    online, readout, teacher, adapters, history, flow, normalizer = _components()
    encoder = EMAFutureTrajectoryEncoder(online, decay=0.5)
    with torch.no_grad():
        next(encoder.online_encoder.parameters()).add_(0.2)
    representation = RepresentationObjective(encoder, readout, teacher)
    forecast = ForecastObjective(
        reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history,
        latent_transition=flow, freeze_reference=False,
    )
    objective = WorldModelObjective(representation, forecast, normalizer)
    batch = _batch()
    batch['_optimizer_update'] = 10
    batch['_action_prefix_lengths'] = torch.zeros(2, dtype=torch.long)
    before = {k: v.clone() for k, v in objective.state_dict().items()}
    with isolated_evaluation(objective, seed=9):
        result = objective(batch)
        direct = representation(batch)
        torch.testing.assert_close(result.metrics['scene_mse'], direct.metrics['scene_mse'])
        assert result.metrics['online_target_latent_mse'] > 0
        assert result.metrics['target_encoder_ema_update'] == 0
    for key, value in objective.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_new_objective_options_fail_closed():
    encoder, _, teacher, adapters, history, transition, normalizer = _components()
    arguments = dict(normalizer=normalizer, history_state_adapters=adapters,
                     history_encoder=history, latent_transition=transition)
    with pytest.raises(ValueError, match="action teacher"):
        ForecastObjective(reference_encoder=encoder, **arguments, control_weight=1.0)
    with pytest.raises(ValueError, match="control reader"):
        ForecastObjective(reference_encoder=encoder, **arguments, control_reader="other")
    reader = ForecastObjective(reference_encoder=encoder, **arguments,
                               control_teacher=teacher, control_reader="student")
    with pytest.raises(ValueError, match="independent teacher and student"):
        reader(_batch())
    student = clone_independent_expert(teacher)
    with torch.no_grad():
        student.adapters["toy"].mean.add_(1)
    with pytest.raises(ValueError, match="action normalization disagrees"):
        reader(_batch(), control_student=student)
