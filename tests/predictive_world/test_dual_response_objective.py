"""One model's T/T and G/T response losses share a single causal flow sample."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from psi.data.predictive_world.objective import masked_per_sample_mse
from psi.models.predictive_world.action_expert import clone_independent_expert
from psi.models.predictive_world.phases import (
    ForecastObjective,
    JointObjective,
    PolicyObjective,
    RepresentationObjective,
    WorldModelObjective,
)


def setup(state_count=5):
    from test_phases import _batch, _components

    torch.manual_seed(2620)
    encoder, readout, teacher, adapters, history, transition, normalizer = _components(action_steps=8)
    batch = _batch(action_steps=8)
    if state_count == 1:
        # Build the already-migrated tiny architecture; transfer itself has its
        # own exact source-bound checkpoint tests.
        teacher.state_history_count = history.state_history_count = 1
        teacher.state_position = nn.Parameter(teacher.state_position[-1:].detach().clone())
        history.state_history_position = nn.Parameter(history.state_history_position[-1:].detach().clone())
        observed = batch["observed"]
        batch["observed"] = replace(observed, raw_states=observed.raw_states[:, -1:],
                                    state_valid=observed.state_valid[:, -1:], state_times=observed.state_times[:, -1:])
    student = clone_independent_expert(teacher)
    with torch.no_grad():
        student.adapters["toy"].output[-1].bias.add_(.4)
        torch.nn.init.normal_(transition.output_projection[-1].weight, std=.05)
    batch["_action_prefix_lengths"] = torch.tensor([1, 2])
    batch["_rtc_training_max_delay"] = 6
    return (encoder, readout, teacher, student, adapters, history, transition, normalizer), batch


def forecast(modules, reader, **kwargs):
    encoder, readout, teacher, _, adapters, history, transition, normalizer = modules
    return ForecastObjective(reference_encoder=encoder, normalizer=normalizer,
        history_state_adapters=adapters, history_encoder=history, latent_transition=transition,
        control_teacher=teacher, control_reader=reader, control_weight=1.,
        prior_weight=0., world_weight=0., freeze_reference=False, **kwargs)


def test_both_is_half_sum_of_independent_readers_and_predictor_gradients():
    modules, batch = setup()
    teacher, student = modules[2:4]
    batch["group_valid"][1] = False
    both = forecast(modules, "both")(batch, control_student=student)
    full = forecast(modules, "teacher")(batch)
    ert = forecast(modules, "student")(batch, control_student=student)
    expected = .5 * (full.per_sample + ert.per_sample)
    torch.testing.assert_close(both.per_sample, expected, rtol=1e-6, atol=1e-7)
    assert both.valid.tolist() == [True, False]
    parameters = tuple(parameter for module in (modules[4], modules[5], modules[6]) for parameter in module.parameters())
    combined_gradient = torch.autograd.grad(both.per_sample.sum(), parameters, retain_graph=True, allow_unused=True)
    separate_gradient = torch.autograd.grad(expected.sum(), parameters, allow_unused=True)
    for actual, wanted in zip(combined_gradient, separate_gradient, strict=True):
        if actual is None or wanted is None:
            assert actual is wanted
        else:
            torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=1e-7)
    for name in ("objective_mse", "response_mse", "predicted_flow_mse"):
        expected_metric = .5 * (full.metrics["prior_control_"+name] + ert.metrics["prior_control_"+name])
        torch.testing.assert_close(both.metrics["prior_control_"+name], expected_metric)
    for name in ("teacher", "student"):
        assert both.metrics[f"prior_control_{name}_response_weight"] == .5
        assert both.metric_weights[f"prior_control_{name}_response_mse"] == 1
    assert all(parameter.grad is None for parameter in (*teacher.parameters(), *student.parameters()))


def test_both_uses_one_noising_prefix_sample_and_shared_teacher_target(monkeypatch):
    import psi.models.predictive_world.phases as phases

    modules, batch = setup()
    teacher, student = modules[2:4]
    del batch["action_noise"], batch["action_sigma"]
    calls, samples = {"teacher": [], "student": []}, []
    handles = []
    for name, reader in (("teacher", teacher), ("student", student)):
        handles.append(reader.register_forward_hook(
            lambda module, args, kwargs, output, name=name: calls[name].append((dict(kwargs), output)),
            with_kwargs=True))
    original = phases._action_training_sample
    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        samples.append(value[1])
        return value
    monkeypatch.setattr(phases, "_action_training_sample", capture)
    objective = forecast(modules, "both")
    result = objective(batch, control_student=student)
    for handle in handles:
        handle.remove()
    assert len(samples) == 1
    assert len(calls["teacher"]) == 2 and len(calls["student"]) == 1
    oracle_args, oracle = calls["teacher"][0]
    teacher_args, predicted_teacher = calls["teacher"][1]
    student_args, predicted_student = calls["student"][0]
    assert not oracle.requires_grad and predicted_teacher.requires_grad and predicted_student.requires_grad
    for key in ("normalized_action_samples", "action_valid", "sigma_a", "latent_group_valid",
                "qwen_context", "raw_states", "state_times"):
        torch.testing.assert_close(oracle_args[key], teacher_args[key], rtol=0, atol=0)
        torch.testing.assert_close(teacher_args[key], student_args[key], rtol=0, atol=0)
    assert teacher_args["z_raw"] is student_args["z_raw"]
    assert not oracle_args["z_raw"].requires_grad
    assert teacher_args["z_raw"].requires_grad
    assert teacher_args["sigma_a"][0, 0] == 0 and teacher_args["sigma_a"][1, :2].eq(0).all()
    teacher_loss, valid = masked_per_sample_mse(predicted_teacher, oracle, samples[0].loss_valid)
    student_loss, _ = masked_per_sample_mse(predicted_student, oracle, samples[0].loss_valid)
    torch.testing.assert_close(result.per_sample, .5 * (teacher_loss + student_loss))
    assert torch.equal(result.valid, valid)


def test_auxiliary_freezes_both_readers_without_changing_direct_action_gradients():
    modules, batch = setup()
    encoder, readout, teacher, student, adapters, history, transition, normalizer = modules
    representation = RepresentationObjective(encoder, readout, teacher)
    policy = PolicyObjective(student=student, normalizer=normalizer, history_state_adapters=adapters,
                             history_encoder=history, latent_transition=transition, freeze_world=False)
    direct = representation(batch).per_sample.mean() + policy(batch).per_sample.mean()
    readers = (*teacher.parameters(), *student.parameters())
    expected = torch.autograd.grad(direct, readers, retain_graph=True, allow_unused=True)
    before_buffers = [{name: value.clone() for name, value in reader.named_buffers()} for reader in (teacher, student)]
    forecast(modules, "both")(batch, control_student=student).per_sample.mean().backward()
    assert all(parameter.grad is None for parameter in (*readers, *encoder.parameters(), *readout.parameters()))
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in history.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in transition.parameters())
    direct.backward()
    for parameter, wanted in zip(readers, expected, strict=True):
        if wanted is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, wanted, rtol=0, atol=0)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in teacher.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in student.parameters())
    for reader, before in zip((teacher, student), before_buffers, strict=True):
        for name, value in reader.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_single_state_keeps_future_body_reconstruction_and_temporal_objectives():
    modules, batch = setup(state_count=1)
    encoder, readout, teacher, student, adapters, history, transition, normalizer = modules
    representation = RepresentationObjective(encoder, readout, teacher, body_weight=1., time_weight=.25)
    forecaster = forecast(modules, "both", readout=readout, readout_weight=.25, temporal_weight=.25)
    # forecast() already owns the readout keyword through its optional args.
    world = WorldModelObjective(representation, forecaster, normalizer)
    policy = PolicyObjective(student=student, normalizer=normalizer, history_state_adapters=adapters,
                             history_encoder=history, latent_transition=transition, freeze_world=False)
    joint = JointObjective(world, policy)
    result = joint(batch)
    result.validate()
    assert batch["observed"].raw_states.shape[1] == 1
    for metric in ("body_mse", "temporal_delta_mse", "prior_body_mse", "prior_temporal_delta_mse",
                   "prior_control_teacher_response_mse", "prior_control_student_response_mse"):
        assert metric in result.metrics and result.metrics[metric] > 0
        assert result.metric_weights[metric] > 0
    result.per_sample.mean().backward()
    for module in (encoder, readout, teacher, student, history, transition):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())


def test_future_targets_change_supervision_but_not_prior_or_deployable_student_inputs():
    modules, batch = setup(state_count=1)
    student = modules[3]
    objective = forecast(modules, "both")
    prior_inputs, student_inputs = [], []
    first = modules[6].register_forward_pre_hook(lambda module, args, kwargs: prior_inputs.append((args[0].detach().clone(), dict(kwargs))), with_kwargs=True)
    second = student.register_forward_pre_hook(lambda module, args, kwargs: student_inputs.append(dict(kwargs)), with_kwargs=True)
    baseline = objective(batch, control_student=student)
    changed = dict(batch)
    changed["future_patches"] = batch["future_patches"] + 50
    changed["future_measurements"] = batch["future_measurements"] - 10
    alternative = objective(changed, control_student=student)
    first.remove()
    second.remove()
    torch.testing.assert_close(prior_inputs[0][0], prior_inputs[1][0], rtol=0, atol=0)
    assert prior_inputs[0][1]["mode"] == prior_inputs[1][1]["mode"] == "prior"
    torch.testing.assert_close(student_inputs[0]["z_raw"], student_inputs[1]["z_raw"], rtol=0, atol=0)
    assert not torch.equal(baseline.per_sample, alternative.per_sample)
    policy = PolicyObjective(student=student, normalizer=modules[7], history_state_adapters=modules[4],
                             history_encoder=modules[5], latent_transition=modules[6], freeze_world=False)
    expected = policy(batch)
    causal = {key: value for key, value in batch.items() if not key.startswith("future_")}
    causal["group_valid"] = torch.tensor([[True, False, False]]).expand(2, -1)
    actual = policy(causal)
    torch.testing.assert_close(actual.per_sample, expected.per_sample, rtol=0, atol=0)


def test_both_consumes_same_rng_draws_as_single_reader_and_revalidates_copied_student():
    modules, batch = setup()
    student = modules[3]
    del batch["action_noise"], batch["action_sigma"], batch["_action_prefix_lengths"]
    objective = forecast(modules, "both")
    torch.manual_seed(123)
    expected = forecast(modules, "teacher")(batch)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    result = objective(batch, control_student=student)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not torch.equal(expected.per_sample, result.per_sample)
    with pytest.raises(ValueError, match="independent teacher and student"):
        objective(batch)
    incompatible = copy.deepcopy(student)
    incompatible.adapters["toy"].mean.add_(1.)
    with pytest.raises(ValueError, match="normalization disagrees"):
        objective(batch, control_student=incompatible)
