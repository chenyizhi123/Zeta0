import pytest
import torch

from psi.data.predictive_world.contracts import NativeActionBatch
from psi.models.predictive_world.action_expert import (
    TemporalNativeActionExpert,
    assert_independent_experts,
    clone_independent_expert,
    make_committed_action_prefix,
    prepare_action_flow_training,
    sample_action_prefix_lengths,
)


def _expert():
    torch.manual_seed(23)
    return TemporalNativeActionExpert(
        {"toy": {"dim": 2, "mean": [1.0, -1.0], "std": [2.0, 4.0]}},
        {"state": 3},
        qwen_dim=6,
        context_tokens=3,
        latent_dim=4,
        latent_groups=3,
        latent_slots=2,
        state_history_count=5,
        action_steps=4,
        hidden_dim=24,
        num_blocks=2,
        num_heads=4,
        ff_dim=48,
    ).eval()


def _inputs(batch=2):
    return {
        "qwen_context": torch.randn(batch, 5, 6),
        "context_valid": torch.ones(batch, 5, dtype=torch.bool),
        "raw_states": torch.randn(batch, 5, 3),
        "state_dimension_valid": torch.ones(batch, 5, 3, dtype=torch.bool),
        "state_schema_ids": ("state",) * batch,
        "state_time_valid": torch.ones(batch, 5, dtype=torch.bool),
        "state_times": torch.linspace(-0.4, 0, 5)[None].expand(batch, -1),
        "z_raw": torch.randn(batch, 3, 2, 4),
        "latent_group_valid": torch.ones(batch, 3, dtype=torch.bool),
        "protocol_ids": ("toy",) * batch,
        "native_dims": torch.full((batch,), 2),
    }


def _native(values, valid=None):
    batch, steps, _ = values.shape
    return NativeActionBatch(
        values=values,
        valid=torch.ones_like(values, dtype=torch.bool) if valid is None else valid,
        times=torch.arange(steps).float()[None].expand(batch, -1) / 50,
        protocol_ids=("toy",) * batch,
        native_dims=torch.full((batch,), 2),
        normalization_spaces=("native",) * batch,
    )


def test_native_normalization_roundtrip_preserves_protocol_units():
    expert = _expert()
    native = _native(torch.randn(2, 4, 2))
    normalized, valid = expert.normalize_actions(native)
    assert valid.all()
    restored = expert.denormalize_actions(normalized, native.protocol_ids, native.native_dims)
    torch.testing.assert_close(restored, native.values)


@pytest.mark.parametrize('maximum', [6, 13, 15])
def test_rtc_delay_is_uniform_and_includes_both_endpoints(maximum):
    torch.manual_seed(2026)
    actions = _native(torch.zeros((maximum + 1) * 1000, 30, 2))
    lengths = sample_action_prefix_lengths(actions, max_delay=maximum)
    counts = torch.bincount(lengths, minlength=maximum + 1)
    assert lengths.min() == 0 and lengths.max() == maximum
    assert ((counts - 1000).abs() < 150).all(), counts
    assert (30 - lengths).min() == 30 - maximum


def test_rtc_delay_cap_retains_a_suffix_for_short_or_empty_windows():
    leading = torch.tensor([0, 1, 2, 6, 7, 30]).repeat(100)
    valid = (torch.arange(30)[None] < leading[:, None])[:, :, None].expand(-1, -1, 2)
    actions = _native(torch.zeros(len(leading), 30, 2), valid)
    lengths = sample_action_prefix_lengths(actions, max_delay=6)
    assert (lengths >= 0).all() and (lengths <= 6).all()
    assert (lengths[leading > 0] < leading[leading > 0]).all()
    assert (lengths[leading == 0] == 0).all()
    assert (sample_action_prefix_lengths(actions, max_delay=0) == 0).all()


def test_unspecified_rtc_delay_preserves_legacy_sampling_and_rng():
    actions = _native(torch.zeros(64, 30, 2))
    torch.manual_seed(73)
    expected = torch.floor(torch.rand(64) * 30).long()
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(73)
    actual = sample_action_prefix_lengths(actions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert actual.max() > 6


@pytest.mark.parametrize('maximum', [-1, 6.5, True])
def test_rtc_delay_rejects_invalid_bounds(maximum):
    with pytest.raises(ValueError, match='max_delay'):
        sample_action_prefix_lengths(_native(torch.zeros(1, 30, 2)), max_delay=maximum)


def test_teacher_and_student_are_distinct_parameter_objects_and_gradients():
    teacher = _expert()
    student = clone_independent_expert(teacher)
    assert_independent_experts(teacher, student)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    inputs = _inputs()
    action = torch.randn(2, 4, 2)
    output = student.velocity(
        normalized_action_samples=action,
        action_valid=torch.ones_like(action, dtype=torch.bool),
        sigma_a=torch.full((2,), 0.5),
        **{key: value for key, value in inputs.items() if key != "native_dims"},
    )
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_rtc_committed_prefix_is_restored_after_every_solver_step():
    expert = _expert()
    inputs = _inputs(batch=1)
    prefix_values = torch.tensor([[[3.0, 2.0], [5.0, 4.0], [0.0, 0.0], [0.0, 0.0]]])
    prefix_valid = torch.tensor([[[True, True], [True, True], [False, False], [False, False]]])
    prefix = _native(prefix_values, prefix_valid)
    with torch.no_grad():
        result = expert.sample(
            **inputs,
            noise=torch.randn(1, 4, 2),
            committed_prefix=prefix,
            steps=3,
        )
    torch.testing.assert_close(result[:, :2], prefix_values[:, :2])


def test_narrow_committed_prefix_is_padded_to_wider_action_noise():
    expert = _expert()
    inputs = _inputs(batch=1)
    prefix_values = torch.tensor([[[3.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    prefix_valid = torch.tensor([[[True, True], [False, False], [False, False], [False, False]]])
    prefix = _native(prefix_values, prefix_valid)
    with torch.no_grad():
        result = expert.sample(
            **inputs,
            noise=torch.randn(1, 4, 5),
            committed_prefix=prefix,
            steps=1,
        )
    assert result.shape == (1, 4, 5)
    torch.testing.assert_close(result[:, :1, :2], prefix_values[:, :1])
    assert torch.equal(result[..., 2:], torch.zeros_like(result[..., 2:]))


def test_rtc_prefix_with_a_hole_is_rejected():
    expert = _expert()
    inputs = _inputs(batch=1)
    valid = torch.tensor([[[True, True], [False, False], [True, True], [False, False]]])
    with pytest.raises(ValueError, match="left-contiguous"):
        expert.sample(
            **inputs,
            noise=torch.randn(1, 4, 2),
            committed_prefix=_native(torch.zeros(1, 4, 2), valid),
            steps=1,
        )


def test_masked_action_context_and_latent_nan_are_sanitized():
    expert = _expert()
    inputs = _inputs(batch=1)
    inputs["context_valid"][:, -1] = False
    inputs["qwen_context"][:, -1] = float("nan")
    inputs["latent_group_valid"][:, -1] = False
    inputs["z_raw"][:, -1] = float("nan")
    action = torch.randn(1, 4, 2)
    action_valid = torch.ones_like(action, dtype=torch.bool)
    action_valid[:, -1] = False
    action[:, -1] = float("nan")
    with torch.no_grad():
        velocity = expert.velocity(
            normalized_action_samples=action,
            action_valid=action_valid,
            sigma_a=torch.full((1,), 0.5),
            **{key: value for key, value in inputs.items() if key != "native_dims"},
        )
    assert torch.isfinite(velocity).all()
    assert torch.equal(velocity[:, -1], torch.zeros_like(velocity[:, -1]))


def test_partial_action_validity_is_an_explicit_model_input():
    expert = _expert()
    inputs = _inputs(batch=1)
    action = torch.zeros(1, 4, 2)
    fully_valid = torch.ones_like(action, dtype=torch.bool)
    partially_valid = fully_valid.clone()
    partially_valid[:, 1, 1] = False
    with torch.no_grad():
        complete = expert.velocity(
            normalized_action_samples=action,
            action_valid=fully_valid,
            sigma_a=torch.full((1,), 0.5),
            **{key: value for key, value in inputs.items() if key != "native_dims"},
        )
        partial = expert.velocity(
            normalized_action_samples=action,
            action_valid=partially_valid,
            sigma_a=torch.full((1,), 0.5),
            **{key: value for key, value in inputs.items() if key != "native_dims"},
        )
    assert not torch.equal(complete[:, 1, 0], partial[:, 1, 0])
    assert partial[0, 1, 1] == 0


def test_training_flow_uses_clean_prefix_step_sigma_and_suffix_only_loss():
    expert = _expert()
    actions = _native(torch.tensor([[[1.0, -1.0], [3.0, 3.0], [5.0, 7.0], [7.0, 11.0]]]))
    clean, valid = expert.normalize_actions(actions)
    prefix = make_committed_action_prefix(actions, torch.tensor([2]))
    prefix_clean, _ = expert.normalize_actions(prefix)
    noise = torch.full_like(clean, 2.0)
    sample = prepare_action_flow_training(
        clean,
        valid,
        noise,
        torch.tensor([0.5]),
        committed_prefix=prefix,
        prefix_clean=prefix_clean,
    )
    assert sample.sigma.shape == (1, 4)
    torch.testing.assert_close(sample.sigma, torch.tensor([[0.0, 0.0, 0.5, 0.5]]))
    torch.testing.assert_close(sample.mixed[:, :2], clean[:, :2])
    assert not sample.loss_valid[:, :2].any()
    assert sample.loss_valid[:, 2:].all()
