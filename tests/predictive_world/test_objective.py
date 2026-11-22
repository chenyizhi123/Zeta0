import torch

from psi.data.predictive_world.objective import (
    UpdateLossNormalizer,
    flow_matching_target,
    masked_per_sample_mse,
)


def test_masked_mse_is_per_sample_and_nan_safe():
    prediction = torch.tensor([[2.0, float("nan")], [1.0, 1.0]], requires_grad=True)
    target = torch.zeros_like(prediction)
    per_sample, valid = masked_per_sample_mse(
        prediction, target, torch.tensor([[True, False], [True, True]])
    )
    torch.testing.assert_close(per_sample, torch.tensor([4.0, 1.0]))
    denominator = UpdateLossNormalizer(distributed=False).denominator(
        [torch.tensor([True]), torch.tensor([True])]
    )
    loss = UpdateLossNormalizer.scale_micro(per_sample, valid, denominator)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    assert prediction.grad[0, 1] == 0


def test_accumulation_uses_one_update_denominator_without_extra_division():
    normalizer = UpdateLossNormalizer(distributed=False)
    denominator = normalizer.denominator(
        [torch.tensor([True, False]), torch.tensor([True])]
    )
    first = normalizer.scale_micro(
        torch.tensor([2.0, 100.0]), torch.tensor([True, False]), denominator
    )
    second = normalizer.scale_micro(torch.tensor([4.0]), torch.tensor([True]), denominator)
    torch.testing.assert_close(first + second, torch.tensor(3.0))


def test_masked_mse_supports_unique_measurement_weights():
    prediction = torch.tensor([[1.0, 3.0, 5.0]])
    target = torch.zeros_like(prediction)
    per_sample, valid = masked_per_sample_mse(
        prediction,
        target,
        torch.ones_like(prediction, dtype=torch.bool),
        element_weight=torch.tensor([[0.5, 0.5, 1.0]]),
    )
    # The first two resampled slots jointly have the weight of the third slot.
    torch.testing.assert_close(per_sample, torch.tensor([(0.5 + 4.5 + 25.0) / 2.0]))
    assert valid.item()


def test_flow_matching_direction_uses_noise_minus_clean():
    clean = torch.tensor([[2.0]])
    noise = torch.tensor([[6.0]])
    mixed, target = flow_matching_target(clean, noise, torch.tensor([0.25]))
    torch.testing.assert_close(mixed, torch.tensor([[3.0]]))
    torch.testing.assert_close(target, torch.tensor([[4.0]]))
