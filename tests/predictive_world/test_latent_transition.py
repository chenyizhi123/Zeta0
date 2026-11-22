import torch
from torch import nn
import pytest

from psi.data.predictive_world.contracts import NativeActionBatch
from psi.models.predictive_world.latent_transition import LatentNormalizer, TemporalLatentTransition


def _condition(values):
    batch, steps, width = values.shape
    return NativeActionBatch(
        values=values,
        valid=torch.ones_like(values, dtype=torch.bool),
        times=torch.tensor([[0.0, 0.1, 0.25, 0.45]]).expand(batch, -1),
        protocol_ids=("toy",) * batch,
        native_dims=torch.full((batch,), width),
        normalization_spaces=("native",) * batch,
    )


def _model():
    torch.manual_seed(7)
    model = TemporalLatentTransition(
        {"toy": {"dim": 2, "mean": [0.0, 0.0], "std": [1.0, 1.0]}},
        context_dim=12,
        latent_shape=(3, 2, 4),
        hidden_dim=16,
        num_blocks=2,
        num_heads=4,
        ff_dim=32,
    )
    # Move beyond the deliberately zero-initialized readout to exercise dynamics.
    nn.init.normal_(model.output_projection[-1].weight, std=0.05)
    return model.eval()


def test_world_late_action_cannot_change_early_prediction():
    model = _model()
    context = torch.randn(1, 3, 12)
    values = torch.zeros(1, 4, 2)
    changed = values.clone()
    changed[:, 2:] = 100
    base = model(context, mode="world", action_condition=_condition(values))
    other = model(context, mode="world", action_condition=_condition(changed))
    torch.testing.assert_close(base[:, 0], other[:, 0], rtol=0, atol=0)
    assert not torch.allclose(base[:, 2], other[:, 2])


def test_segment_boundary_is_exclusive_and_earlier_effects_propagate():
    model = _model()
    context = torch.randn(1, 3, 12)
    base = _condition(torch.zeros(1, 4, 2))
    base.times[:, 2] = 0.2
    from dataclasses import replace
    values = base.values.clone()
    values[:, 2] = 4
    first = model(context, mode="world", action_condition=base)
    second = model(context, mode="world", action_condition=replace(base, values=values))
    torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
    assert not torch.allclose(first[:, 1], second[:, 1])
    assert not torch.allclose(first[:, 2], second[:, 2])


def test_world_preserves_within_segment_command_timing():
    from dataclasses import replace
    model = _model()
    context = torch.randn(1, 3, 12)
    base = _condition(torch.tensor([[[1., -1.], [-2., 2.], [0., 0.], [0., 0.]]]))
    swapped = replace(base, values=base.values[:, [1, 0, 2, 3]])
    first = model(context, mode="world", action_condition=base)
    second = model(context, mode="world", action_condition=swapped)
    assert not torch.allclose(first[:, 0], second[:, 0], atol=1e-6, rtol=1e-6)


def test_direct_prior_is_deterministic_and_does_not_accept_noise_or_labels():
    model = _model()
    context = torch.randn(1, 3, 12)
    first = model(context, mode="prior")
    torch.randn(100)
    torch.testing.assert_close(first, model(context, mode="prior"), rtol=0, atol=0)
    assert first.shape == (1, 3, 2, 4)
    for argument in ("noise", "sigma_z", "latent_group_valid"):
        with pytest.raises(TypeError):
            model(context, mode="prior", **{argument: torch.zeros(1)})
    with pytest.raises(ValueError, match="requires"):
        model(context, mode="world")


def test_masked_prefix_values_do_not_enter_prediction_and_zero_is_known():
    from dataclasses import replace
    model = _model()
    context = torch.randn(1, 3, 12)
    base = _condition(torch.zeros(1, 4, 2))
    valid = torch.zeros_like(base.valid)
    valid[:, :2] = True
    base = replace(base, valid=valid)
    changed = replace(base, values=base.values.masked_fill(~valid, float("nan")))
    first = model(context, mode="prior", action_condition=base)
    second = model(context, mode="prior", action_condition=changed)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.allclose(first, model(context, mode="prior"))
    empty = replace(base, valid=torch.zeros_like(valid))
    torch.testing.assert_close(
        model(context, mode="prior", action_condition=empty),
        model(context, mode="prior"), atol=1e-6, rtol=1e-6,
    )


def test_parallel_blocks_run_once_and_later_loss_reaches_earlier_group():
    model = _model()
    memories, calls = [], []
    def capture(_module, _inputs, output):
        output.retain_grad()
        memories.append(output)
    handle = model.blocks[0].register_forward_hook(capture)
    handle2 = model.blocks[0].register_forward_hook(lambda *_args: calls.append(1))
    output = model(torch.randn(2, 3, 12), mode="prior")
    output[:, -1].square().mean().backward()
    handle.remove()
    handle2.remove()
    assert len(calls) == len(memories) == 1
    assert memories[0].shape == (2, 6, 16)
    assert memories[0].grad is not None and memories[0].grad[:, :2].abs().sum() > 0
    assert model.blocks[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_masked_context_nan_does_not_contaminate_rollout():
    model = _model()
    context = torch.randn(1, 3, 12)
    valid = torch.tensor([[True, True, False]])
    base = model(context, context_valid=valid, mode="prior")
    context[:, -1] = float("nan")
    changed = model(context, context_valid=valid, mode="prior")
    assert torch.isfinite(changed).all()
    torch.testing.assert_close(base, changed, rtol=0, atol=0)


def test_latent_normalizer_roundtrip_and_fit_gate():
    normalizer = LatentNormalizer((3, 2, 4), std_floor=0.05)
    latent = torch.randn(8, 3, 2, 4)
    with pytest.raises(RuntimeError, match="not been fitted"):
        normalizer.whiten(latent)
    normalizer.fit(latent, torch.ones(8, 3, dtype=torch.bool))
    torch.testing.assert_close(normalizer.unwhiten(normalizer.whiten(latent)), latent)
