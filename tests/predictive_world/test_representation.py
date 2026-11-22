import torch

from psi.models.predictive_world.representation import (
    EMAFutureTrajectoryEncoder,
    FutureTrajectoryEncoder,
    WorldReadout,
    representation_regularizer,
    representation_regularizer_per_sample,
)


def _encoder_inputs():
    torch.manual_seed(11)
    return {
        "anchor_patches": torch.randn(2, 3, 6),
        "anchor_patch_valid": torch.ones(2, 3, dtype=torch.bool),
        "current_measurements": torch.randn(2, 4),
        "current_measurement_valid": torch.ones(2, 4, dtype=torch.bool),
        "future_patches": torch.randn(2, 3, 5, 6),
        "future_patch_valid": torch.ones(2, 3, 5, dtype=torch.bool),
        "future_scene_times": torch.tensor([[0.2, 0.4, 0.6], [0.2, 0.4, 0.6]]),
        "future_measurements": torch.randn(2, 3, 2, 4),
        "future_measurement_valid": torch.ones(2, 3, 2, 4, dtype=torch.bool),
        "future_measurement_times": torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]] * 2
        ),
        "group_valid": torch.ones(2, 3, dtype=torch.bool),
    }


def _encoder():
    torch.manual_seed(5)
    return FutureTrajectoryEncoder(
        visual_dim=6,
        measurement_dim=4,
        latent_shape=(3, 8, 4),
        hidden_dim=16,
        num_blocks=2,
        num_heads=4,
        ff_dim=32,
    ).eval()


def test_future_encoder_first_group_cannot_read_later_evidence():
    model = _encoder()
    inputs = _encoder_inputs()
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in inputs.items()}
    changed["future_patches"][:, 1:] += 100
    changed["future_measurements"][:, 1:] -= 100
    with torch.no_grad():
        original = model(**inputs)
        modified = model(**changed)
    torch.testing.assert_close(original[:, 0], modified[:, 0])
    assert not torch.equal(original[:, 2], modified[:, 2])


def test_world_readout_first_group_cannot_read_later_latents():
    torch.manual_seed(13)
    model = WorldReadout(
        visual_dim=6,
        measurement_dim=4,
        latent_shape=(3, 8, 4),
        max_scene_patches=5,
        body_points_per_group=2,
        hidden_dim=16,
        num_blocks=2,
        num_heads=4,
        ff_dim=32,
    ).eval()
    anchor = torch.randn(2, 3, 6)
    anchor_valid = torch.ones(2, 3, dtype=torch.bool)
    latent = torch.randn(2, 3, 8, 4)
    changed = latent.clone()
    changed[:, 1:] += 100
    group_valid = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        scene_a, body_a = model(
            anchor_patches=anchor,
            anchor_patch_valid=anchor_valid,
            z_raw=latent,
            group_valid=group_valid,
            scene_patch_count=5,
        )
        scene_b, body_b = model(
            anchor_patches=anchor,
            anchor_patch_valid=anchor_valid,
            z_raw=changed,
            group_valid=group_valid,
            scene_patch_count=5,
        )
    torch.testing.assert_close(scene_a[:, 0], scene_b[:, 0])
    torch.testing.assert_close(body_a[:, 0], body_b[:, 0])
    assert not torch.equal(scene_a[:, 2], scene_b[:, 2])


def test_representation_regularizer_is_finite_with_no_valid_groups():
    latent = torch.zeros(2, 3, 8, 4, requires_grad=True)
    loss = representation_regularizer(latent, torch.zeros(2, 3, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(latent.grad, torch.zeros_like(latent))


def test_representation_regularizer_penalizes_collapsed_valid_slots_per_sample():
    collapsed = torch.ones(2, 3, 8, 4)
    diverse = collapsed.clone()
    diverse[0, 0, :, 0] = torch.linspace(-1, 1, 8)
    valid = torch.tensor([[True, False, False], [False, False, False]])
    collapsed_loss, collapsed_valid = representation_regularizer_per_sample(collapsed, valid)
    diverse_loss, diverse_valid = representation_regularizer_per_sample(diverse, valid)
    assert torch.equal(collapsed_valid, torch.tensor([True, False]))
    assert torch.equal(diverse_valid, collapsed_valid)
    assert diverse_loss[0] < collapsed_loss[0]
    assert diverse_loss[1].item() == 0


def test_ema_future_encoder_publishes_target_and_checkpoints_stage_markers():
    online = _encoder().train()
    stable = EMAFutureTrajectoryEncoder(online, decay=0.5).train()
    name, parameter = next(iter(stable.online_encoder.named_parameters()))
    target_before = dict(stable.target_encoder.named_parameters())[name].detach().clone()
    with torch.no_grad():
        parameter.add_(2.0)
    assert stable.update_target(0)
    assert not stable.update_target(0)
    target_after = dict(stable.target_encoder.named_parameters())[name]
    torch.testing.assert_close(target_after, target_before + 1.0)
    assert not stable.target_encoder.training
    stable.mark_normalizer_update(0)
    stable.freeze_normalizer_stage()
    restored = EMAFutureTrajectoryEncoder(_encoder(), decay=0.5)
    restored.load_state_dict(stable.state_dict(), strict=True)
    assert int(restored.last_target_update) == 0
    assert int(restored.last_normalizer_update) == 0
    assert bool(restored.normalizer_frozen)
    with torch.no_grad():
        published = restored(**_encoder_inputs())
        expected = restored.target_encoder(**_encoder_inputs())
    torch.testing.assert_close(published, expected)


def test_future_encoder_ignores_nonfinite_times_in_masked_measurements():
    model = _encoder().train()
    inputs = _encoder_inputs()
    inputs['future_measurement_valid'][:, 0, 1] = False
    inputs['future_measurement_times'][:, 0, 1] = 0
    expected = model(**inputs)
    inputs['future_measurement_times'][:, 0, 1] = float('nan')
    actual = model(**inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_future_encoder_rejects_nonfinite_time_for_valid_measurement():
    import pytest

    inputs = _encoder_inputs()
    inputs['future_measurement_times'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='future measurement times'):
        _encoder()(**inputs)
