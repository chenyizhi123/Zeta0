import pytest
import torch

from psi.data.predictive_world.contracts import (
    FutureSupervision,
    NativeActionBatch,
    ObservedHistory,
    PredictiveWorldBatch,
    RaggedPatchBatch,
    SampleMetadata,
    assert_observed_only,
    sanitize_masked_tensor,
)


def _actions(batch=2, horizon=30):
    return NativeActionBatch(
        values=torch.zeros(batch, horizon, 3),
        valid=torch.ones(batch, horizon, 3, dtype=torch.bool),
        times=torch.arange(horizon).float()[None].expand(batch, -1) / 50,
        protocol_ids=("toy",) * batch,
        native_dims=torch.full((batch,), 3),
        normalization_spaces=("native",) * batch,
    )


def _batch():
    patches = RaggedPatchBatch(
        values=torch.zeros(6, 4),
        offsets=torch.tensor([0, 2, 4, 6]),
        grid_hw=torch.tensor([[1, 2], [1, 2], [1, 2]]),
        content_valid=torch.ones(6, dtype=torch.bool),
        sample_indices=torch.tensor([0, 0, 0]),
        group_indices=torch.tensor([0, 1, 2]),
        camera_ids=("primary",) * 3,
        actual_times=torch.tensor([0.2, 0.4, 0.6]),
    )
    observed = ObservedHistory(
        qwen_inputs={"input_ids": torch.ones(2, 2, dtype=torch.long)},
        raw_states=torch.zeros(2, 21, 5),
        state_valid=torch.ones(2, 21, 5, dtype=torch.bool),
        state_times=torch.linspace(-0.4, 0, 21)[None].expand(2, -1),
        state_schema_ids=("s", "s"),
        action_protocol_ids=("toy", "toy"),
        action_native_dims=torch.tensor([3, 3]),
        action_normalization_spaces=("native", "native"),
    )
    future = FutureSupervision(
        visual=patches,
        measured_states=torch.zeros(2, 30, 2),
        measured_valid=torch.ones(2, 30, 2, dtype=torch.bool),
        measured_times=torch.arange(1, 31).float()[None].expand(2, -1) / 50,
        measured_source_indices=torch.arange(30)[None].expand(2, -1),
        latent_target_valid=torch.ones(2, 3, dtype=torch.bool),
    )
    metadata = SampleMetadata(
        sample_ids=("a", "b"), source_ids=("x", "x"), episode_ids=("1", "2"),
        occurrence_indices=torch.tensor([0, 1]), split="train",
    )
    return PredictiveWorldBatch(observed, future, _actions(), metadata)


def test_predictive_world_batch_keeps_future_outside_deployable_boundary():
    batch = _batch()
    batch.validate()
    assert assert_observed_only(batch.deployable_observed()) is batch.observed
    with pytest.raises(TypeError, match="ObservedHistory only"):
        assert_observed_only(batch)


def test_invalid_nan_is_cleaned_but_valid_nan_fails():
    values = torch.tensor([[1.0, float("nan")]])
    cleaned = sanitize_masked_tensor(values, torch.tensor([[True, False]]), name="toy")
    assert torch.equal(cleaned, torch.tensor([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        sanitize_masked_tensor(values, torch.tensor([[True, True]]), name="toy")


def test_future_floor_sample_cannot_be_labeled_as_observed():
    batch = _batch()
    observed = ObservedHistory(
        **{**batch.observed.__dict__, "state_times": torch.ones(2, 21) * 0.01}
    )
    with pytest.raises(ValueError, match="cannot be later"):
        observed.validate(batch_size=2, state_count=21, action_horizon=30)
