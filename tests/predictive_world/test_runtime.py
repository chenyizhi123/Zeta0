import pytest
import torch

from psi.data.predictive_world.runtime import PredictiveWorldCollator, PredictiveWorldTrainingCollator
from psi.config.predictive_world import PredictiveWorldDataConfig
from psi.data.predictive_world.index import temporal_abi_from_data_config


def _row(sample_id, occurrence):
    return {
        "sample_id": sample_id,
        "source_id": "source",
        "episode_id": "episode",
        "occurrence_index": occurrence,
        "split": "train",
        "qwen_inputs": {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.ones(3, dtype=torch.long),
        },
        "raw_states": torch.randn(3, 2),
        "state_valid": torch.ones(3, 2, dtype=torch.bool),
        "state_times": torch.tensor([-0.04, -0.02, 0.0]),
        "state_schema_id": "state",
        "actions": torch.randn(4, 2),
        "action_valid": torch.ones(4, 2, dtype=torch.bool),
        "action_times": torch.tensor([0.0, 0.02, 0.04, 0.06]),
        "action_protocol_id": "native",
        "action_normalization_space": "native",
        "world_action_group_valid": (True, True, True),
        "current_measurements": torch.randn(2),
        "current_measurement_valid": torch.ones(2, dtype=torch.bool),
        "future_patch_fields": [
            [{"values": torch.randn(4, 5), "grid_hw": (2, 2), "camera_id": "head", "actual_time": 0.2}],
            [{"values": torch.randn(2, 5), "grid_hw": (1, 2), "camera_id": "head", "actual_time": 0.4}],
            [{"values": torch.randn(1, 5), "grid_hw": (1, 1), "camera_id": "head", "actual_time": 0.6}],
        ],
        "future_measurements": torch.randn(3, 2),
        "future_measurement_valid": torch.ones(3, 2, dtype=torch.bool),
        "future_measurement_times": torch.tensor([0.2, 0.4, 0.6]),
        "future_measurement_source_indices": torch.tensor([10, 20, 30]),
    }


def test_collator_preserves_ragged_geometry_native_protocol_and_metadata():
    batch = PredictiveWorldCollator(groups=3, state_count=3, action_horizon=4)(
        [_row("a", 7), _row("b", 8)]
    )
    assert batch.future.visual.values.shape == (14, 5)
    assert batch.future.visual.offsets.tolist() == [0, 4, 6, 7, 11, 13, 14]
    assert batch.observed.action_native_dims.tolist() == [2, 2]
    assert batch.metadata.sample_ids == ("a", "b")
    assert batch.future.latent_target_valid.all()


def test_collator_rejects_split_mixing():
    first, second = _row("a", 0), _row("b", 1)
    second["split"] = "validation"
    try:
        PredictiveWorldCollator(groups=3, state_count=3, action_horizon=4)([first, second])
    except ValueError as error:
        assert "split" in str(error)
    else:
        raise AssertionError("mixed splits were accepted")


def test_fixed_action_condition_grid_preserves_logged_times_values_and_masks():
    row = _row("a", 0)
    row["action_times"] = torch.tensor([.015, .036, .049, .073])
    collator = PredictiveWorldTrainingCollator(
        future_measurement_offsets_s=(.2,.4,.6), state_count=3, action_horizon=4,
        action_condition_time_mode="fixed_action_grid_v1", action_hz=50)
    batch = collator([row])
    torch.testing.assert_close(batch["actions"].times, torch.tensor([[.02,.04,.06,.08]]))
    torch.testing.assert_close(batch["typed_batch"].demonstrated.times[0], row["action_times"])
    torch.testing.assert_close(batch["actions"].values[0], row["actions"])
    torch.testing.assert_close(batch["actions"].valid[0], row["action_valid"])
    torch.testing.assert_close(batch["observed"].state_times[0], row["state_times"])
    legacy = PredictiveWorldTrainingCollator(
        future_measurement_offsets_s=(.2,.4,.6), state_count=3, action_horizon=4)([row])
    torch.testing.assert_close(legacy["actions"].times[0], row["action_times"])


def test_training_collator_uses_fixed_body_capacity_and_rejects_overflow():
    row = _row("a", 0)
    batch = PredictiveWorldTrainingCollator(
        future_measurement_offsets_s=(0.2, 0.4, 0.6),
        future_group_ends_s=(0.2, 0.4, 0.6), state_count=3,
        action_horizon=4, body_points_per_group=2,
    )([row])
    assert batch["future_measurements"].shape == (1, 3, 2, 2)
    torch.testing.assert_close(
        batch["future_measurement_weights"][:, :, 0],
        torch.ones(1, 3, 2),
    )
    row["future_measurement_times"] = torch.tensor([0.1, 0.15, 0.2])
    try:
        PredictiveWorldTrainingCollator(
            future_measurement_offsets_s=(0.1, 0.15, 0.2),
            future_group_ends_s=(0.2, 0.4, 0.6), state_count=3,
            action_horizon=4, body_points_per_group=2,
        )([row])
    except ValueError as error:
        assert "body points" in str(error)
    else:
        raise AssertionError("body-point overflow was accepted")


def test_training_collator_does_not_overweight_repeated_physical_measurements():
    row = _row("a", 0)
    row["future_measurement_source_indices"] = torch.tensor([10, 10, 20])
    batch = PredictiveWorldTrainingCollator(
        future_measurement_offsets_s=(0.2, 0.4, 0.6),
        future_group_ends_s=(0.2, 0.4, 0.6), state_count=3,
        action_horizon=4, body_points_per_group=2,
    )([row])
    weights = batch["future_measurement_weights"][0, :, 0, 0]
    torch.testing.assert_close(weights, torch.tensor([0.5, 0.5, 1.0]))


def test_missing_measurements_and_asof_boundaries_do_not_change_query_slots():
    row = _row("a", 0)
    queries = (0.02, 0.04, 0.22, 0.24, 0.42, 0.44)
    row["future_measurements"] = torch.arange(12).reshape(6, 2).float()
    row["future_measurement_valid"] = torch.ones(6, 2, dtype=torch.bool)
    # Query .22 has no observation; .24's causal-floor measurement is at .20.
    row["future_measurement_valid"][2] = False
    row["future_measurements"][2] = 0
    row["future_measurement_times"] = torch.tensor([0.02, 0.04, 0, 0.20, 0.42, 0.44])
    row["future_measurement_source_indices"] = torch.tensor([1, 2, -1, 10, 21, 22])
    batch = PredictiveWorldTrainingCollator(
        future_measurement_offsets_s=queries, state_count=3,
        action_horizon=4, body_points_per_group=2,
    )([row])
    assert batch["future_measurement_valid"].any(-1).tolist() == [[[True, True], [False, True], [True, True]]]
    torch.testing.assert_close(batch["future_measurements"][0, 1, 1], torch.tensor([6., 7.]))
    torch.testing.assert_close(batch["future_measurement_times"][0, 1], torch.tensor([0., 0.20]))
    assert batch["future_measurement_pair_valid"].all()
    paired = batch["future_measurement_valid"][:, 1:] & batch["future_measurement_valid"][:, :-1]
    assert paired.any(-1).tolist() == [[[False, True], [False, True]]]
    assert batch["future_measurement_weights"][0, 1, 0].eq(0).all()
    assert batch["group_valid"].all()  # Neither the sample nor its visual targets were dropped.


def test_temporal_pairs_require_matching_relative_query_offsets():
    row = _row("a", 0)
    collator = PredictiveWorldTrainingCollator(
        future_measurement_offsets_s=(0.02, 0.24, 0.44),
        state_count=3, action_horizon=4, body_points_per_group=2,
    )
    batch = collator([row])
    assert batch["future_measurement_pair_valid"].tolist() == [[False, False], [True, False]]
    assert batch["future_measurement_valid"].sum() == 6
    row["future_measurement_times"] = row["future_measurement_times"][:2]
    with pytest.raises(ValueError, match="every nominal query slot"):
        collator([row])


def test_production_query_grid_preserves_all_thirty_slots():
    queries = temporal_abi_from_data_config(PredictiveWorldDataConfig())["future_measurement_offsets_s"]
    collator = PredictiveWorldTrainingCollator(future_measurement_offsets_s=queries)
    assert [indices.tolist() for indices in collator.measurement_indices] == [
        list(range(0, 10)), list(range(10, 20)), list(range(20, 30)),
    ]
    assert collator.measurement_pair_valid.shape == (2, 10)
    assert collator.measurement_pair_valid.all()
