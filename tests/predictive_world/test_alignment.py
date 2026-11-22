import numpy as np
import pytest

from psi.data.predictive_world.alignment import (
    action_prefix_visibility,
    causal_floor_indices,
    select_predictive_window,
    unique_measurement_weights,
)


def test_causal_floor_never_uses_a_later_frame():
    indices, valid, actual = causal_floor_indices(
        [0.0, 0.11, 0.19, 0.31], [0.1, 0.2, 0.3], max_age_seconds=0.2
    )
    assert valid.all()
    assert indices.tolist() == [0, 2, 2]
    assert np.all(actual <= [0.1, 0.2, 0.3])


def test_default_causal_floor_does_not_snap_nearby_asynchronous_frames():
    indices, valid, actual = causal_floor_indices(
        [0.08, 0.100000001], [0.1], max_age_seconds=0.04
    )
    assert indices.tolist() == [0] and valid.all()
    assert actual.tolist() == [0.08]


def test_frame_grid_removes_query_arithmetic_error_without_changing_tail_hold():
    source = np.arange(2001, dtype=np.float64) / 50
    for anchor in (4, 20, 31, 499, 997, 1969):
        offsets = np.arange(-20, 31)
        indices, valid, actual = causal_floor_indices(
            source, source[anchor] + offsets / 50,
            max_age_seconds=0.04, frame_grid_hz=50,
        )
        expected = anchor + offsets
        np.testing.assert_array_equal(valid, expected >= 0)
        np.testing.assert_array_equal(indices[valid], expected[valid])
        np.testing.assert_array_equal(actual[valid], expected[valid] / 50)
    indices, valid, actual = causal_floor_indices(
        source, source[-1] + np.arange(4) / 50,
        max_age_seconds=0.04, frame_grid_hz=50,
    )
    assert valid.tolist() == [True, True, True, False]
    assert indices[:3].tolist() == [2000] * 3
    assert actual[:3].tolist() == [40.0] * 3


@pytest.mark.parametrize('hz', [0, -50, True, float('nan'), float('inf')])
def test_frame_grid_rejects_invalid_rate(hz):
    with pytest.raises(ValueError, match='frame_grid_hz'):
        causal_floor_indices([0, 0.02], [0], max_age_seconds=0.04, frame_grid_hz=hz)


def test_frame_grid_requires_canonical_source_and_on_grid_queries():
    with pytest.raises(ValueError, match='canonical frame grid'):
        causal_floor_indices(
            np.array([0, 0.02], dtype=np.float32), [0.02],
            max_age_seconds=0.04, frame_grid_hz=50,
        )
    with pytest.raises(ValueError, match='query_times must lie'):
        causal_floor_indices([0, 0.02], [0.019999], max_age_seconds=0.04, frame_grid_hz=50)


def test_predictive_window_does_not_turn_current_frame_into_future_truth():
    window = select_predictive_window(
        anchor_time=1.0,
        visual_timestamps=[0.6, 0.8, 1.0],
        state_timestamps=np.arange(0.6, 1.001, 0.02),
        visual_max_age_s=1.0,
        state_max_age_s=0.041,
    )
    assert window.history_visual_valid.all()
    assert not window.future_visual_valid.any()


def test_duplicate_resampling_has_one_unique_measurement_of_weight():
    weights = unique_measurement_weights([5, 5, 5, 7], [True, True, True, True])
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1])
    assert np.isclose(weights.sum(), 2.0)


def test_action_visibility_is_strictly_before_each_group_end():
    result = action_prefix_visibility(
        np.array([[0.0, 0.19, 0.2, 0.39, 0.4, 0.59]]),
        np.ones((1, 6), dtype=bool),
        (0.2, 0.4, 0.6),
    )
    assert result[0].sum(axis=1).tolist() == [2, 4, 6]
