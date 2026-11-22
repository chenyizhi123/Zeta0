import numpy as np
import pytest

from psi.data.predictive_world.action_grid_projection import extend_action_row


@pytest.mark.parametrize('anchor', [0, 20, 48, 49])
def test_extended_labels_follow_episode_clock_and_tail_masks(anchor):
    n = 50
    queries = anchor + np.arange(1, 51)
    selected = np.minimum(queries, n-1)
    available = queries-selected <= 2
    # Causal floor uses slot zero for unavailable lookups, without crossing episodes.
    selected = np.where(available, selected, 0)
    valid = available & (selected > anchor)
    relative = np.where(available, (selected-anchor)/50, np.nan)
    row = dict(anchor_time=anchor/50, action_indices=(selected[:30]+100).tolist(),
               action_valid=valid[:30].tolist(), action_actual_times=relative[:30].tolist(),
               future_visual_indices=[10, 20, 30], sample_id='unchanged')
    result = extend_action_row(row, times=np.arange(n)/50, native_rows=np.arange(n)+100,
                               horizon=50, max_age=.04)
    np.testing.assert_array_equal(result['action_indices'], selected+100)
    np.testing.assert_array_equal(result['action_valid'], valid)
    np.testing.assert_array_equal(result['action_actual_times'], relative)
    assert result['future_visual_indices'] is row['future_visual_indices']
    assert len(row['action_indices']) == 30
    row['action_indices'][0] += 1
    with pytest.raises(ValueError, match='disagree'):
        extend_action_row(row, times=np.arange(n)/50, native_rows=np.arange(n)+100,
                          horizon=50, max_age=.04)
