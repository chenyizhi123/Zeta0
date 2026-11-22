import numpy as np
import pytest

from psi.data.predictor_pretrain.alignment import (
    align_visual_anchors,
    causal_state_history_indices,
)


def test_visual_alignment_uses_seconds_and_never_pads_episode_tail():
    timestamps = np.arange(0.0, 2.0, 1 / 30, dtype=np.float64)
    anchors = align_visual_anchors(timestamps, nominal_fps=30)
    assert [row.current_index for row in anchors] == [0, 15, 30]
    assert [row.target_index for row in anchors] == [18, 33, 48]
    assert all(abs(row.actual_delta_t - 0.6) < 1e-8 for row in anchors)
    assert all(row.target_index < len(timestamps) for row in anchors)


def test_visual_alignment_rejects_missing_target_outside_tolerance():
    timestamps = [0.0, 0.5, 1.0, 1.5]
    assert align_visual_anchors(timestamps, nominal_fps=30) == []
    with pytest.raises(ValueError, match="strictly increasing"):
        align_visual_anchors([0.0, 0.0, 1.0], nominal_fps=30)


def test_state_history_is_causal_and_marks_stale_slots_invalid():
    timestamps = np.arange(0.0, 1.01, 0.02)
    indices, valid, selected = causal_state_history_indices(
        timestamps, current_timestamp=1.0, nominal_state_hz=50
    )
    assert valid.all()
    assert np.all(selected <= np.arange(0.86, 1.001, 0.02) + 1e-12)
    sparse_indices, sparse_valid, _ = causal_state_history_indices(
        [0.0, 0.5, 1.0], current_timestamp=1.0, nominal_state_hz=50
    )
    assert sparse_valid.tolist() == [False] * 7 + [True]
    assert sparse_indices[-1] == 2
