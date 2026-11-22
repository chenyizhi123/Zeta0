import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "hiw_converter", Path(__file__).resolve().parents[1] / "scripts/data/convert_hiw500_mcap.py"
)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def test_causal_selection_never_uses_a_future_frame():
    indices, ages = converter.causal_indices([100, 130, 170], [129, 131, 179], 30)
    assert indices.tolist() == [0, 1, 2]
    assert ages.tolist() == [29, 1, 9]


@pytest.mark.parametrize("times,queries", [
    ([100, 100], [100]), ([100, 130], [99]), ([100, 130], [200]),
])
def test_missing_stale_or_duplicate_samples_are_not_silently_padded(times, queries):
    with pytest.raises(ValueError):
        converter.causal_indices(times, queries, 30)


def test_action_is_command_not_measured_end_effector_state():
    state, action = converter.wbc_vectors({
        "pivot": list(range(7)), "ee_state": [1]*12, "ee_action": [2]*12,
        "gripper_controls": dict(zip(converter.GRIPPERS, [3, 4, 5, 6])),
    })
    assert state.shape == action.shape == (23,)
    np.testing.assert_array_equal(state[7:19], [1]*12)
    np.testing.assert_array_equal(action[7:19], [2]*12)
    np.testing.assert_array_equal(action[-4:], [3, 4, 5, 6])
