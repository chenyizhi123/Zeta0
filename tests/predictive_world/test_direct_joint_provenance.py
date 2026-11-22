"""Imported direct-joint pretraining retains its real ancestry on adaptation."""
import copy

import pytest

from psi.models.predictive_world.checkpoint import _validate_phase_metadata


def direct_origin():
    return {
        "training_origin": {"kind": "joint_from_scratch", "optimizer_update_offset": 0, "seed": 101},
        "optimization_contract": {"phase": "joint", "from_scratch": True, "seed": 101,
                                  "normalizer_calibration_updates": 500, "forecast_warmup_updates": 500},
    }


def test_direct_joint_and_adapted_joint_preserve_ancestry():
    origin = direct_origin()
    _validate_phase_metadata("joint", origin)
    adapted = {"initialization": {
        "weights_sha256": "a" * 64, "source_config_sha256": "b" * 64,
        "source_joint_provenance": {"phase": "joint", "completed_updates": 15000,
                                    "weights_sha256": "a" * 64, "resolved_config_sha256": "b" * 64,
                                    "metadata": origin},
    }}
    _validate_phase_metadata("joint", adapted)
    corrupted = copy.deepcopy(adapted)
    corrupted["initialization"]["source_joint_provenance"]["weights_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="provenance mismatch"):
        _validate_phase_metadata("joint", corrupted)


@pytest.mark.parametrize("damage", ["seed", "parent", "pretrained", "calibration", "missing_origin"])
def test_direct_joint_rejects_contradictory_origin(damage):
    metadata = direct_origin()
    if damage == "seed":
        metadata["training_origin"]["seed"] = 42
    elif damage == "parent":
        metadata["parent_world_model"] = {"phase": "world_model"}
    elif damage == "pretrained":
        metadata["optimization_contract"]["init_joint"] = "pretrained"
    elif damage == "calibration":
        metadata["optimization_contract"]["normalizer_calibration_updates"] = 0
    else:
        del metadata["training_origin"]
    with pytest.raises(ValueError):
        _validate_phase_metadata("joint", metadata)
