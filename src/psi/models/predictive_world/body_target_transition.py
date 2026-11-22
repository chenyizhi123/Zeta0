"""Explicit Arena arm14 -> measured whole-body q29 coordinate transition.

The original 29 slots describe arms, hands and height, with only the arm14
available in Arena. This experiment keeps the arm slots and replaces the 15
unavailable slots by measured leg/waist joints. Equal widths do not imply equal
semantics: their input columns and output rows must be reset explicitly.
"""
from __future__ import annotations

import copy

from psi.data.predictive_world.cache import canonical_sha256
from psi.data.predictive_world.measurement_schema import MeasurementSchema

LAYOUT = "g1_measured_arm14_leg12_waist3_v1"
SCHEMA = "sonic_measured_whole_body_q29_v1"
REAL_SCHEMA = "real_g1_measured_whole_body_q29_brainco12_v2"
ARM_NAMES = tuple(
    f"state.dof_pos.{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
)
EXTRA_NAMES = tuple(
    f"state.dof_pos.{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
) + tuple(f"state.dof_pos.waist_{axis}_joint" for axis in ("yaw", "roll", "pitch"))
ARM_SLOTS = tuple(f"{side}_arm_joint_{i}" for side in ("left", "right") for i in range(7))
OLD_SLOTS = ARM_SLOTS + tuple(
    f"{side}_hand_joint_{i}" for side in ("left", "right") for i in range(7)
) + ("base_height",)


def whole_body_schema(arm_schema, names, body_mean, body_std):
    """Gather only the 29 named, measured dof_pos fields; never action labels."""
    if len(names) != 64 or len(set(names)) != 64 or len(body_mean) != 64 or len(body_std) != 64:
        raise ValueError("whole-body targets require the audited state64 schema and train statistics")
    old = arm_schema["canonical_body"]
    if tuple(old["slots"]) != OLD_SLOTS or old["raw_indices"][14:] != [None] * 15:
        raise ValueError("whole-body transition requires the original Arena arm14 mapping")
    indices = [names.index(name) for name in ARM_NAMES + EXTRA_NAMES]
    if set(indices) != set(range(6, 35)) or old["raw_indices"][:14] != indices[:14]:
        raise ValueError("named measured joints disagree with the audited Arena state64 order")
    result = copy.deepcopy(arm_schema)
    result["canonical_body"] = {
        "layout_id": LAYOUT,
        "slots": list(ARM_SLOTS) + [name.removeprefix("state.dof_pos.") for name in EXTRA_NAMES],
        "raw_indices": indices,
        "mean": old["mean"][:14] + [body_mean[i] for i in indices[14:]],
        "std": old["std"][:14] + [max(body_std[i], 1e-6) for i in indices[14:]],
        "normalization_space": old["normalization_space"],
    }
    MeasurementSchema.from_mapping(SCHEMA, result)
    return result


def real_whole_body_schema(arm_schema, names, body_mean, body_std):
    """Use audited SONIC state41 measurements: arm14, leg12, waist3 (no commands)."""
    from psi.data.predictive_world.sonic_lerobot_post_training import STATE_NAMES
    old = arm_schema["canonical_body"]
    if (list(names) != STATE_NAMES or len(body_mean) != 41 or len(body_std) != 41
            or arm_schema["width"] != 41 or tuple(old["slots"]) != OLD_SLOTS
            or old["raw_indices"] != list(range(15, 29)) + [None] * 15):
        raise ValueError("real whole-body targets require audited native state41 and arm14 mapping")
    result = copy.deepcopy(arm_schema)
    indices = list(range(15, 29)) + list(range(15))
    result["canonical_body"] = dict(
        layout_id=LAYOUT,
        slots=list(ARM_SLOTS) + [name.removeprefix("state.dof_pos.") for name in EXTRA_NAMES],
        raw_indices=indices,
        mean=old["mean"][:14] + [body_mean[i] for i in indices[14:]],
        std=old["std"][:14] + [max(body_std[i], 1e-6) for i in indices[14:]],
        normalization_space=old["normalization_space"],
    )
    MeasurementSchema.from_mapping(REAL_SCHEMA, result)
    return result


def make_body_target_transition(source, source_payload, target_registry, *, schema=SCHEMA):
    if source.get("phase") != "joint" or canonical_sha256(source_payload) != source["metadata"]["resolved_config_sha256"]:
        raise ValueError("body-target transition requires the exact joint source/config")
    if schema not in {SCHEMA, REAL_SCHEMA}:
        raise ValueError("unsupported whole-body measurement schema")
    target = target_registry[schema]
    canonical = target["canonical_body"]
    if (canonical["layout_id"] != LAYOUT or tuple(canonical["slots"][:14]) != ARM_SLOTS
            or canonical["slots"][14:] != [n.removeprefix("state.dof_pos.") for n in EXTRA_NAMES]
            or (schema == SCHEMA and set(canonical["raw_indices"]) != set(range(6, 35)))
            or (schema == REAL_SCHEMA and (target["width"] != 41
                or canonical["raw_indices"] != list(range(15, 29)) + list(range(15))))):
        raise ValueError("unexpected whole-body target semantics")
    MeasurementSchema.from_mapping(schema, target)
    if source["metadata"]["architecture"]["measurement_dim"] != 29:
        raise ValueError("whole-body transition requires the original 29-slot architecture")
    return {
        "format": "zeta0.predictive_world.body_target_transition.v1",
        "source_weights_sha256": source["weights_sha256"],
        "source_config_sha256": canonical_sha256(source_payload),
        "target_measurement_registry_sha256": canonical_sha256(target_registry),
        "target_schema": schema, "target_layout": LAYOUT,
        "retained_slots": list(range(14)), "reset_slots": list(range(14, 29)),
        "initialization": "preserve_arm14_zero_new_value_and_validity_columns_and_readout_rows",
        "optimizer_restored": False,
    }


def verify_body_target_transition(record, source, source_payload, target_registry, source_registry):
    if any(target_registry.get(k) != v for k, v in source_registry.items()):
        raise ValueError("body-target transition cannot change existing measurement schemas")
    if record != make_body_target_transition(source, source_payload, target_registry,
                                            schema=record.get("target_schema")):
        raise ValueError("body-target transition is not bound to the exact source and target registry")
    return dict(record)


def migrate_body_target_tensors(tensors):
    """Copy all tensors except explicitly reset semantic coordinates."""
    result = dict(tensors)
    for encoder in ("online_encoder", "target_encoder"):
        key = f"encoder.{encoder}.measurement_projection.0.weight"
        value = tensors[key]
        if value.ndim != 2 or value.shape[1] != 58:
            raise ValueError(f"unexpected body input tensor: {key}")
        value = value.clone()
        value[:, 14:29] = 0
        value[:, 43:58] = 0
        result[key] = value
    for suffix in ("weight", "bias"):
        key = f"readout.body_output.1.{suffix}"
        value = tensors[key]
        if value.shape[0] != 29:
            raise ValueError(f"unexpected body output tensor: {key}")
        value = value.clone()
        value[14:] = 0
        result[key] = value
    return result
