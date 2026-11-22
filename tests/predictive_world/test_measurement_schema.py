import json

import pytest

from psi.data.predictive_world.measurement_schema import (
    MeasurementSchema,
    SourceCapability,
    canonical_body_target_signature,
    load_capability_registry,
    validate_predictive_world_row_capability,
)


def test_only_audited_measurements_are_body_targets():
    schema = MeasurementSchema.from_mapping(
        "hiw_g1_q29_wbc23",
        {
            "width": 52,
            "world_action_eligible": False,
            "fields": [
                {
                    "name": "joint_observation",
                    "start": 0,
                    "stop": 29,
                    "kind": "measurement",
                    "unit": "rad",
                    "time_key": "timestamp.state",
                    "statistics_id": "hiw_joint_observation_v1",
                },
                {
                    "name": "pivot_unknown",
                    "start": 29,
                    "stop": 30,
                    "kind": "unknown",
                    "unit": None,
                    "time_key": "timestamp.state",
                },
                {
                    "name": "gripper_controls",
                    "start": 45,
                    "stop": 52,
                    "kind": "command",
                    "unit": None,
                    "time_key": "timestamp.action",
                },
            ],
        },
    )
    assert schema.body_target_indices == tuple(range(29))
    assert not schema.world_action_eligible


def test_overlapping_physical_fields_are_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        MeasurementSchema.from_mapping(
            "bad",
            {
                "width": 4,
                "fields": [
                    {"name": "a", "start": 0, "stop": 3, "kind": "unknown", "time_key": "t"},
                    {"name": "b", "start": 2, "stop": 4, "kind": "unknown", "time_key": "t"},
                ],
            },
        )


def test_source_capability_cannot_overclaim_world_actions(tmp_path):
    schema = MeasurementSchema.from_mapping(
        "measured",
        {
            "width": 2,
            "world_action_eligible": False,
            "fields": [{
                "name": "joint", "start": 0, "stop": 2, "kind": "measurement",
                "unit": "rad", "time_key": "state_time", "statistics_id": "stats-v1",
            }],
        },
    )
    path = tmp_path / "capabilities.json"
    path.write_text(json.dumps({"source": {
        "measurement_schema_id": "measured",
        "action_protocol_id": "native",
        "visual_keys": ["head"],
        "supports_future_visual": True,
        "supports_future_measurement": True,
        "supports_world_action": True,
        "audit_revision": "audit-sha",
    }}))
    with pytest.raises(ValueError, match="forbidden"):
        load_capability_registry(
            path, measurement_schemas={"measured": schema}, action_protocol_ids={"native"}
        )


def _measured_schema(schema_id="measured", statistics_id="stats-v1"):
    return MeasurementSchema.from_mapping(schema_id, {
        "width": 2,
        "world_action_eligible": True,
        "fields": [{
            "name": "joint", "start": 0, "stop": 2, "kind": "measurement",
            "unit": "rad", "time_key": "state_time", "statistics_id": statistics_id,
        }],
    })


def _source_capability(**overrides):
    payload = {
        "measurement_schema_id": "measured",
        "action_protocol_id": "native",
        "visual_keys": ["head"],
        "supports_future_visual": True,
        "supports_future_measurement": True,
        "supports_world_action": True,
        "audit_revision": "audit-sha",
    }
    payload.update(overrides)
    return SourceCapability.from_mapping("source", payload)


def _row(capability):
    return {
        "source_id": "source",
        "source_capability": capability.index_contract(),
        "future_visual_valid": [True],
        "future_measurement_valid": [True],
        "world_action_group_valid": [True],
        "materialization": {
            "measurement_schema_id": "measured",
            "action_protocol_id": "native",
            "video_paths": {"head": "/data/head.mp4"},
            "measurement_indices": [0, 1],
        },
    }


def test_row_capability_flags_and_measurement_indices_fail_closed():
    schema = _measured_schema()
    capability = _source_capability()
    row = _row(capability)
    validate_predictive_world_row_capability(
        row, capabilities={"source": capability}, measurement_schemas={"measured": schema}
    )
    row["materialization"]["measurement_indices"] = [1, 0]
    with pytest.raises(ValueError, match="exactly equal"):
        validate_predictive_world_row_capability(
            row, capabilities={"source": capability}, measurement_schemas={"measured": schema}
        )
    unsupported = _source_capability(supports_future_visual=False)
    row = _row(unsupported)
    with pytest.raises(ValueError, match="future visual"):
        validate_predictive_world_row_capability(
            row, capabilities={"source": unsupported}, measurement_schemas={"measured": schema}
        )


def test_multi_camera_shared_indices_require_explicit_sync_audit():
    capability = _source_capability(visual_keys=["head", "wrist"])
    with pytest.raises(ValueError, match="explicitly audited"):
        capability.index_contract()
    synchronized = _source_capability(
        visual_keys=["head", "wrist"],
        shared_frame_index_across_visual_keys=True,
        visual_synchronization_audit_revision="sync-audit-sha",
    )
    assert synchronized.index_contract()["visual_alignment_mode"] == "shared_frame_index"


def test_statistics_ids_are_part_of_cross_source_target_signature():
    first = _measured_schema("first", "stats-v1")
    second = _measured_schema("second", "stats-v2")
    capabilities = {
        "first": SourceCapability.from_mapping("first", {
            **_source_capability().__dict__, "measurement_schema_id": "first",
        }),
        "second": SourceCapability.from_mapping("second", {
            **_source_capability().__dict__, "measurement_schema_id": "second",
        }),
    }
    with pytest.raises(ValueError, match="normalization signature"):
        canonical_body_target_signature(
            capabilities, {"first": first, "second": second}
        )
