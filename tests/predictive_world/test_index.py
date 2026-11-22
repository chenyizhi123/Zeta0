import pytest
from types import SimpleNamespace

from psi.data.predictive_world.index import (
    assert_predictive_world_index_temporal_abi,
    build_episode_windows,
    predictive_world_temporal_abi,
    publish_predictive_world_index,
    verify_predictive_world_index,
)
from psi.data.predictive_world.measurement_schema import MeasurementSchema, SourceCapability


def _schema():
    return MeasurementSchema.from_mapping("measurement", {
        "width": 2,
        "world_action_eligible": True,
        "fields": [{
            "name": "joint", "start": 0, "stop": 2, "kind": "measurement",
            "unit": "rad", "time_key": "state_time", "statistics_id": "stats-v1",
        }],
    })


def _capability(**overrides):
    value = {
        "measurement_schema_id": "measurement",
        "action_protocol_id": "native",
        "visual_keys": ["head"],
        "supports_future_visual": True,
        "supports_future_measurement": True,
        "supports_world_action": True,
        "audit_revision": "audit-sha",
    }
    value.update(overrides)
    return SourceCapability.from_mapping("source", value)


def _provenance():
    return {"audit": "sha", "registry_binding": {
        "measurement_registry_sha256": "0" * 64,
        "capability_registry_sha256": "1" * 64,
        "action_protocol_registry_sha256": "2" * 64,
        "state_schema_registry_sha256": "3" * 64,
    }}


def _temporal(**overrides):
    values = {
        "history_visual_offsets_s": (-0.4, -0.2, 0.0),
        "state_history_count": 3,
        "state_query_hz": 50.0,
        "future_group_ends_s": (0.2, 0.4, 0.6),
        "action_horizon": 3,
        "action_hz": 50.0,
        "visual_max_age_s": 0.021,
        "state_max_age_s": 0.021,
        "action_max_age_s": 0.021,
    }
    values.update(overrides)
    return predictive_world_temporal_abi(**values)


def _episode():
    timestamps = [index * 0.02 for index in range(51)]
    return {
        "source_id": "source",
        "episode_id": "episode",
        "split": "train",
        "anchor_times": [0.4],
        "visual_timestamps": timestamps,
        "state_timestamps": timestamps,
        "action_timestamps": timestamps,
        "materialization": {
            "measurement_schema_id": "measurement",
            "action_protocol_id": "native",
            "measurement_indices": [0, 1],
            "video_paths": {"head": "/dataset/head.mp4"},
        },
    }


def test_index_uses_causal_floor_and_never_crosses_episode(tmp_path):
    rows = build_episode_windows(
        _episode(),
        history_visual_offsets_s=(-0.4, -0.2, 0.0),
        state_offsets_s=(-0.04, -0.02, 0.0),
        future_group_ends_s=(0.2, 0.4, 0.6),
        action_offsets_s=(0.02, 0.04, 0.06),
        future_measurement_offsets_s=tuple(index / 50 for index in range(1, 31)),
        visual_max_age_s=0.021,
        state_max_age_s=0.021,
        action_max_age_s=0.021,
        capability=_capability(),
        measurement_schema=_schema(),
    )
    assert len(rows) == 1
    assert rows[0]["future_visual_indices"] == [30, 40, 50]
    assert rows[0]["future_visual_actual_times"] == pytest.approx([0.2, 0.4, 0.6])
    assert rows[0]["action_indices"] == [21, 22, 23]
    assert rows[0]["action_actual_times"] == pytest.approx([0.02, 0.04, 0.06])
    path, _ = publish_predictive_world_index(
        tmp_path / "index.jsonl", rows=rows, provenance=_provenance(),
        temporal_abi=_temporal(),
    )
    assert verify_predictive_world_index(path)["rows"] == 1
    data = SimpleNamespace(
        history_visual_offsets_s=(-0.4, -0.2, 0.0),
        state_history_count=3,
        state_query_hz=50.0,
        future_group_ends_s=(0.2, 0.4, 0.6),
        action_horizon=3,
        action_hz=50.0,
        visual_max_age_s=0.021,
        state_max_age_s=0.021,
        action_max_age_s=0.021,
    )
    assert_predictive_world_index_temporal_abi(path, data)
    data.visual_max_age_s = 0.04
    with pytest.raises(ValueError, match="selection.history_visual.max_age_s"):
        assert_predictive_world_index_temporal_abi(path, data)


def test_future_action_grid_does_not_invalidate_asynchronous_camera_anchor():
    temporal = _temporal(action_horizon=30)
    assert temporal["action_offsets_s"][0] == pytest.approx(0.02)
    assert temporal["action_offsets_s"][-1] == pytest.approx(0.6)
    episode = _episode()
    episode["anchor_times"] = [0.405]
    rows = build_episode_windows(
        episode,
        history_visual_offsets_s=(0.0,),
        state_offsets_s=(0.0,),
        future_group_ends_s=(0.2, 0.4, 0.6),
        action_offsets_s=tuple(index / 50 for index in range(1, 31)),
        future_measurement_offsets_s=tuple(index / 50 for index in range(1, 31)),
        visual_max_age_s=0.021,
        state_max_age_s=0.021,
        action_max_age_s=0.021,
        capability=_capability(),
        measurement_schema=_schema(),
    )
    assert rows[0]["action_actual_times"][0] > 0
    assert rows[0]["world_action_group_valid"] == [True, True, True]


def test_index_refuses_downstream_episode_overlap(tmp_path):
    rows = build_episode_windows(
        _episode(),
        history_visual_offsets_s=(0.0,),
        state_offsets_s=(0.0,),
        future_group_ends_s=(0.2,),
        action_offsets_s=(0.02,),
        future_measurement_offsets_s=tuple(index / 50 for index in range(1, 11)),
        visual_max_age_s=0.021,
        state_max_age_s=0.021,
        action_max_age_s=0.021,
        capability=_capability(),
        measurement_schema=_schema(),
    )
    with pytest.raises(ValueError, match="overlaps"):
        publish_predictive_world_index(
            tmp_path / "index.jsonl", rows=rows, provenance=_provenance(),
            temporal_abi=_temporal(
                history_visual_offsets_s=(0.0,), state_history_count=1,
                future_group_ends_s=(0.2,), action_horizon=1,
            ),
            excluded_episode_ids={"episode"},
        )
