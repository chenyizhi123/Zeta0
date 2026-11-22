"""Preserve real collection snapshots and naturally delayed feedback in training."""
import copy
import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_real_sonic_post_training import recording, registries
from psi.config.predictive_world import PredictiveWorldDataConfig
from psi.data.predictive_world.action_labels import materialize_action_labels
from psi.data.predictive_world.cache import canonical_sha256
from psi.data.predictive_world.index import temporal_abi_from_data_config
from psi.data.predictive_world.measurement_schema import MeasurementSchema, SourceCapability
from psi.data.predictive_world.real_sonic_post_training import MEASUREMENT, episode_windows, extend_registries, inspect_dataset
from psi.data.predictive_world.real_sonic_selection import RealSonicSampleSelector, selection_contract
from psi.sonic_actions import controller_command_values


CONTRACT = dict(version=1, camera_max_skew_ms=25., camera_wait_ms=20., max_age_ms=200.,
                body_max_offset_ms=40., hand_max_offset_ms=50., history_size=128, image_shape=[480, 640, 3])
CAMERAS = tuple("observation.images." + key for key in ("ego_view", "left_wrist", "right_wrist"))


def collector_recording(root):
    recording(root)
    for episode in range(2):
        timing_path = root / f"meta/chenyz_timing_{episode:06d}.jsonl"
        timing = [json.loads(line) for line in timing_path.read_text().splitlines()]
        for index, row in enumerate(timing):
            decision = row["decision_monotonic"]
            for camera, age in zip(CAMERAS, (.09, .083, .104)):
                row["cameras"][camera.removeprefix("observation.images.")]["received_monotonic"] = decision - age
            if index == 10:
                row["cameras"] = copy.deepcopy(timing[9]["cameras"])
            row["anchor_monotonic"] = row["cameras"]["ego_view"]["received_monotonic"]
            row["body"]["received_monotonic"] = decision - .088
            row["hand_state"].update(received_monotonic=decision - .02,
                                      channel_received_monotonic=[decision - .095, decision - .08])
            row["action"].update(hand_received_monotonic=decision - .005,
                                  hand_receive_monotonic=[decision - .005, decision - .005])
            row["contract"] = CONTRACT
        timing_path.write_text("".join(json.dumps(row) + "\n" for row in timing))
        quality_path = root / f"meta/chenyz_episode_{episode:06d}.json"
        quality = json.loads(quality_path.read_text())
        quality["observation_contract"] = CONTRACT
        quality_path.write_text(json.dumps(quality))
        path = root / f"data/file-{episode:03d}.parquet"
        values = pq.read_table(path).to_pylist()
        for row in values:
            row["observation.hand_q"] = [.2] * 12
            row["action.motion_token"] = [.25] * 64
            row["action.hand_q"] = [.3] * 12
        pq.write_table(pa.Table.from_pylist(values), path)


def data_config():
    return PredictiveWorldDataConfig(
        state_history_count=1, observed_visual_mode="current_multiview",
        future_component_alignment="independent_components_v1", real_observation_mode="collector_snapshot_v1",
        real_observation_contract=CONTRACT, real_action_label_mode="held_command_v1",
        state_max_age_s=.2, visual_max_age_s=.2, future_measurement_max_age_s=.04, future_visual_max_age_s=.08,
        measurement_registry="fixture", canonical_future_capabilities="fixture", action_protocol_registry="fixture",
        state_schema_registry="fixture", index_path="fixture", feature_cache="fixture",
    )


def prepare(root, *, recorded_projection=False):
    info, instruction, episodes, audit, statistics = inspect_dataset(root, task="doll",
        future_component_alignment="independent_components_v1", real_observation_mode="collector_snapshot_v1",
        real_action_label_mode="held_command_v1", recorded_action_projection=recorded_projection)
    extended = extend_registries(registries(), info, statistics, task="doll", audit_identity="fixture")
    schema = MeasurementSchema.from_mapping(MEASUREMENT, extended["measurement_registry"][MEASUREMENT])
    capability = SourceCapability.from_mapping("doll", extended["capability_registry"]["doll"])
    data = replace(data_config(), real_action_projection=audit.get("real_action_projection"))
    temporal = temporal_abi_from_data_config(data)
    rows = [row for episode in episodes for row in episode_windows(episode, task="doll", instruction=instruction,
        temporal=temporal, capability=capability, measurement_schema=schema)]
    return rows, audit, temporal


def test_recorded_projection_preserves_full_source_label_range_and_all_snapshots(tmp_path):
    collector_recording(tmp_path)
    path = tmp_path / "data/file-000.parquet"
    frames = pq.read_table(path).to_pylist()
    frames[6]["action.motion_token"][0] = -.75
    frames[7]["action.motion_token"][0] = .6875
    pq.write_table(pa.Table.from_pylist(frames), path)
    rows, audit, temporal = prepare(tmp_path, recorded_projection=True)
    projection = audit["real_action_projection"]
    assert projection["motion_token_min"] == -.75 and projection["motion_token_max"] == .6875
    assert projection["source_sha256"] == canonical_sha256(audit["real_action_projection_source"])
    data = replace(data_config(), real_action_projection=projection)
    data.validate()
    assert data.observation_contract["real_action_projection"] == projection
    assert temporal["real_action_projection"] == projection
    selector = RealSonicSampleSelector(tmp_path, selection_contract(data, audit))
    for row in rows:
        assert selector.reasons(row) == []
        assert row["materialization"]["real_action_projection"] == projection
        table = pq.read_table(row["materialization"]["data_path"])
        raw = np.concatenate([np.asarray(table[name].to_pylist(), dtype=np.float32)
                              for name in ("action.motion_token", "action.hand_q")], axis=1)
        labels, valid = materialize_action_labels(row, lambda indices: raw[indices])
        projected = controller_command_values(labels, projection=projection)
        np.testing.assert_array_equal(projected[valid], labels[valid])
    assert "real_action_projection" not in data_config().observation_contract
    assert "real_action_projection" not in temporal_abi_from_data_config(data_config())


def test_preserves_same_recorded_snapshot_and_actual_hand_channel_clock(tmp_path):
    collector_recording(tmp_path)
    rows, audit, temporal = prepare(tmp_path)
    selector = RealSonicSampleSelector(tmp_path, selection_contract(data_config(), audit))
    for row in rows:
        index = row["collector_snapshot_row_index"]
        assert row["state_indices"] == [index] and row["state_valid"] == [True]
        assert row["state_actual_times"][0] == pytest.approx(-.08, abs=1e-12)
        assert all(row["history_visual_indices_by_camera"][camera][-1] == index for camera in CAMERAS)
        assert all(row["history_visual_valid_by_camera"][camera][-1] for camera in CAMERAS)
        assert selector.reasons(row) == []
    repeated = rows[10]
    assert repeated["history_visual_indices_by_camera"][CAMERAS[0]][-1] == 10
    assert repeated["history_visual_actual_times_by_camera"][CAMERAS[0]][-1] == pytest.approx(-.09 - 1 / 30)
    assert temporal["selection"]["state_history"]["max_age_s"] == .2
    assert temporal["selection"]["future_measurement"]["max_age_s"] == .04
    assert temporal["selection"]["future_visual"]["max_age_s"] == .08
    assert temporal["selection"]["future_visual"]["strictly_after_anchor"]
    assert temporal["selection"]["future_measurement"]["strictly_after_anchor"]
    assert not temporal["selection"]["demonstrated_action"]["strictly_after_anchor"]


def test_held_commands_allow_update_before_anchor_without_future_supervision_leak(tmp_path):
    collector_recording(tmp_path)
    rows, _, temporal = prepare(tmp_path)
    row = rows[12]
    assert row["action_valid"][0]
    assert row["action_actual_times_by_component"]["left_hand"][0] < 0
    raw = np.concatenate([np.asarray(pq.read_table(row["materialization"]["data_path"])[name].to_pylist(), dtype=np.float32)
                          for name in ("action.motion_token", "action.hand_q")], axis=1)
    values, valid = materialize_action_labels(row, lambda indices: raw[indices])
    assert valid[0].all()
    np.testing.assert_array_equal(values[0, 64:], np.full(12, .3, dtype=np.float32))
    for candidate in rows:
        for field, offsets, age in (
            ("future_visual", temporal["future_group_ends_s"], .08),
            ("future_measurement", temporal["future_measurement_offsets_s"], .04),
        ):
            mask = np.asarray(candidate[field + "_valid"], dtype=bool)
            actual = np.asarray(candidate[field + "_actual_times"])
            assert (actual[mask] > 0).all()
            assert (np.asarray(offsets)[mask] - actual[mask] <= age + 1e-12).all()


@pytest.mark.parametrize("field,value", [("state_max_age_s", .3), ("visual_max_age_s", .08),
                                         ("future_measurement_max_age_s", .2), ("future_visual_max_age_s", .2)])
def test_config_rejects_unbound_input_or_relaxed_teacher_age(field, value):
    with pytest.raises(ValueError):
        replace(data_config(), **{field: value}).validate()


def test_collector_selection_checks_pairing_contract_without_40ms_decision_age(tmp_path):
    collector_recording(tmp_path)
    rows, audit, _ = prepare(tmp_path)
    selector = RealSonicSampleSelector(tmp_path, selection_contract(data_config(), audit))
    assert selector.reasons(rows[12]) == []
    path = tmp_path / "meta/chenyz_timing_000000.jsonl"
    timing = [json.loads(line) for line in path.read_text().splitlines()]
    timing[12]["hand_state"]["channel_received_monotonic"][0] = timing[12]["anchor_monotonic"] - .06
    path.write_text("".join(json.dumps(row) + "\n" for row in timing))
    selector = RealSonicSampleSelector(tmp_path, selector.contract)
    assert [reason["code"] for reason in selector.reasons(rows[12])] == ["snapshot_hand_offset"]
    bad_row = copy.deepcopy(rows[12])
    bad_row["history_visual_indices_by_camera"][CAMERAS[1]][-1] -= 1
    with pytest.raises(ValueError, match="same recorded row"):
        selector.reasons(bad_row)
