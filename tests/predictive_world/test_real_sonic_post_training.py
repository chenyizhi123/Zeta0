import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi.data.predictive_world.index import predictive_world_temporal_abi
from psi.data.predictive_world.measurement_schema import MeasurementSchema, SourceCapability
from psi.data.predictive_world.real_sonic_post_training import (
    ACTION_COLUMNS, CAMERA, MEASUREMENT, STATE_COLUMNS,
    episode_windows, extend_registries, inspect_dataset, unique_stream,
)


def recording(root, *, fps=30):
    cameras = [CAMERA, "observation.images.left_wrist", "observation.images.right_wrist"]
    arms = [f"{side}_{joint}_joint" for side in ("left", "right")
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    features = {k: dict(shape=[d]) for k, d in zip((*STATE_COLUMNS, *ACTION_COLUMNS), (29, 12, 64, 12))}
    features["observation.state"]["names"] = [f"lower{i}" for i in range(15)] + arms
    features.update({k: dict(dtype="video") for k in cameras})
    info = dict(codebase_version="v3.0", fps=fps, robot_type="g1_sonic_brainco", total_episodes=2,
                total_frames=180, features=features,
                data_path="data/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/file-{file_index:03d}.mp4")
    (root / "meta/episodes").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist([dict(task_index=0, task="Place the doll on the table.")]), root / "meta/tasks.parquet")
    for ep in range(2):
        n = 90
        physical = np.arange(n) / fps + np.where(np.arange(n) >= 45, .8, 0.)
        values, timing = [], []
        for i, time in enumerate(physical):
            decision = 1000. + time
            values.append(dict(episode_index=ep, frame_index=i, task_index=0, timestamp=float(np.float32(i / fps)),
                               **{"observation.sample_time": time, "action.wbc": [float("nan")] * 29,
                                  **{k: (np.arange(d) + ep * 100 + i).tolist()
                                     for k, d in zip((*STATE_COLUMNS, *ACTION_COLUMNS), (29, 12, 64, 12))}}))
            timing.append(dict(frame_index=i, decision_monotonic=decision, sample_segment=0,
                               body=dict(received_monotonic=decision - .018),
                               hand_state=dict(received_monotonic=decision - .02),
                               action=dict(body_received_monotonic=decision - .02, hand_received_monotonic=decision - .018),
                               cameras={k.removeprefix("observation.images."): dict(received_monotonic=decision - age)
                                        for k, age in zip(cameras, (.01, .015, .025))}))
        pq.write_table(pa.Table.from_pylist(values), root / f"data/file-{ep:03d}.parquet")
        meta = {"episode_index": ep, "length": n, "data/chunk_index": 0, "data/file_index": ep}
        for k in cameras:
            meta.update({f"videos/{k}/chunk_index": 0, f"videos/{k}/file_index": ep,
                         f"videos/{k}/from_timestamp": 0., f"videos/{k}/to_timestamp": n / fps})
            video = root / f"videos/{k}/file-{ep:03d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"numeric-audit-only")
        pq.write_table(pa.Table.from_pylist([meta]), root / f"meta/episodes/file-{ep:03d}.parquet")
        (root / f"meta/chenyz_timing_{ep:06d}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in timing))
        (root / f"meta/chenyz_episode_{ep:06d}.json").write_text(json.dumps(dict(
            episode=ep, frames=n, storage_valid=True, timing_file=f"meta/chenyz_timing_{ep:06d}.jsonl",
            training_eligible=False, task_success=None, quality_exclusions=["task_success_unknown", "time_gap_present"])))
    return info


def test_real_50hz_storage_uses_native_frame_addresses_and_physical_times(tmp_path):
    recording(tmp_path, fps=50)
    _, _, episodes, audit, _ = inspect_dataset(tmp_path, task="doll")
    assert audit["time_alignment"]["storage_fps"] == 50
    assert len(episodes) == 4  # The physical recording gap still splits both episodes.
    assert all(item["frames"] == 90 for item in audit["episodes"])
    info_path = tmp_path / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["fps"] = 30
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="storage timestamps"):
        inspect_dataset(tmp_path, task="doll")


def test_explicit_episode_exclusion_also_excludes_normalization_statistics(tmp_path):
    recording(tmp_path, fps=50)
    _, _, episodes, audit, stats = inspect_dataset(tmp_path, task="doll", excluded_episodes=[1])
    assert len(episodes) == 2
    assert audit["excluded_episodes"] == [1] and audit["excluded_frames"] == 90
    assert audit["train_episodes"] == [0]
    assert [item["episode"] for item in audit["episodes"]] == [0]
    assert stats["body_mean"][0] == 44.5
    with pytest.raises(ValueError, match="no training data"):
        inspect_dataset(tmp_path, task="doll", excluded_episodes=[0, 1])


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), None])
def test_real_recording_rejects_nonfinite_or_missing_storage_timestamps(tmp_path, timestamp):
    recording(tmp_path)
    path = tmp_path / "data/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[12]["timestamp"] = timestamp
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="storage timestamps"):
        inspect_dataset(tmp_path, task="doll")


@pytest.mark.parametrize("version", [2, 3])
def test_native_cycle_commands_are_labels_after_causal_observations(tmp_path, version):
    recording(tmp_path, fps=50)
    contract = dict(version=2, selection="latest_available_v1", max_age_ms=200.,
                    history_size=128, image_shape=[480, 640, 3],
                    cameras=["ego_view", "left_wrist", "right_wrist"],
                    timestamp_kind="workstation_source_receive_time")
    if version == 3:
        contract.update(version=3, max_age_ms=None, age_policy="record_latest_available")
    for ep in range(2):
        quality_path = tmp_path / f"meta/chenyz_episode_{ep:06d}.json"
        quality = json.loads(quality_path.read_text())
        quality["observation_contract"] = contract
        quality_path.write_text(json.dumps(quality))
        path = tmp_path / quality["timing_file"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for i, row in enumerate(rows):
            decision = row["decision_monotonic"]
            row["contract"] = contract
            row["action"].update(body_received_monotonic=decision+.001,
                hand_received_monotonic=decision+.002, hand_receive_monotonic=[decision+.002]*2,
                body_sequence=i, body_source_session="session",
                timestamp_semantics="native_command_commit_and_returned_dds_write")
            row["control_cycle"] = dict(mode="native_control_cycle_v1", observed=decision,
                committed=decision+.001, completed_monotonic=decision+.003, sequence=i, session="session")
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    kwargs = dict(task="doll", future_component_alignment="independent_components_v1",
                  real_observation_mode="collector_snapshot_v1", real_action_label_mode="held_command_v1")
    info, instruction, episodes, audit, stats = inspect_dataset(tmp_path, **kwargs)
    assert audit["time_alignment"]["command_after_observation"] is True
    if version == 3:
        from psi.config.predictive_world import PredictiveWorldDataConfig
        from psi.data.predictive_world.index import temporal_abi_from_data_config, _validated_temporal_abi
        data = PredictiveWorldDataConfig(observed_visual_mode="current_multiview", state_history_count=1,
            real_observation_mode="collector_snapshot_v1", real_observation_contract=contract,
            real_snapshot_max_age_s=.2, real_action_label_mode="held_command_v1",
            state_max_age_s=.2, visual_max_age_s=.2, future_measurement_max_age_s=.04,
            future_visual_max_age_s=.08, future_component_alignment="independent_components_v1",
            measurement_registry="fixture/measurements", canonical_future_capabilities="fixture/capabilities",
            action_protocol_registry="fixture/actions", state_schema_registry="fixture/states",
            index_path="fixture/index", feature_cache="fixture/cache")
        data.validate()
        temporal = temporal_abi_from_data_config(data)
        assert _validated_temporal_abi(temporal) == temporal
        extended = extend_registries(registries(), info, stats, task="doll", audit_identity="test")
        schema = MeasurementSchema.from_mapping(MEASUREMENT, extended["measurement_registry"][MEASUREMENT])
        capability = SourceCapability.from_mapping("doll", extended["capability_registry"]["doll"])
        windows = episode_windows(episodes[0], task="doll", instruction=instruction,
                                  temporal=temporal, capability=capability, measurement_schema=schema)
        assert windows[5]["state_indices"] == [5]
        assert windows[5]["state_valid"] == [True]
        assert windows[5]["collector_snapshot_row_index"] == 5
        assert any(windows[5]["action_valid"])
        from psi.data.predictive_world.index import publish_predictive_world_index
        provenance = dict(registry_binding={key + "_sha256": "a"*64 for key in (
            "measurement_registry", "capability_registry", "action_protocol_registry", "state_schema_registry")})
        ordinary, _ = publish_predictive_world_index(tmp_path / "sorted/index.jsonl", rows=iter(windows),
                                                     provenance=provenance, temporal_abi=temporal)
        streamed, _ = publish_predictive_world_index(tmp_path / "streamed/index.jsonl", rows=iter(windows),
            provenance=provenance, temporal_abi=temporal, presorted=True)
        assert ordinary.read_bytes() == streamed.read_bytes()
        episodes[0]["snapshot_state_component_times"][5, 0] = episodes[0]["times"][5] - .201
        stale = episode_windows(episodes[0], task="doll", instruction=instruction,
                               temporal=temporal, capability=capability, measurement_schema=schema)
        assert stale[5]["state_valid"] == [False]
    path = tmp_path / "meta/chenyz_timing_000000.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["control_cycle"]["completed_monotonic"] = rows[0]["decision_monotonic"]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match="native control-cycle"):
        inspect_dataset(tmp_path, **kwargs)


def registries():
    return dict(action_protocol_registry={"old": dict(dim=40)}, state_schema_registry={"old": dict(dim=64)},
                measurement_registry={"old": dict(canonical_body=dict(
                    layout_id="humanoid_upper_body_and_height_29_v1", slots=[f"slot{i}" for i in range(29)],
                    raw_indices=[None] * 29, mean=[0.] * 29, std=[1.] * 29,
                    normalization_space="per_source_registry_zscore_v1"))})


def test_real_motion76_physical_time_segments_and_independent_cameras(tmp_path):
    recording(tmp_path)
    before = (tmp_path / "data/file-000.parquet").read_bytes()
    info, instruction, episodes, audit, stats = inspect_dataset(tmp_path, task="doll", validation_fraction=.5)
    assert len(episodes) == 4
    assert stats["action"]["dim"] == 76 and stats["state"]["dim"] == 41
    assert stats["body_mean"][0] == audit["train_episodes"][0] * 100 + 44.5
    assert set(audit["train_episodes"]).isdisjoint(audit["validation_episodes"])
    assert all(ep["recorded_task_success"] is None for ep in audit["episodes"])
    assert (tmp_path / "data/file-000.parquet").read_bytes() == before
    source = registries()
    original = copy.deepcopy(source)
    extended = extend_registries(source, info, stats, task="doll", audit_identity="test")
    assert source == original
    for k in original:
        assert extended[k]["old"] == original[k]["old"]
    schema = MeasurementSchema.from_mapping(MEASUREMENT, extended["measurement_registry"][MEASUREMENT])
    assert schema.canonical_body.raw_indices == tuple(range(15, 29)) + (None,) * 15
    capability = SourceCapability.from_mapping("doll", extended["capability_registry"]["doll"])
    assert capability.index_contract()["visual_alignment_mode"] == "independent_camera_physical_time"
    temporal = predictive_world_temporal_abi(history_visual_offsets_s=[-.4, -.2, 0], state_history_count=21,
        state_query_hz=50, future_group_ends_s=[.2, .4, .6], action_horizon=30, action_hz=50,
        visual_max_age_s=.08, state_max_age_s=.04, action_max_age_s=.04)
    eligible = dict(prior=0, world=0, policy=0)
    for ep in episodes:
        rows = episode_windows(ep, task="doll", instruction=instruction, temporal=temporal,
                               capability=capability, measurement_schema=schema)
        allowed = set(ep["row_indices"].tolist())
        for row in rows:
            assert row["materialization"]["action_columns"] == list(ACTION_COLUMNS)
            assert row["materialization"]["state_columns"] == list(STATE_COLUMNS)
            for indices_key, valid_key in [("state_indices", "state_valid"), ("action_indices", "action_valid"),
                                           ("future_visual_indices", "future_visual_valid"),
                                           ("future_measurement_indices", "future_measurement_valid")]:
                assert all(index in allowed for index, valid in zip(row[indices_key], row[valid_key]) if valid)
            for camera in capability.visual_keys:
                actual = row["history_visual_actual_times_by_camera"][camera]
                valid = row["history_visual_valid_by_camera"][camera]
                assert all(t <= 1e-12 for t, ok in zip(actual, valid) if ok)
            for k, v in row["eligibility"].items():
                eligible[k] += v
        if ep["segment"] == 0:
            assert not any(rows[-1]["future_visual_valid"])
    assert all(v > 0 for v in eligible.values())


def test_repeated_availability_keeps_first_storage_address():
    stream = unique_stream([1., 1., 2., 2., 3.], [4, 5, 6, 7, 8])
    np.testing.assert_array_equal(stream["row_indices"], [4, 6, 8])
    with pytest.raises(ValueError, match="nondecreasing"):
        unique_stream([2., 1.], [0, 1])


def test_rejects_future_received_input(tmp_path):
    recording(tmp_path)
    path = tmp_path / "meta/chenyz_timing_000000.jsonl"
    timing = [json.loads(line) for line in path.read_text().splitlines()]
    timing[10]["cameras"]["ego_view"]["received_monotonic"] = timing[10]["decision_monotonic"] + .001
    path.write_text("".join(json.dumps(t) + "\n" for t in timing))
    with pytest.raises(ValueError, match="not yet available"):
        inspect_dataset(tmp_path, task="doll")
