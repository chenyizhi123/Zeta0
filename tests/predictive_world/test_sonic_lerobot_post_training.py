import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi.data.predictive_world import sonic_lerobot_post_training as data
from psi.data.predictive_world.index import predictive_world_temporal_abi
from psi.data.predictive_world.measurement_schema import MeasurementSchema, SourceCapability
from psi.data.predictive_world.real_sonic_post_training import MEASUREMENT


def recording(root):
    for folder in ("meta/episodes", "meta/sonic_sessions", "meta/sonic_outcomes", "data"):
        (root / folder).mkdir(parents=True)
    features = {"observation.state": dict(shape=[41], names=data.STATE_NAMES),
                "action": dict(shape=[76], names=data.ACTION_NAMES)}
    features.update({k: dict(dtype="video") for k in data.CAMERAS})
    info = dict(codebase_version="v3.0", robot_type="sonic_g1", fps=30, total_episodes=2,
        total_frames=180, features=features, data_path="data/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/file-{file_index:03d}.mp4")
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist([dict(task_index=0, task="Clean the table")]), root / "meta/tasks.parquet")
    (root / "meta/sonic_sessions/session.json").write_text(json.dumps(dict(schema="sonic_g1_lerobot_v1",
        state_names=data.STATE_NAMES, action_names=data.ACTION_NAMES, state_layout="body29,left_hand6,right_hand6",
        action_layout="actual_sonic_token64,actual_left_hand_command6,actual_right_hand_command6",
        local_config=dict(dataset_frequency=30))))
    values, metadata = [], []
    for ep in range(2):
        row = dict(episode_index=ep, length=90, dataset_from_index=ep*90, dataset_to_index=(ep+1)*90,
                   **{"data/chunk_index": 0, "data/file_index": 0})
        for camera in data.CAMERAS:
            row.update({f"videos/{camera}/chunk_index": 0, f"videos/{camera}/file_index": 0,
                        f"videos/{camera}/from_timestamp": ep*3., f"videos/{camera}/to_timestamp": (ep+1)*3.})
            video = root / f"videos/{camera}/file-000.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"numeric boundary fixture")
        metadata.append(row)
        (root / f"meta/sonic_outcomes/episode_{ep:06d}.json").write_text(json.dumps(dict(episode=ep, frames=90, task_success=None)))
        for i in range(90):
            values.append(dict(episode_index=ep, frame_index=i, index=ep*90+i, task_index=0,
                timestamp=float(np.float32(i/30)), **{"observation.state": [float(ep*100+i)]*41,
                "action": [.0625*(i % 4)]*64+[.5]*12}))
    pq.write_table(pa.Table.from_pylist(values), root / "data/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/file-000.parquet")


def test_native_shards_and_offsets_make_true_50_step_targets_without_crossing_episodes(tmp_path):
    from test_real_sonic_post_training import registries

    recording(tmp_path)
    # Each camera can roll its video shards at a different episode boundary.
    p = tmp_path / "meta/episodes/file-000.parquet"
    metadata = pq.read_table(p).to_pylist()
    for field in ("from_timestamp", "to_timestamp"):
        metadata[1]["videos/observation.images.left_wrist/"+field] += 1.
    pq.write_table(pa.Table.from_pylist(metadata), p)
    info, instruction, episodes, audit, stats = data.inspect_dataset(tmp_path, task="clean")
    assert audit["time_alignment"]["physical_receive_clocks_available"] is False
    assert len(episodes) == 2 and stats["state"]["dim"] == 41 and stats["action"]["dim"] == 76
    registry = data.extend_registries(registries(), info, stats, task="clean", audit_identity="audit")
    schema = MeasurementSchema.from_mapping(MEASUREMENT, registry["measurement_registry"][MEASUREMENT])
    assert schema.canonical_body.raw_indices == tuple(range(15, 29)) + (None,)*15
    capability = SourceCapability.from_mapping("clean", registry["capability_registry"]["clean"])
    assert capability.index_contract()["visual_alignment_mode"] == "independent_camera_nominal_time"
    assert capability.index_contract()["visual_synchronization_audit_revision"] is None
    temporal = predictive_world_temporal_abi(history_visual_offsets_s=[-.4, -.2, 0], state_history_count=1,
        state_query_hz=50, future_group_ends_s=[.2, .4, .6], action_horizon=50, action_hz=30,
        visual_max_age_s=.08, state_max_age_s=.04, action_max_age_s=.04)
    rows = data.episode_windows(episodes[1], task="clean", instruction=instruction, temporal=temporal,
                               capability=capability, measurement_schema=schema)
    row = rows[15]
    assert row["state_indices"] == [105]
    assert row["action_indices"] == list(range(106, 156))
    assert all(row["action_valid"])
    assert row["history_visual_indices"][-1] == 105
    assert row["history_visual_indices_by_camera"]["observation.images.left_wrist"][-1] == 135
    assert row["history_visual_indices_by_camera"]["observation.images.right_wrist"][-1] == 105
    assert row["future_visual_indices"] == [111, 117, 123]
    assert row["materialization"]["state_columns"] == ["observation.state"]
    assert row["materialization"]["action_columns"] == ["action"]
    for row in rows:
        assert all(90 <= i < 180 for i, valid in zip(row["action_indices"], row["action_valid"]) if valid)


@pytest.mark.parametrize("damage", ["names", "clock", "layout", "video_offset", "nonfinite"])
def test_native_recorder_rejects_wrong_schemas_clocks_and_video_addresses(tmp_path, damage):
    recording(tmp_path)
    if damage == "names":
        path = tmp_path / "meta/info.json"
        info = json.loads(path.read_text())
        info["features"]["action"]["names"][0] = "reference_joint_angle"
        path.write_text(json.dumps(info))
    elif damage == "layout":
        path = tmp_path / "meta/sonic_sessions/session.json"
        info = json.loads(path.read_text()); info["action_layout"] = "unknown"
        path.write_text(json.dumps(info))
    elif damage == "video_offset":
        path = tmp_path / "meta/episodes/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["videos/observation.images.ego_view/from_timestamp"] = .01
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        path = tmp_path / "data/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        if damage == "clock": rows[12]["timestamp"] += .01
        else: rows[12]["observation.state"][0] = float("nan")
        pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError): data.inspect_dataset(tmp_path, task="clean")


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), None])
def test_native_recorder_rejects_nonfinite_or_missing_timestamps(tmp_path, timestamp):
    recording(tmp_path)
    path = tmp_path / "data/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[12]["timestamp"] = timestamp
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="non-finite/misaligned"):
        data.inspect_dataset(tmp_path, task="clean")


def test_real_whole_body_targets_keep_all_measured_joints_in_the_materialized_windows(tmp_path):
    from test_real_sonic_post_training import registries
    from psi.models.predictive_world.body_target_transition import REAL_SCHEMA, real_whole_body_schema
    from psi.data.predictive_world.dataset import _read_vectors

    recording(tmp_path)
    info, instruction, episodes, audit, statistics = data.inspect_dataset(tmp_path, task="clean")
    source = registries()
    from psi.models.predictive_world.body_target_transition import OLD_SLOTS
    source["measurement_registry"]["old"]["canonical_body"]["slots"] = list(OLD_SLOTS)
    result = data.extend_registries(source, info, statistics, task="clean", audit_identity="a")
    original = result["measurement_registry"][MEASUREMENT]
    target = real_whole_body_schema(original, data.STATE_NAMES, statistics["body_mean"], statistics["body_std"])
    capability = SourceCapability.from_mapping("clean", {**result["capability_registry"]["clean"], "measurement_schema_id": REAL_SCHEMA})
    schema = MeasurementSchema.from_mapping(REAL_SCHEMA, target)
    temporal = predictive_world_temporal_abi(history_visual_offsets_s=[-.4, -.2, 0], state_history_count=1,
        state_query_hz=50, future_group_ends_s=[.2, .4, .6], action_horizon=50, action_hz=30,
        visual_max_age_s=.08, state_max_age_s=.04, action_max_age_s=.04)
    rows = data.episode_windows(episodes[0], task="clean", instruction=instruction, temporal=temporal,
                               capability=capability, measurement_schema=schema)
    row = rows[20]
    assert row["materialization"]["measurement_schema_id"] == REAL_SCHEMA
    indices = row["materialization"]["measurement_indices"]
    assert indices == list(range(15, 29)) + list(range(15))
    assert len(set(indices)) == 29 and all(i < 29 for i in indices)
    table = pq.read_table(tmp_path / "data/file-000.parquet")
    raw = _read_vectors(table, [21, 22], ["observation.state"])
    assert raw[:, indices].shape == (2, 29)
    for field in ("raw_indices", "slots", "mean", "std"):
        assert target["canonical_body"][field][:14] == original["canonical_body"][field][:14]
    assert original["canonical_body"]["raw_indices"][14:] == [None] * 15
    with pytest.raises(ValueError):
        real_whole_body_schema(original, list(reversed(data.STATE_NAMES)), statistics["body_mean"], statistics["body_std"])


def test_native_launcher_dry_run_binds_replacement_model_and_horizon_without_dispatch(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    recording(tmp_path / "recording")
    for name in ("source.json", "map.json", "psi0/config.json", "joint/.complete"):
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("{}")
    env = {**os.environ, "DATASET_FORMAT": "sonic_lerobot", "ACTION_HORIZON": "50",
        "QWEN_MODEL": str(tmp_path / "psi0"), "DATASET": str(tmp_path / "recording"),
        "OUTPUT_DIR": str(tmp_path / "run"), "GPUS": "2", "BATCH_SIZE": "256", "GRAD_ACCUMULATION": "1",
        "INIT_JOINT": str(tmp_path / "joint"), "SOURCE_CONFIG": str(tmp_path / "source.json"),
        "ASSET_PATH_MAP": str(tmp_path / "map.json"), "PYTHON_BIN": sys.executable}
    env.update(PARQUET_CACHE_CAPACITY="8", VIDEO_CACHE_CAPACITY="32")
    result = subprocess.run(["bash", str(root / "scripts/train/predictive_world/finetune-real-sonic.sh"), "--dry-run"],
        env=env, cwd="/tmp", text=True, capture_output=True, check=True)
    assert "prepare_sonic_lerobot_task.py" in result.stdout and "--action-horizon 50" in result.stdout
    assert "--model " + str(tmp_path / "psi0") in result.stdout
    assert "--collector-snapshots" not in result.stdout and "--processor-migration" not in result.stdout
    assert result.stdout.count("--parquet-cache-capacity 8 --video-cache-capacity 32") == 2
    assert not (tmp_path / "run").exists()
    env["VIDEO_CACHE_CAPACITY"] = "0"
    failed = subprocess.run(["bash", str(root / "scripts/train/predictive_world/finetune-real-sonic.sh"), "--dry-run"],
        env=env, cwd="/tmp", text=True, capture_output=True)
    assert failed.returncode == 2 and "Command:" not in failed.stdout
    assert not (tmp_path / "run").exists()
