"""Native SONIC LeRobot state41/action76 recordings on their nominal frame grid.

These exports have no per-component receive clocks. The row grid addresses
recorded observations and labels, not synchronized exposure or hardware time.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.index import build_episode_windows
from psi.data.predictive_world import real_sonic_post_training as native

CAMERAS = tuple("observation.images." + key for key in ("ego_view", "left_wrist", "right_wrist"))
HAND_NAMES = [f"{side}.{joint}.q" for side in ("left", "right")
              for joint in ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")]
STATE_NAMES = [f"body.q_{i:02d}" for i in range(29)] + HAND_NAMES
ACTION_NAMES = [f"sonic.motion_token_{i:02d}" for i in range(64)] + HAND_NAMES


def inspect_dataset(root, *, task, seed=292285, validation_fraction=0.0):
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    features = info["features"]
    fps, count = info.get("fps"), info.get("total_episodes")
    if (info.get("codebase_version") != "v3.0" or info.get("robot_type") != "sonic_g1"
            or fps != 30 or type(count) is not int or count <= 0
            or features.get("observation.state", {}).get("names") != STATE_NAMES
            or features.get("observation.state", {}).get("shape") != [41]
            or features.get("action", {}).get("names") != ACTION_NAMES
            or features.get("action", {}).get("shape") != [76]
            or any(features.get(k, {}).get("dtype") != "video" for k in CAMERAS)):
        raise ValueError("requires audited sonic_g1 LeRobot v3 at 30 Hz, native state41/action76")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    if len(tasks) != 1 or not str(tasks[0]["task"]).strip():
        raise ValueError("requires one unambiguous task instruction")
    if not 0 <= validation_fraction < 1 or validation_fraction and count < 2:
        raise ValueError("invalid validation split")
    sessions = {}
    for path in sorted((root / "meta/sonic_sessions").glob("*.json")):
        session = json.loads(path.read_text())
        if (session.get("schema") != "sonic_g1_lerobot_v1" or session.get("state_names") != STATE_NAMES
                or session.get("action_names") != ACTION_NAMES
                or session.get("state_layout") != "body29,left_hand6,right_hand6"
                or session.get("action_layout") != "actual_sonic_token64,actual_left_hand_command6,actual_right_hand_command6"
                or session.get("local_config", {}).get("dataset_frequency") != fps):
            raise ValueError("SONIC recorder identity/layout disagrees with the dataset")
        sessions[str(path)] = file_sha256(path)
    if not sessions:
        raise ValueError("native SONIC recording requires recorder provenance")
    metadata, metadata_hashes = {}, {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        metadata_hashes[str(path)] = file_sha256(path)
        for row in pq.read_table(path).to_pylist():
            ep = row["episode_index"]
            if ep in metadata:
                raise ValueError("duplicate episode metadata")
            metadata[ep] = row
    if set(metadata) != set(range(count)):
        raise ValueError("incomplete episode metadata")
    nval = max(1, min(count - 1, round(count * validation_fraction))) if validation_fraction else 0
    validation = set(sorted(metadata, key=lambda ep: canonical_sha256([seed, task, ep]))[:nval])
    cache, hashes, episodes, inventory, train_states, train_actions = {}, {}, [], [], [], []
    for ep, meta in sorted(metadata.items()):
        path = root / info["data_path"].format(chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
        if path not in cache:
            cache.clear()
            cache[path] = pq.read_table(path)
            hashes[path] = file_sha256(path)
        table = cache[path]
        ids = np.asarray(table["episode_index"].to_pylist())
        indices = np.flatnonzero(ids == ep)
        table = table.take(indices)
        n = table.num_rows
        times = np.arange(n, dtype=np.float64) / fps
        stored = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        tolerance = np.maximum(np.spacing(times.astype(np.float32)), 1e-12)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if (n != meta["length"] or n < 2 or states.shape != (n, 41) or actions.shape != (n, 76)
                or not np.isfinite(states).all() or not np.isfinite(actions).all()
                or not np.isfinite(stored).all()
                or not np.array_equal(table["frame_index"].to_pylist(), np.arange(n))
                or np.any(np.abs(stored - times) > tolerance)
                or not np.all(np.asarray(table["task_index"].to_pylist()) == tasks[0]["task_index"])
                or not np.array_equal(table["index"].to_pylist(), np.arange(meta["dataset_from_index"], meta["dataset_to_index"]))):
            raise ValueError(f"non-finite/misaligned SONIC episode {ep}")
        if (np.any(np.abs(actions[:, :64] / native.MOTION_TOKEN_STEP - np.rint(actions[:, :64] / native.MOTION_TOKEN_STEP)) > 1e-5)
                or np.any(actions[:, 64:] < 0) or np.any(actions[:, 64:] > 1)):
            raise ValueError("native motion tokens/hand labels violate the recorded protocol")
        videos, offsets, video_inventory = {}, {}, {}
        for camera in CAMERAS:
            key = "videos/" + camera
            video = root / info["video_path"].format(video_key=camera,
                chunk_index=meta[key + "/chunk_index"], file_index=meta[key + "/file_index"])
            start, end = meta[key + "/from_timestamp"] * fps, meta[key + "/to_timestamp"] * fps
            if not video.is_file() or not np.isclose(start, round(start), atol=1e-3, rtol=0) or not np.isclose(end-start, n, atol=.05, rtol=0):
                raise ValueError(f"video addresses disagree with episode {ep}: {video}")
            videos[camera], offsets[camera] = str(video), round(start)
            stat = video.stat()
            video_inventory[camera] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        split = "validation" if ep in validation else "train"
        outcome_path = root / f"meta/sonic_outcomes/episode_{ep:06d}.json"
        outcome = json.loads(outcome_path.read_text())
        if outcome.get("episode") != ep or outcome.get("frames") != n:
            raise ValueError("outcome metadata disagrees with saved episode")
        episodes.append(dict(episode=ep, split=split, times=times, rows=indices,
            parquet=str(path), video_paths=videos, video_offsets=offsets))
        inventory.append(dict(episode=ep, split=split, frames=n, parquet=str(path), parquet_sha256=hashes[path],
            video_paths=videos, video_inventory=video_inventory, task_success=outcome.get("task_success"),
            outcome_sha256=file_sha256(outcome_path)))
        if split == "train":
            train_states.append(states)
            train_actions.append(actions)
    if sum(ep["frames"] for ep in inventory) != info["total_frames"]:
        raise ValueError("dataset total frame count disagrees")
    audit = dict(format="zeta0.sonic_lerobot_post_training.release.v1", task=task, root=str(root),
        info_sha256=file_sha256(root / "meta/info.json"), tasks_sha256=file_sha256(root / "meta/tasks.parquet"),
        metadata_sha256=metadata_hashes, session_sha256=sessions,
        time_alignment=dict(version="sonic_lerobot_nominal_rows_v1", fps=fps,
            clock="frame_index/fps", physical_receive_clocks_available=False,
            hardware_exposure_synchronized=False, video_offsets="per_camera_episode_metadata",
            action_labels="next_recorded_rows_on_native_grid", timestamp_validation="one_float32_ulp"),
        state_columns=["observation.state"], action_columns=["action"], statistics_split="train",
        split_seed=seed, validation_fraction=validation_fraction, train_episodes=sorted(set(metadata)-validation),
        validation_episodes=sorted(validation), episodes=inventory)
    return info, str(tasks[0]["task"]), episodes, audit, native.fit_statistics(np.concatenate(train_states), np.concatenate(train_actions))


def extend_registries(source, info, statistics, *, task, audit_identity):
    resolved = copy.deepcopy(info)
    # G1 native q29: legs 0..11, waist 12..14, left/right arms 15..28.
    arms = [f"{side}_{joint}_joint" for side in ("left", "right")
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    resolved["features"]["observation.state"]["names"][15:29] = arms
    result = native.extend_registries(source, resolved, statistics, task=task, audit_identity=audit_identity)
    for field in result["measurement_registry"][native.MEASUREMENT]["fields"]:
        field["time_key"] = "recorded_frame_index_over_fps"
    capability = result["capability_registry"][task]
    capability.update(visual_timestamp_semantics="nominal_frame_grid",
                      audit_revision="sonic_lerobot_storage_rows_" + audit_identity)
    return result


def episode_windows(episode, *, task, instruction, temporal, capability, measurement_schema):
    primary = native.CAMERA
    times, rows = episode["times"], episode["rows"]
    offsets = episode["video_offsets"]
    visual_rows = np.arange(len(times)) + offsets[primary]
    record = dict(source_id=task, episode_id=f"{task}/episode_{episode['episode']:06d}",
        split=episode["split"], anchor_times=times, visual_timestamps=times, visual_row_indices=visual_rows,
        visual_streams={camera: dict(timestamps=times, row_indices=np.arange(len(times))+offsets[camera])
                        for camera in CAMERAS},
        state_timestamps=times, state_row_indices=rows, action_timestamps=times, action_row_indices=rows,
        materialization=dict(data_path=episode["parquet"], video_paths=episode["video_paths"], primary_camera_id=primary,
            instruction=instruction, state_columns=["observation.state"], action_columns=["action"],
            measurement_schema_id=capability.measurement_schema_id, state_schema_id=native.STATE, action_protocol_id=native.PROTOCOL,
            action_normalization_space=native.SPACE, measurement_indices=list(measurement_schema.canonical_body.raw_indices)))
    return build_episode_windows(record, history_visual_offsets_s=temporal["history_visual_offsets_s"],
        state_offsets_s=temporal["state_offsets_s"], future_group_ends_s=temporal["future_group_ends_s"],
        action_offsets_s=temporal["action_offsets_s"], future_measurement_offsets_s=temporal["future_measurement_offsets_s"],
        # 150 Hz is the exact common tick grid of recorded 30 Hz rows and
        # unchanged 50 Hz body queries; selection still floors to real rows.
        frame_grid_hz=150, visual_max_age_s=temporal["selection"]["history_visual"]["max_age_s"],
        state_max_age_s=temporal["selection"]["state_history"]["max_age_s"],
        action_max_age_s=temporal["selection"]["demonstrated_action"]["max_age_s"],
        future_measurement_max_age_s=temporal["selection"]["future_measurement"]["max_age_s"],
        future_visual_max_age_s=temporal["selection"]["future_visual"]["max_age_s"],
        capability=capability, measurement_schema=measurement_schema)
