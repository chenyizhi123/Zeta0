"""Audited HumanoidArena semantic40 episodes for joint post-training v1."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.index import build_episode_windows


TASKS = ("HSI_sit_sofa", "HSI_open_door", "HSI_boxing", "HSI_vision_navi",
         "HOI_football", "HOI_pp_box", "HOI_double_desk")
PROTOCOL = "sonic_semantic40_v31"
STATE = "sonic_state64_v31"
MEASUREMENT = "sonic_measured_state64_v31"
SPACE = "sonic_semantic40_raw_bounds_q99_affine_v1"
TIME_ALIGNMENT = {
    "version": "sonic_frame_grid_v2",
    "clock": "episode_frame_index / fps",
    "fps": 50,
    "timestamp_validation": "one_float32_ulp",
    "query_rounding": "integer_frame_ticks_float64_roundoff_only",
    "tail_selection": "causal_floor_with_existing_max_age",
}


def canonical_frame_timestamps(timestamps, frame_indices):
    """Recover the audited SONIC 50 Hz clock, without rewriting source data.

    The LeRobot writer derives timestamps from frame_index/fps and stores them
    as float32. Accept that rounding, not arbitrary jitter or missing frames.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    frames = np.asarray(frame_indices)
    if (times.ndim != 1 or not len(times) or frames.shape != times.shape
            or not np.isfinite(times).all()
            or not np.array_equal(frames, np.arange(len(times)))):
        raise ValueError("50 Hz clock requires finite timestamps and contiguous episode frame indices")
    canonical = frames.astype(np.float64) / TIME_ALIGNMENT["fps"]
    tolerance = np.maximum(np.spacing(canonical.astype(np.float32)).astype(np.float64), 1e-12)
    if np.any(tolerance >= 0.25 / TIME_ALIGNMENT["fps"]) or np.any(np.abs(times-canonical) > tolerance):
        raise ValueError("timestamps disagree with the 50 Hz frame_index clock beyond float32 rounding")
    return canonical


def inspect_dataset(root, *, task, seed=292285, validation_fraction=0.0):
    """Read metadata and all numeric rows; never decode video or use a GPU."""
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    features = info["features"]
    camera_keys = sorted(key for key, value in features.items()
                         if key.startswith("observation.images.") and value.get("dtype") == "video")
    camera_keys = camera_keys or ["observation.images.front"]
    if len(camera_keys) > 1 and not info.get("visual_synchronization_audit_revision"):
        raise ValueError("multicamera data requires an explicit shared-frame synchronization audit")
    if "observation.images.front" not in camera_keys:
        raise ValueError("SONIC post-training requires the front camera as primary")
    if (info["fps"] != 50 or features["observation.state"]["shape"] != [64]
            or features["action"]["shape"] != [40]
            or info.get("vla_protocol", {}).get("schema") != "unitree_g1_gmt_refpose_v3_1"):
        raise ValueError("expected HumanoidArena v3.1 semantic40 at 50 Hz with measured state64")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    tasks_path = root / "meta/tasks.jsonl"
    if tasks_path.exists():
        tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line]
    else:
        tasks_path = root / "meta/tasks.parquet"
        tasks = pq.read_table(tasks_path).to_pylist()
    if len(tasks) != 1:
        raise ValueError("each seven-task job requires one unambiguous task instruction")
    count = int(info["total_episodes"])
    if count < 1 or (validation_fraction > 0 and count < 2):
        raise ValueError("at least one episode is required; held-out validation requires two")
    validation_count = (max(1, min(count-1, int(round(count * validation_fraction))))
                        if validation_fraction > 0 else 0)
    ordered = sorted(range(count), key=lambda episode: canonical_sha256([seed, task, episode]))
    validation = set(ordered[:validation_count])
    episodes, train_states, train_actions = [], [], []
    is_v3 = str(info["codebase_version"]).startswith("v3")
    if is_v3 and len(camera_keys) > 1:
        raise ValueError("multicamera SONIC preparation currently requires synchronized per-episode v2.1 videos")
    metadata = {}
    if is_v3:
        meta_columns = ["episode_index", "length", "data/chunk_index", "data/file_index",
                        "videos/observation.images.front/chunk_index", "videos/observation.images.front/file_index",
                        "videos/observation.images.front/from_timestamp", "videos/observation.images.front/to_timestamp"]
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            for row in pq.read_table(path, columns=meta_columns).to_pylist():
                if row["episode_index"] in metadata:
                    raise ValueError("duplicate v3 episode metadata")
                metadata[int(row["episode_index"])] = row
        if set(metadata) != set(range(count)):
            raise ValueError("v3 episode metadata is incomplete")
    table_cache, hashes = {}, {}
    for episode in range(count):
        if is_v3:
            ep_meta = metadata[episode]
            parquet = root / info["data_path"].format(chunk_index=ep_meta["data/chunk_index"], file_index=ep_meta["data/file_index"])
            video = root / info["video_path"].format(video_key="observation.images.front",
                chunk_index=ep_meta["videos/observation.images.front/chunk_index"],
                file_index=ep_meta["videos/observation.images.front/file_index"])
            video_start = float(ep_meta["videos/observation.images.front/from_timestamp"]) * 50
            if not np.isclose(video_start, round(video_start), atol=1e-3, rtol=0):
                raise ValueError("v3 video offset does not align with the 50 Hz frame grid")
            frame_offset = round(video_start)
        else:
            fields = dict(episode_chunk=episode // int(info["chunks_size"]), episode_index=episode,
                          video_key="observation.images.front")
            parquet = root / info["data_path"].format(**fields)
            video = root / info["video_path"].format(**fields)
            frame_offset = 0
        if not video.is_file():
            raise FileNotFoundError(video)
        if parquet not in table_cache:
            table_cache.clear()  # bounded shard cache; consecutive v3 episodes reuse it
            table_cache[parquet] = pq.read_table(parquet, columns=["episode_index", "timestamp", "frame_index", "observation.state", "action"])
        full_table = table_cache[parquet]
        episode_ids = np.asarray(full_table["episode_index"].to_pylist(), dtype=np.int64)
        row_indices = np.flatnonzero(episode_ids == episode)
        table = full_table.take(row_indices)
        times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
        frames = np.asarray(table["frame_index"].to_pylist()).reshape(-1)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if is_v3 and (len(times) != int(ep_meta["length"]) or not np.isclose(
                (float(ep_meta["videos/observation.images.front/to_timestamp"]) * 50) - frame_offset,
                len(times), atol=0.05, rtol=0)):
            raise ValueError("v3 episode length/video interval mismatch")
        if (states.shape != (len(times), 64) or actions.shape != (len(times), 40)
                or not all(np.isfinite(x).all() for x in (times, states, actions))
                or len(times) < 2 or not np.all(np.diff(times) > 0)
                or not np.array_equal(frames, np.arange(len(times)))):
            raise ValueError(f"non-finite/misaligned semantic40 episode: {parquet}")
        canonical_times = canonical_frame_timestamps(times, frames)
        timestamp_error = float(np.abs(times-canonical_times).max())
        split = "validation" if episode in validation else "train"
        if split == "train":
            train_states.append(states)
            train_actions.append(actions)
        if parquet not in hashes:
            hashes[parquet] = file_sha256(parquet)
        episodes.append({"episode": episode, "split": split, "times": canonical_times,
                         "frame_grid_hz": TIME_ALIGNMENT["fps"],
                         "timestamp_max_rounding_error_s": timestamp_error,
                         "row_indices": row_indices, "frame_offset": frame_offset,
                         "parquet": str(parquet), "parquet_sha256": hashes[parquet],
                         "video": str(video), "video_size": video.stat().st_size,
                         "video_mtime_ns": video.stat().st_mtime_ns})
        if len(camera_keys) > 1:
            video_paths = {}
            video_inventory = {}
            for camera in camera_keys:
                camera_path = root / info["video_path"].format(
                    episode_chunk=episode // int(info["chunks_size"]),
                    episode_index=episode, video_key=camera)
                if not camera_path.is_file():
                    raise FileNotFoundError(camera_path)
                video_paths[camera] = str(camera_path)
                stat = camera_path.stat()
                video_inventory[camera] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            episodes[-1].update(video_paths=video_paths, video_inventory=video_inventory)
    states, actions = np.concatenate(train_states), np.concatenate(train_actions)
    statistics = fit_statistics(states, actions)
    audit = {
        "format": "zeta0.sonic_post_training.release.v2", "task": task,
        "time_alignment": dict(TIME_ALIGNMENT),
        "root": str(root), "info_sha256": file_sha256(root / "meta/info.json"),
        "tasks_sha256": file_sha256(tasks_path),
        "split_seed": seed, "validation_fraction": validation_fraction,
        "train_episodes": sorted(set(range(count))-validation),
        "validation_episodes": sorted(validation), "statistics_split": "train",
        "episodes": [{k: v for k, v in ep.items() if k not in {"times", "row_indices"}} for ep in episodes],
    }
    return info, str(tasks[0]["task"]), episodes, audit, statistics


def fit_statistics(states, actions):
    """Fit native-frame training statistics, never average per-task quantiles."""
    states, actions = np.asarray(states), np.asarray(actions)
    if (states.ndim != 2 or states.shape[1] != 64 or actions.shape != (len(states), 40)
            or not len(states) or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError("SONIC statistics require finite aligned nonempty state64/action40")
    def bounds(values):
        low, high = np.quantile(values.astype(np.float64), [0.01, 0.99], axis=0)
        # Keep native values for degenerate channels; never divide by zero.
        # The joint adapters use an affine transform, not Psi0's clipping rule.
        constant = np.abs(high-low) <= 1e-6
        return np.where(constant, 0., (high+low)/2).tolist(), np.where(constant, 1., (high-low)/2).tolist()
    action_mean, action_std = bounds(actions)
    state_mean, state_std = bounds(states)
    return {
        "action": {"dim": 40, "mean": action_mean, "std": action_std, "normalization_space": SPACE},
        "state": {"dim": 64, "mean": state_mean, "std": state_std},
        "body_mean": states.mean(axis=0, dtype=np.float64).tolist(),
        "body_std": np.maximum(states.std(axis=0, dtype=np.float64), 1e-6).tolist(),
    }


def extend_registries(source, info, statistics, *, task, audit_identity):
    registries = copy.deepcopy(source)
    names = info["features"]["observation.state"]["names"]
    arm_joints = [f"state.dof_pos.{side}_{joint}_joint" for side in ("left", "right")
                  for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                "wrist_roll", "wrist_pitch", "wrist_yaw")]
    indices = [names.index(name) for name in arm_joints] + [None] * 15
    # Current measured state has no hand positions or base height. Never use
    # reference-pose commands in action[] as measured future body targets.
    canonical = copy.deepcopy(next(iter(source["measurement_registry"].values()))["canonical_body"])
    canonical.update(raw_indices=indices,
                     mean=[statistics["body_mean"][i] if i is not None else 0. for i in indices],
                     std=[statistics["body_std"][i] if i is not None else 1. for i in indices])
    registries["action_protocol_registry"][PROTOCOL] = statistics["action"]
    registries["state_schema_registry"][STATE] = statistics["state"]
    registries["measurement_registry"][MEASUREMENT] = {
        "width": 64, "world_action_eligible": True, "canonical_body": canonical,
        "fields": [{"name": "measured_joint_positions", "start": 6, "stop": 35,
                    "kind": "measurement", "unit": "rad", "coordinate_frame": "robot_native",
                    "time_key": "timestamp", "statistics_id": STATE}],
    }
    registries["capability_registry"] = {task: {
        "measurement_schema_id": MEASUREMENT, "action_protocol_id": PROTOCOL,
        "visual_keys": sorted(key for key, value in info["features"].items()
                              if key.startswith("observation.images.") and value.get("dtype") == "video")
                       or ["observation.images.front"], "supports_future_visual": True,
        "supports_future_measurement": True, "supports_world_action": True,
        "audit_revision": "sonic_v31_measured_q29_" + audit_identity,
    }}
    capability = registries["capability_registry"][task]
    if len(capability["visual_keys"]) > 1:
        revision = info.get("visual_synchronization_audit_revision")
        if not revision:
            raise ValueError("multicamera data requires an explicit shared-frame synchronization audit")
        # Wrist cameras are causal policy inputs. The packed frozen-teacher
        # target remains the front view (16 x 2048), as in single-view training.
        capability.update(future_visual_keys=["observation.images.front"],
                          shared_frame_index_across_visual_keys=True,
                          visual_synchronization_audit_revision=revision)
    return registries


def episode_windows(episode, *, task, instruction, temporal, capability, measurement_schema,
                    protocol=PROTOCOL, state=STATE, measurement=MEASUREMENT,
                    normalization_space=SPACE, state_columns=("observation.state",),
                    primary_camera="observation.images.front"):
    times = episode["times"]
    rows = episode["row_indices"]
    video_frames = np.arange(len(times)) + episode["frame_offset"]
    record = {
        "source_id": task, "episode_id": f"{task}/episode_{episode['episode']:06d}",
        "split": episode["split"], "anchor_times": times,
        "visual_timestamps": times, "visual_row_indices": video_frames,
        "state_timestamps": times, "state_row_indices": rows,
        "action_timestamps": times, "action_row_indices": rows,
        "materialization": {
            "data_path": episode["parquet"],
            "video_paths": episode.get("video_paths", {primary_camera: episode["video"]}),
            "primary_camera_id": primary_camera, "instruction": instruction,
            "state_columns": list(state_columns), "action_columns": ["action"],
            "measurement_schema_id": measurement, "state_schema_id": state,
            "action_protocol_id": protocol, "action_normalization_space": normalization_space,
            "measurement_indices": list(measurement_schema.canonical_body.raw_indices),
        },
    }
    return build_episode_windows(
        record, history_visual_offsets_s=temporal["history_visual_offsets_s"],
        state_offsets_s=temporal["state_offsets_s"], future_group_ends_s=temporal["future_group_ends_s"],
        action_offsets_s=temporal["action_offsets_s"],
        future_measurement_offsets_s=temporal["future_measurement_offsets_s"],
        visual_max_age_s=temporal["selection"]["history_visual"]["max_age_s"], state_max_age_s=temporal["selection"]["state_history"]["max_age_s"],
        action_max_age_s=temporal["selection"]["demonstrated_action"]["max_age_s"], capability=capability,
        measurement_schema=measurement_schema,
        frame_grid_hz=episode["frame_grid_hz"],
    )
