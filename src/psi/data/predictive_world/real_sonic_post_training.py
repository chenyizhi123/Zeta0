"""Recorded G1/BrainCo data -> native motion64 + hand12 joint adaptation.

The video frame grid is only a storage address. Alignment uses workstation
availability times from the timing sidecars and never compresses recording gaps.
The shared body target has audited arm joints; BrainCo's six hand coordinates
are retained as native state/action, without inventing a seven-joint hand map.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.index import build_episode_windows
from psi.data.predictive_world.alignment import causal_floor_indices
from psi.data.predictive_world.index import predictive_world_eligibility
from psi.sonic_actions import MOTION_TOKEN_LIMIT as MOTION_TOKEN_LIMIT, MOTION_TOKEN_STEP as MOTION_TOKEN_STEP

PROTOCOL = "real_sonic_motion64_brainco12_v1"
STATE = "real_g1_q29_brainco12_v1"
MEASUREMENT = "real_g1_measured_q29_brainco12_v1"
SPACE = "real_sonic_motion76_q01_q99_affine_v1"
CAMERA = "observation.images.ego_view"
STATE_COLUMNS = ("observation.state", "observation.hand_q")
ACTION_COLUMNS = ("action.motion_token", "action.hand_q")
ACTION_COMPONENT_SLICES = {"motion_token": [0, 64], "left_hand": [64, 70], "right_hand": [70, 76]}
TIME_ALIGNMENT = {
    "version": "real_sonic_workstation_availability_v1",
    "storage_fps": 30,
    "anchor_clock": "timing.decision_monotonic",
    "state_clock": "max(body.received_monotonic, hand_state.received_monotonic)",
    "visual_clock": "timing.cameras.<camera>.received_monotonic",
    "action_clock": "max(body_received_monotonic, hand_received_monotonic)",
    "selection": "causal_floor_with_source_checkpoint_max_age",
    "gaps": "separate_contiguous_recording_segments",
    "hardware_exposure_synchronized": False,
}


def unique_stream(times, rows):
    """Keep the first stored row for each newly available sample."""
    times, rows = np.asarray(times, dtype=np.float64), np.asarray(rows, dtype=np.int64)
    if (times.ndim != 1 or not len(times) or rows.shape != times.shape
            or not np.isfinite(times).all() or (np.diff(times) < 0).any()):
        raise ValueError("availability stream must be finite and nondecreasing")
    first = np.r_[True, np.diff(times) > 0]
    return dict(timestamps=times[first], row_indices=rows[first])


def fit_statistics(states, actions):
    def affine(values):
        low, high = np.quantile(values.astype(np.float64), [.01, .99], axis=0)
        constant = np.abs(high - low) <= 1e-6
        return dict(dim=values.shape[1], mean=np.where(constant, 0., (high + low) / 2).tolist(),
                    std=np.where(constant, 1., (high - low) / 2).tolist())
    return {"action": {**affine(actions), "normalization_space": SPACE},
            "state": affine(states), "body_mean": states.mean(0, dtype=np.float64).tolist(),
            "body_std": np.maximum(states.std(0, dtype=np.float64), 1e-6).tolist()}


def inspect_dataset(root, *, task, seed=292285, validation_fraction=0.0, future_component_alignment=None,
                    real_observation_mode=None, real_action_label_mode=None, recorded_action_projection=False,
                    excluded_episodes=()):
    """Audit every numeric row and sidecar; source files remain unchanged."""
    import pyarrow.parquet as pq
    if future_component_alignment not in (None, "independent_components_v1"):
        raise ValueError("unsupported future_component_alignment")
    if real_observation_mode not in (None, "collector_snapshot_v1") or real_action_label_mode not in (None, "held_command_v1"):
        raise ValueError("unsupported real collector observation/action mode")
    if real_observation_mode is None and real_action_label_mode is not None:
        raise ValueError("held actions require collector snapshot mode")
    if type(recorded_action_projection) is not bool or (recorded_action_projection and real_observation_mode is None):
        raise ValueError("recorded action projection requires explicit collector snapshots")
    if real_observation_mode is not None and (
        future_component_alignment != "independent_components_v1" or real_action_label_mode != "held_command_v1"
    ):
        raise ValueError("collector snapshots require independently sampled held future actions")
    collector_contract = None
    time_alignment = dict(TIME_ALIGNMENT)
    if future_component_alignment is not None:
        time_alignment.update(version="real_sonic_independent_future_components_v2",
            future_action_clocks={"motion_token": "action.body_received_monotonic",
                "left_hand": "action.hand_receive_monotonic[0] (aggregate fallback if absent)",
                "right_hand": "action.hand_receive_monotonic[1] (aggregate fallback if absent)"},
            future_measurement_clock="body.received_monotonic", future_component_alignment=future_component_alignment)

    root = Path(root).resolve()
    info_path, tasks_path = root / "meta/info.json", root / "meta/tasks.parquet"
    info = json.loads(info_path.read_text())
    storage_fps = info.get("fps")
    features = info["features"]
    cameras = sorted(k for k, v in features.items() if v.get("dtype") == "video")
    expected = {**dict(zip(STATE_COLUMNS, (29, 12))), **dict(zip(ACTION_COLUMNS, (64, 12)))}
    if (info.get("codebase_version") != "v3.0" or storage_fps not in (30, 50)
            or info.get("robot_type") != "g1_sonic_brainco" or CAMERA not in cameras
            or len(cameras) < 2
            or any(features.get(k, {}).get("shape") != [dim] for k, dim in expected.items())):
        raise ValueError("requires real G1/BrainCo v3 data at 30 or 50 Hz: measured q29/hand12 and motion64/hand12")
    time_alignment["storage_fps"] = storage_fps
    tasks = pq.read_table(tasks_path).to_pylist()
    if len(tasks) != 1 or not str(tasks[0]["task"]).strip():
        raise ValueError("requires one unambiguous real-robot instruction")
    count = int(info["total_episodes"])
    excluded = set(excluded_episodes)
    if any(type(ep) is not int or not 0 <= ep < count for ep in excluded):
        raise ValueError("excluded episodes must be original dataset episode indices")
    included = set(range(count)) - excluded
    if not included:
        raise ValueError("episode exclusions leave no training data")
    if count < 1 or not 0 <= validation_fraction < 1 or (validation_fraction and count < 2):
        raise ValueError("invalid episode count or held-out validation fraction")
    nval = max(1, min(len(included) - 1, round(len(included) * validation_fraction))) if validation_fraction else 0
    if validation_fraction and len(included) < 2:
        raise ValueError("held-out validation requires at least two included episodes")
    ordered = sorted(included, key=lambda ep: canonical_sha256([seed, task, ep]))
    validation = set(ordered[:nval])
    metadata = {}
    metadata_hashes = {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        metadata_hashes[str(path)] = file_sha256(path)
        for row in pq.read_table(path).to_pylist():
            ep = int(row["episode_index"])
            if ep in metadata:
                raise ValueError("duplicate real-robot episode metadata")
            metadata[ep] = row
    if set(metadata) != set(range(count)):
        raise ValueError("incomplete real-robot episode metadata")
    episodes, inventory, train_states, train_actions = [], [], [], []
    for ep in range(count):
        if ep in excluded:
            continue
        meta = metadata[ep]
        parquet = root / info["data_path"].format(chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
        table = pq.read_table(parquet, columns=["episode_index", "frame_index", "task_index", "timestamp",
                                              "observation.sample_time", *STATE_COLUMNS, *ACTION_COLUMNS])
        n = table.num_rows
        frames = np.asarray(table["frame_index"].to_pylist())
        states = np.concatenate([np.asarray(table[k].to_pylist(), dtype=np.float32) for k in STATE_COLUMNS], axis=1)
        actions = np.concatenate([np.asarray(table[k].to_pylist(), dtype=np.float32) for k in ACTION_COLUMNS], axis=1)
        if (n != meta["length"] or n < 2 or not np.array_equal(frames, np.arange(n))
                or states.shape != (n, 41) or actions.shape != (n, 76)
                or not np.isfinite(states).all() or not np.isfinite(actions).all()
                or not np.all(np.asarray(table["episode_index"].to_pylist()) == ep)
                or not np.all(np.asarray(table["task_index"].to_pylist()) == tasks[0]["task_index"])):
            raise ValueError(f"non-finite or misaligned native episode: {parquet}")
        stored = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        nominal = np.arange(n, dtype=np.float64) / storage_fps
        tolerance = np.maximum(np.spacing(nominal.astype(np.float32)), 1e-12)
        if not np.isfinite(stored).all() or np.any(np.abs(stored - nominal) > tolerance):
            raise ValueError(f"video storage timestamps disagree with the {storage_fps} Hz frame addresses")
        quality_path = root / f"meta/chenyz_episode_{ep:06d}.json"
        quality = json.loads(quality_path.read_text())
        if quality.get("episode") != ep or quality.get("frames") != n or not quality.get("storage_valid"):
            raise ValueError(f"invalid real episode storage receipt: {quality_path}")
        timing_path = root / quality["timing_file"]
        timing = [json.loads(line) for line in timing_path.open() if line.strip()]
        if len(timing) != n or [r["frame_index"] for r in timing] != list(range(n)):
            raise ValueError("timing sidecar does not address every stored frame")
        if real_observation_mode is not None:
            from psi.real_observation_contract import validate_observation_contract
            contract = validate_observation_contract(quality.get("observation_contract"))
            if any(row.get("contract") != contract for row in timing):
                raise ValueError("collector snapshot timing and episode contract disagree")
            if collector_contract is not None and collector_contract != contract:
                raise ValueError("one real dataset cannot mix collector observation contracts")
            collector_contract = contract
        decision = np.asarray([r["decision_monotonic"] for r in timing], dtype=np.float64)
        if not np.isfinite(decision).all() or (np.diff(decision) <= 0).any():
            raise ValueError("decision clock is not strictly increasing")
        origin = decision[0]
        times = decision - origin
        samples = np.asarray(table["observation.sample_time"].to_pylist(), dtype=np.float64)
        if not np.allclose(samples, times, atol=1e-7, rtol=0):
            raise ValueError("physical sample times disagree with timing sidecar")
        action_times = np.asarray([max(r["action"]["body_received_monotonic"],
                                      r["action"]["hand_received_monotonic"]) for r in timing]) - origin
        native_cycles = collector_contract is not None and collector_contract["version"] in (2, 3)
        if native_cycles:
            # Native cycles record causal observations, then the command commit.
            # Commands are supervised labels at their actual commit/write times.
            for row in timing:
                cycle = row.get("control_cycle", {})
                action = row["action"]
                observed = row["decision_monotonic"]
                completed = cycle.get("completed_monotonic", float("nan"))
                commits = [action["body_received_monotonic"], action["hand_received_monotonic"],
                           *action.get("hand_receive_monotonic", [])]
                if (cycle.get("mode") != "native_control_cycle_v1"
                        or cycle.get("observed") != observed
                        or cycle.get("committed") != action["body_received_monotonic"]
                        or cycle.get("sequence") != action.get("body_sequence")
                        or cycle.get("session") != action.get("body_source_session")
                        or action.get("timestamp_semantics") != "native_command_commit_and_returned_dds_write"
                        or not np.isfinite([observed, completed, *commits]).all()
                        or any(t < observed - 1e-9 or t > completed + 1e-9 for t in commits)):
                    raise ValueError("native control-cycle command timing or source identity mismatch")
        state_times = np.asarray([max(r["body"]["received_monotonic"],
                                     r["hand_state"]["received_monotonic"]) for r in timing]) - origin
        camera_times = {k: np.asarray([r["cameras"][k.removeprefix("observation.images.")]["received_monotonic"]
                                      for r in timing]) - origin for k in cameras}
        snapshot_state_times = None
        if real_observation_mode is not None:
            hands = [r["hand_state"].get("channel_received_monotonic", [r["hand_state"]["received_monotonic"]] * 2)
                     for r in timing]
            if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in hands):
                raise ValueError("collector snapshot requires both actual hand feedback clocks")
            state_components = np.asarray([[r["body"]["received_monotonic"], *pair]
                                           for r, pair in zip(timing, hands)], dtype=np.float64) - origin
            if not np.isfinite(state_components).all() or (state_components > times[:, None] + 1e-9).any():
                raise ValueError("collector feedback was not yet available at its recorded decision")
            snapshot_state_times = state_components.max(axis=1)
        if future_component_alignment is not None:
            hand_times = [r["action"].get("hand_receive_monotonic", [r["action"]["hand_received_monotonic"]] * 2)
                          for r in timing]
            if any(not isinstance(pair, list) or len(pair) != 2 for pair in hand_times):
                raise ValueError("future actions require two actual hand channel times")
            component_times = {
                "motion_token": np.asarray([r["action"]["body_received_monotonic"] for r in timing]) - origin,
                "left_hand": np.asarray([pair[0] for pair in hand_times]) - origin,
                "right_hand": np.asarray([pair[1] for pair in hand_times]) - origin,
            }
            body_times = np.asarray([r["body"]["received_monotonic"] for r in timing]) - origin
            causal_streams = [body_times] if native_cycles else [*component_times.values(), body_times]
            if any(np.any(t > times + 1e-9) for t in causal_streams):
                raise ValueError("future supervision source was not yet available at its stored decision")
        causal_streams = [state_times, *camera_times.values()]
        if not native_cycles:
            causal_streams.append(action_times)
        if any(np.any(t > times + 1e-9) for t in causal_streams):
            raise ValueError("recorded input was not yet available at its decision time")
        unique_stream(action_times, frames)
        unique_stream(state_times, frames)
        for t in camera_times.values():
            unique_stream(t, frames)
        video_paths, video_inventory, offsets = {}, {}, {}
        for camera in cameras:
            prefix = f"videos/{camera}"
            video = root / info["video_path"].format(video_key=camera, chunk_index=meta[prefix + "/chunk_index"],
                                                     file_index=meta[prefix + "/file_index"])
            offset = float(meta[prefix + "/from_timestamp"]) * storage_fps
            length = float(meta[prefix + "/to_timestamp"]) * storage_fps - offset
            if not np.isclose(offset, round(offset), atol=1e-4) or not np.isclose(length, n, atol=.05):
                raise ValueError("video interval does not match episode frame addresses")
            offsets[camera] = round(offset)
            stat = video.stat()
            video_paths[camera] = str(video)
            video_inventory[camera] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        split = "validation" if ep in validation else "train"
        if split == "train":
            train_states.append(states)
            train_actions.append(actions)
        # The writer marks segments after interruptions. Also split a missed
        # sample >1.75 periods; never supervise across an unrecorded interval.
        segment_ids = np.asarray([r["sample_segment"] for r in timing])
        cuts = np.flatnonzero((segment_ids[1:] != segment_ids[:-1]) | (np.diff(times) > 1.75 / storage_fps)) + 1
        parquet_hash = file_sha256(parquet)
        for seg, rows in enumerate(np.split(np.arange(n), cuts)):
            streams = {k: unique_stream(t[rows], rows + offsets[k]) for k, t in camera_times.items()}
            episodes.append(dict(episode=ep, segment=seg, split=split, times=times[rows], row_indices=rows,
                                 visual_streams=streams, action_stream=unique_stream(action_times[rows], rows),
                                 state_stream=unique_stream(state_times[rows], rows),
                                 parquet=str(parquet), parquet_sha256=parquet_hash,
                                 video_paths=video_paths, video_inventory=video_inventory))
            if future_component_alignment is not None:
                episodes[-1].update(future_component_alignment=future_component_alignment,
                    action_streams_by_component={name: unique_stream(t[rows], rows) for name, t in component_times.items()},
                    future_measurement_stream=unique_stream(body_times[rows], rows))
            if real_observation_mode is not None:
                episodes[-1].update(real_observation_mode=real_observation_mode,
                    real_observation_contract=collector_contract, real_action_label_mode=real_action_label_mode,
                    snapshot_state_times=snapshot_state_times[rows], snapshot_state_component_times=state_components[rows],
                    snapshot_visual_times_by_camera={key: values[rows] for key, values in camera_times.items()},
                    snapshot_video_offsets=offsets)
        inventory.append(dict(episode=ep, split=split, frames=n, contiguous_segments=len(cuts) + 1,
                              parquet=str(parquet), parquet_sha256=parquet_hash,
                              timing_sha256=file_sha256(timing_path), quality_sha256=file_sha256(quality_path),
                              video_paths=video_paths, video_inventory=video_inventory,
                              recorded_training_eligible=quality.get("training_eligible"),
                              recorded_task_success=quality.get("task_success"),
                              recorded_quality_exclusions=quality.get("quality_exclusions")))
    if sum(meta["length"] for meta in metadata.values()) != info["total_frames"]:
        raise ValueError("dataset total_frames mismatch")
    if sum(r["frames"] for r in inventory) != sum(metadata[ep]["length"] for ep in included):
        raise ValueError("selected dataset total_frames mismatch")
    if real_observation_mode is not None:
        time_alignment.update(real_observation_mode=real_observation_mode,
            real_observation_contract=collector_contract, real_action_label_mode=real_action_label_mode,
            current_observation_selection="same_recorded_row_snapshot_v1",
            current_state_clock="max(body.received_monotonic, actual_left_hand, actual_right_hand)",
            observation_delay="native_measured_capture_and_feedback_jitter")
        if collector_contract["version"] in (2, 3):
            time_alignment.update(action_label_timing="native_command_commit_and_returned_dds_write",
                                  command_after_observation=True)
    audit = dict(format="zeta0.real_sonic_post_training.release.v1", task=task, root=str(root),
                 time_alignment=time_alignment, info_sha256=file_sha256(info_path), tasks_sha256=file_sha256(tasks_path),
                 metadata_sha256=metadata_hashes, state_columns=list(STATE_COLUMNS), action_columns=list(ACTION_COLUMNS),
                 action_layout={"motion_token": [0, 64], "left_hand_q": [64, 70], "right_hand_q": [70, 76]},
                 quality_policy="explicit_dataset_selection_keep_unknown_success_split_gaps_and_mask_unavailable_targets",
                 statistics_split="train", split_seed=seed, validation_fraction=validation_fraction,
                 train_episodes=sorted(included - validation), validation_episodes=sorted(validation),
                 episodes=inventory)
    if excluded:
        audit["excluded_episodes"] = sorted(excluded)
        audit["excluded_frames"] = sum(metadata[ep]["length"] for ep in excluded)
    if real_observation_mode is not None:
        audit.update(real_observation_mode=real_observation_mode, real_observation_contract=collector_contract,
                     real_action_label_mode=real_action_label_mode)
    if recorded_action_projection:
        from psi.real_observation_contract import build_recorded_action_projection
        tokens = np.concatenate(train_actions)[:, :64]
        if np.any(np.abs(tokens / MOTION_TOKEN_STEP - np.rint(tokens / MOTION_TOKEN_STEP)) > 1e-6):
            raise ValueError("recorded motion tokens are not on the actual FSQ1/16 source grid")
        source = dict(train_episodes=audit["train_episodes"], action_columns=list(ACTION_COLUMNS),
                      motion_token_slice=[0, 64], native_action_dim=76,
                      numeric_parquets=[dict(episode=entry["episode"], sha256=entry["parquet_sha256"])
                                        for entry in inventory if entry["split"] == "train"])
        projection = build_recorded_action_projection(float(tokens.min()), float(tokens.max()), canonical_sha256(source))
        audit.update(real_action_projection=projection, real_action_projection_source=source)
        audit["time_alignment"]["real_action_projection"] = projection
        for episode in episodes:
            episode["real_action_projection"] = projection
    return info, str(tasks[0]["task"]), episodes, audit, fit_statistics(np.concatenate(train_states), np.concatenate(train_actions))


def extend_registries(source, info, statistics, *, task, audit_identity):
    registries = copy.deepcopy(source)
    names = info["features"]["observation.state"]["names"]
    arms = [f"{side}_{joint}_joint" for side in ("left", "right")
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    indices = [names.index(k) for k in arms] + [None] * 15
    canonical = copy.deepcopy(next(iter(source["measurement_registry"].values()))["canonical_body"])
    canonical.update(raw_indices=indices,
                     mean=[statistics["body_mean"][i] if i is not None else 0. for i in indices],
                     std=[statistics["body_std"][i] if i is not None else 1. for i in indices])
    entries = (("action_protocol_registry", PROTOCOL, statistics["action"]),
               ("state_schema_registry", STATE, statistics["state"]),
               ("measurement_registry", MEASUREMENT, dict(width=41, world_action_eligible=True, canonical_body=canonical,
                fields=[dict(name="measured_q29", start=0, stop=29, kind="measurement", unit="rad",
                             coordinate_frame="robot_native", time_key="state_received_monotonic", statistics_id=STATE),
                        dict(name="measured_brainco12", start=29, stop=41, kind="measurement", unit="normalized_position",
                             coordinate_frame="brainco_native", time_key="state_received_monotonic", statistics_id=STATE)])))
    for registry, key, value in entries:
        if key in registries[registry]:
            raise ValueError(f"source already contains {key}; use exact resume or a new protocol version")
        registries[registry][key] = value
    registries["capability_registry"] = {task: dict(measurement_schema_id=MEASUREMENT, action_protocol_id=PROTOCOL,
        visual_keys=sorted(k for k, v in info["features"].items() if v.get("dtype") == "video"),
        future_visual_keys=[CAMERA], independent_timestamps_across_visual_keys=True,
        supports_future_visual=True, supports_future_measurement=True, supports_world_action=True,
        audit_revision="real_sonic_workstation_availability_" + audit_identity)}
    return registries


def episode_windows(episode, *, task, instruction, temporal, capability, measurement_schema):
    primary = episode["visual_streams"][CAMERA]
    action = episode["action_stream"]
    state = episode["state_stream"]
    record = dict(source_id=task, episode_id=f"{task}/episode_{episode['episode']:06d}/segment_{episode['segment']:04d}",
                  split=episode["split"], anchor_times=episode["times"],
                  visual_timestamps=primary["timestamps"], visual_row_indices=primary["row_indices"],
                  visual_streams=episode["visual_streams"], state_timestamps=state["timestamps"], state_row_indices=state["row_indices"],
                  action_timestamps=action["timestamps"], action_row_indices=action["row_indices"],
                  materialization=dict(data_path=episode["parquet"], video_paths=episode["video_paths"],
                    primary_camera_id=CAMERA, instruction=instruction, state_columns=list(STATE_COLUMNS),
                    action_columns=list(ACTION_COLUMNS), measurement_schema_id=MEASUREMENT, state_schema_id=STATE,
                    action_protocol_id=PROTOCOL, action_normalization_space=SPACE,
                    measurement_indices=list(measurement_schema.canonical_body.raw_indices)))
    components = episode.get("action_streams_by_component")
    if components is not None:
        measured = episode["future_measurement_stream"]
        record.update(future_measurement_timestamps=measured["timestamps"], future_measurement_row_indices=measured["row_indices"])
        record["materialization"].update(action_component_slices=copy.deepcopy(ACTION_COMPONENT_SLICES),
            future_component_alignment=episode["future_component_alignment"])
    if episode.get("real_observation_mode") is not None:
        if len(temporal["state_offsets_s"]) != 1:
            raise ValueError("collector snapshots require exactly one current state")
        record["materialization"].update(real_observation_mode=episode["real_observation_mode"],
            real_observation_contract=episode["real_observation_contract"],
            real_action_label_mode=episode["real_action_label_mode"],
            collector_video_offsets=episode["snapshot_video_offsets"])
    if episode.get("real_action_projection") is not None:
        record["materialization"]["real_action_projection"] = episode["real_action_projection"]
    rows = build_episode_windows(record, history_visual_offsets_s=temporal["history_visual_offsets_s"],
        state_offsets_s=temporal["state_offsets_s"], future_group_ends_s=temporal["future_group_ends_s"],
        action_offsets_s=temporal["action_offsets_s"], future_measurement_offsets_s=temporal["future_measurement_offsets_s"],
        visual_max_age_s=temporal["selection"]["history_visual"]["max_age_s"],
        state_max_age_s=temporal["selection"]["state_history"]["max_age_s"],
        action_max_age_s=temporal["selection"]["demonstrated_action"]["max_age_s"],
        future_measurement_max_age_s=temporal["selection"]["future_measurement"]["max_age_s"],
        future_visual_max_age_s=temporal["selection"]["future_visual"]["max_age_s"],
        capability=capability, measurement_schema=measurement_schema)
    if components is not None:
        offsets = np.asarray(temporal["action_offsets_s"], dtype=np.float64)
        for row in rows:
            anchor = row["anchor_time"]
            indices_by_component, valid_by_component, times_by_component = {}, {}, {}
            for name, stream in components.items():
                indices, valid, actual = causal_floor_indices(stream["timestamps"], anchor + offsets,
                    max_age_seconds=temporal["selection"]["demonstrated_action"]["max_age_s"])
                if episode.get("real_action_label_mode") != "held_command_v1":
                    valid &= actual > anchor + 1e-12
                indices_by_component[name] = stream["row_indices"][indices].tolist()
                valid_by_component[name] = valid.tolist()
                times_by_component[name] = np.where(valid, actual - anchor, np.nan).tolist()
            component_valid = np.asarray(list(valid_by_component.values()), dtype=bool)
            actual = np.asarray(list(times_by_component.values()), dtype=np.float64)
            complete = component_valid.all(axis=0)
            aggregate = np.max(np.where(component_valid, actual, -np.inf), axis=0)
            row.update(action_indices_by_component=indices_by_component,
                action_valid_by_component=valid_by_component, action_actual_times_by_component=times_by_component,
                action_indices=indices_by_component["motion_token"], action_valid=complete.tolist(),
                action_actual_times=np.where(component_valid.any(axis=0), aggregate, np.nan).tolist(),
                world_action_group_valid=[bool(capability.supports_world_action and (offsets <= end + 1e-12).any()
                    and complete[offsets <= end + 1e-12].all()) for end in temporal["future_group_ends_s"]])
            row["eligibility"] = predictive_world_eligibility(row)
    if episode.get("real_observation_mode") is not None:
        from psi.real_observation_contract import snapshot_age_limit_s
        max_age = snapshot_age_limit_s(episode["real_observation_contract"], temporal.get("real_snapshot_max_age_s"))
        for ordinal, row in enumerate(rows):
            source_row = int(episode["row_indices"][ordinal])
            anchor = row["anchor_time"]
            component_times = episode["snapshot_state_component_times"][ordinal]
            state_valid = bool(np.all((component_times <= anchor + 1e-9) & (anchor - component_times <= max_age + 1e-9)))
            row.update(state_indices=[source_row], state_valid=[state_valid],
                       state_actual_times=[float(episode["snapshot_state_times"][ordinal] - anchor)],
                       collector_snapshot_row_index=source_row)
            for camera in capability.visual_keys:
                time = float(episode["snapshot_visual_times_by_camera"][camera][ordinal] - anchor)
                valid = bool(-max_age - 1e-9 <= time <= 1e-9)
                image_row = source_row + episode["snapshot_video_offsets"][camera]
                row["history_visual_indices_by_camera"][camera][-1] = image_row
                row["history_visual_valid_by_camera"][camera][-1] = valid
                row["history_visual_actual_times_by_camera"][camera][-1] = time
                if camera == CAMERA:
                    row["history_visual_indices"][-1] = image_row
                    row["history_visual_valid"][-1] = valid
                    row["history_visual_actual_times"][-1] = time
            row["eligibility"] = predictive_world_eligibility(row)
    return rows
