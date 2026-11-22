"""Build a Zeta index over the audited, packed seven-task LeRobot aggregate.

The aggregate's episode/task mapping and original semantic feature definitions
are explicit inputs. Numeric rows and video offsets are read in their actual
execution namespace; no relocation of an old index or feature reuse is assumed.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from psi.config.predictive_world import PredictiveWorldDataConfig
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.index import publish_predictive_world_index, temporal_abi_from_data_config
from psi.data.predictive_world.measurement_schema import load_capability_registry, load_measurement_registry
from psi.data.predictive_world.sonic_joint_training import JOINT_DATA_FORMAT, immutable_json
from psi.data.predictive_world.sonic_post_training import (
    PROTOCOL, STATE, TASKS, TIME_ALIGNMENT, canonical_frame_timestamps, episode_windows, fit_statistics,
)


def prepare_aggregate(dataset, aggregate_manifest, source_info_dir, registry_reference, output):
    """Audit all 700 demonstrations and publish fresh metadata/index, without a GPU."""
    import pyarrow.parquet as pq

    dataset, aggregate_manifest, source_info_dir, registry_reference, output = (
        Path(p).resolve() for p in (dataset, aggregate_manifest, source_info_dir, registry_reference, output))
    if output.exists():
        raise FileExistsError(output)
    info = json.loads((dataset / "meta/info.json").read_text())
    manifest = json.loads(aggregate_manifest.read_text())
    sources = {value["task"]: value for value in manifest["sources"]}
    if (set(sources) != set(TASKS) or len(manifest["sources"]) != 7
            or manifest.get("native_aggregate") is not True
            or manifest.get("concatenate_data") is not False
            or manifest.get("concatenate_videos") is not False
            or info.get("codebase_version") != "v3.0" or info.get("fps") != 50
            or info.get("total_episodes") != 700 or info.get("total_tasks") != 7
            or info.get("total_frames") != manifest.get("frames")
            or info.get("splits") != {"train": "0:700"}):
        raise ValueError("expected the audited all-train seven-task LeRobot v3 aggregate")
    instructions = {value["task"]: value["task_text"] for value in manifest["sample_checks"]}
    if set(instructions) != set(TASKS) or len(set(instructions.values())) != 7:
        raise ValueError("aggregate must preserve seven distinct task instructions")
    task_rows = pq.read_table(dataset / "meta/tasks.parquet").to_pylist()
    instruction_ids = {row["task"]: row["task_index"] for row in task_rows}
    if (len(task_rows) != 7 or set(instruction_ids) != set(instructions.values())
            or set(instruction_ids.values()) != set(range(7))):
        raise ValueError("aggregate task table differs from its preparation audit")
    episode_tasks = {}
    for task, source in sources.items():
        original = source_info_dir / (task + ".json")
        if file_sha256(original) != source["original_info_sha256"]:
            raise ValueError("source semantic metadata digest differs from aggregate provenance")
        old = json.loads(original.read_text())
        if old.get("vla_protocol", {}).get("schema") != "unitree_g1_gmt_refpose_v3_1":
            raise ValueError("source action semantics are not the audited SONIC protocol")
        for key in ("observation.state", "action"):
            expected, actual = dict(old["features"][key]), dict(info["features"][key])
            expected.pop("fps", None)
            actual.pop("fps", None)
            if expected != actual:
                raise ValueError("aggregate changed ordered state/action semantic coordinates")
        start, stop = source["episode_from"], source["episode_to"]
        if source["episodes"] != 100 or stop - start != 100:
            raise ValueError("each joint task must contain exactly 100 demonstrations")
        for episode in range(start, stop):
            if episode in episode_tasks:
                raise ValueError("aggregate episode task ranges overlap")
            episode_tasks[episode] = task
    if set(episode_tasks) != set(range(700)):
        raise ValueError("aggregate episode task mapping is incomplete")
    cameras = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    if cameras != ["observation.images.front"]:
        raise ValueError("joint aggregate requires one front camera")
    columns = ["episode_index", "length", "tasks", "data/chunk_index", "data/file_index",
        "videos/observation.images.front/chunk_index", "videos/observation.images.front/file_index",
        "videos/observation.images.front/from_timestamp", "videos/observation.images.front/to_timestamp"]
    metadata = {}
    for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            if row["episode_index"] in metadata:
                raise ValueError("duplicate aggregate episode metadata")
            metadata[row["episode_index"]] = row
    if set(metadata) != set(episode_tasks):
        raise ValueError("aggregate episode metadata coverage mismatch")
    episodes, states, actions, hashes, table_cache = [], [], [], {}, {}
    frame_counts = dict.fromkeys(TASKS, 0)
    for episode, task in sorted(episode_tasks.items()):
        row = metadata[episode]
        if row["tasks"] != [instructions[task]]:
            raise ValueError("episode task instruction differs from aggregate audit")
        parquet = dataset / info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
        video = dataset / info["video_path"].format(video_key=cameras[0],
            chunk_index=row["videos/observation.images.front/chunk_index"],
            file_index=row["videos/observation.images.front/file_index"])
        if parquet not in table_cache:
            table_cache.clear()
            table_cache[parquet] = pq.read_table(parquet, columns=["episode_index", "frame_index", "timestamp",
                "task_index", "observation.state", "action"])
            hashes[str(parquet)] = file_sha256(parquet)
        full = table_cache[parquet]
        indices = np.flatnonzero(np.asarray(full["episode_index"].to_pylist()).reshape(-1) == episode)
        table = full.take(indices)
        times = np.asarray(table["timestamp"].to_pylist()).reshape(-1)
        frames = np.asarray(table["frame_index"].to_pylist()).reshape(-1)
        local_states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        local_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        offset = float(row["videos/observation.images.front/from_timestamp"]) * 50
        stop = float(row["videos/observation.images.front/to_timestamp"]) * 50
        if (len(times) != row["length"] or len(times) < 2 or local_states.shape != (len(times), 64)
                or local_actions.shape != (len(times), 40) or not np.isfinite(local_states).all()
                or not np.isfinite(local_actions).all()
                or not np.all(np.asarray(table["task_index"].to_pylist()) == instruction_ids[instructions[task]])
                or not np.isclose(offset, round(offset), atol=1e-3, rtol=0)
                or not np.isclose(stop - round(offset), len(times), atol=.05, rtol=0)):
            raise ValueError("aggregate episode numeric/task/video alignment mismatch")
        times = canonical_frame_timestamps(times, frames)
        stat = video.stat()
        episodes.append(dict(episode=episode, task=task, split="train", times=times,
            row_indices=indices, frame_offset=round(offset), frame_grid_hz=50,
            parquet=str(parquet), video=str(video), video_size=stat.st_size, video_mtime_ns=stat.st_mtime_ns))
        states.append(local_states)
        actions.append(local_actions)
        frame_counts[task] += len(times)
    if frame_counts != {task: source["frames"] for task, source in sources.items()}:
        raise ValueError("aggregate task frame counts differ from the audited source demonstrations")
    statistics = fit_statistics(np.concatenate(states), np.concatenate(actions))
    del states, actions, table_cache
    names = ("action_protocol_registry", "state_schema_registry", "measurement_registry", "capability_registry")
    registries = {name: json.loads((registry_reference / (name + ".json")).read_text()) for name in names}
    if (set(registries["action_protocol_registry"]) != {PROTOCOL}
            or set(registries["state_schema_registry"]) != {STATE}
            or len(registries["measurement_registry"]) != 1
            or set(registries["capability_registry"]) != set(TASKS)):
        raise ValueError("reference must contain exactly the shared seven-task SONIC schemas")
    measurement_id, measurement = next(iter(registries["measurement_registry"].items()))
    canonical = measurement["canonical_body"]
    if len(canonical["raw_indices"]) != 29 or set(canonical["raw_indices"]) != set(range(6, 35)):
        raise ValueError("body supervision must cover all 29 measured joint positions")
    for registry, key, fitted in (("action_protocol_registry", PROTOCOL, statistics["action"]),
                                  ("state_schema_registry", STATE, statistics["state"])):
        for field in ("mean", "std"):
            np.testing.assert_allclose(registries[registry][key][field], fitted[field], rtol=1e-6, atol=1e-7,
                err_msg="aggregate numeric statistics differ from the original joint study")
        registries[registry][key].update(fitted)
    for field in ("mean", "std"):
        values = [statistics["body_" + field][i] for i in canonical["raw_indices"]]
        np.testing.assert_allclose(canonical[field], values, rtol=1e-6, atol=1e-7)
        canonical[field] = values
    audit = dict(format="zeta0.sonic_aggregate.audit.v1", dataset=str(dataset),
        info_sha256=file_sha256(dataset / "meta/info.json"), aggregate_manifest_sha256=file_sha256(aggregate_manifest),
        reference_registries_sha256={name: file_sha256(registry_reference / (name + ".json")) for name in names},
        parquet_sha256=hashes, frame_counts=frame_counts, time_alignment=TIME_ALIGNMENT,
        train_episodes=700, validation_episodes=0, statistics_split="train",
        statistics_weighting="native_training_frames", original_statistics_verified=True,
        episodes=[{key: value for key, value in ep.items() if key not in {"times", "row_indices"}} for ep in episodes])
    audit_id = canonical_sha256(audit)
    for capability in registries["capability_registry"].values():
        capability["audit_revision"] = "sonic_aggregate_" + audit_id
    output.mkdir(parents=True)
    paths = {name: immutable_json(output / (name + ".json"), payload) for name, payload in registries.items()}
    immutable_json(output / "dataset_audit.json", audit)
    immutable_json(output / "statistics.json", statistics)
    data = PredictiveWorldDataConfig(action_horizon=40, state_history_count=1,
        observed_visual_mode="current_multiview", current_visual_pooling="content_area_v1", source_sampling_exponent=0.,
        action_protocol_registry=str(paths["action_protocol_registry"]), state_schema_registry=str(paths["state_schema_registry"]),
        measurement_registry=str(paths["measurement_registry"]), canonical_future_capabilities=str(paths["capability_registry"]),
        index_path=str(output / "predictive_world_index.jsonl"), feature_cache=str(output / "feature_cache"))
    schemas = load_measurement_registry(paths["measurement_registry"])
    capabilities = load_capability_registry(paths["capability_registry"], measurement_schemas=schemas,
        action_protocol_ids={PROTOCOL})
    temporal = temporal_abi_from_data_config(data)

    def windows():
        for ep in sorted(episodes, key=lambda ep: (ep["task"], ep["episode"])):
            yield from episode_windows(ep, task=ep["task"], instruction=instructions[ep["task"]],
                temporal=temporal, capability=capabilities[ep["task"]], measurement_schema=schemas[measurement_id],
                measurement=measurement_id)

    publish_predictive_world_index(data.index_path, rows=windows(), presorted=True, temporal_abi=temporal,
        provenance=dict(registry_binding={key + "_sha256": file_sha256(path) for key, path in paths.items()},
            aggregate_audit_identity=audit_id, source_sampling_exponent=0., statistics_split="train"))
    release = dict(format=JOINT_DATA_FORMAT, data=asdict(data), state_schema_id=STATE, action_protocol_id=PROTOCOL,
        tasks=sorted(TASKS), demonstrations_per_task=100, train_episodes=700, validation_episodes=0,
        task_sampling_probability=1/7, feature_cache_built=False, dataset_audit_identity=audit_id)
    release["identity"] = canonical_sha256(release)
    immutable_json(output / "joint_data.json", release)
    return dict(source_data=str(output / "joint_data.json"), rows=sum(frame_counts.values()), training_started=False)
