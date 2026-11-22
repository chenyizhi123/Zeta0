"""Shared train-only SONIC statistics and an immutable multi-task data release.

The visual tensors are independent of state/action/body normalization. Reuse
requires identical feature provenance and a row-by-row check of future visual
queries against the audited source index; no cached latent targets are reused.
"""
from __future__ import annotations

import copy
import errno
import itertools
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from psi.config.predictive_world import PredictiveWorldDataConfig
from psi.data.predictive_world.assets import resolve_asset_path
from psi.data.predictive_world.cache import (
    PACKED_FEATURE_FORMAT, canonical_sha256, file_sha256,
    verify_predictive_world_feature_manifest,
)
from psi.data.predictive_world.dataset import assert_predictive_world_training_data_contract
from psi.data.predictive_world.index import (
    predictive_world_index_binding, publish_predictive_world_index,
    verify_predictive_world_index,
)
from psi.data.predictive_world.measurement_schema import load_capability_registry, load_measurement_registry
from psi.data.predictive_world.sonic_post_training import (
    MEASUREMENT, PROTOCOL, STATE, TASKS, TIME_ALIGNMENT, episode_windows, fit_statistics, inspect_dataset,
)

JOINT_DATA_FORMAT = "zeta0.sonic_joint_training.data.v1"


def immutable_json(path, value):
    """Publish complete JSON without replacing a prior contract, including races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"immutable joint data differs: {path}")
        return path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != encoded:
                raise ValueError(f"immutable joint data differs: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _manifest_entries(path, expected_tasks):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    entries = value.get("tasks") if isinstance(value, dict) else None
    if entries is None:
        entries = [dict(task=task, **item) for task, item in value.items()]
    tasks = [item["task"] for item in entries]
    roots = [str(Path(item["dataset"]).expanduser().resolve()) for item in entries]
    if (not tasks or len(tasks) != len(set(tasks)) or len(roots) != len(set(roots))
            or set(tasks) != set(expected_tasks)):
        raise ValueError("joint manifest must contain exactly the expected distinct tasks and dataset roots")
    return path, sorted(entries, key=lambda item: item["task"])


def inspect_task_manifest(path, *, seed=292285, validation_fraction=0.0,
                          expected_tasks=TASKS, expected_episodes=100):
    """Audit each source and fit common statistics on raw training frames only.

    ``row_indices`` select an episode within a v3 shared parquet shard, so a
    held-out episode cannot leak into the pooled training statistics.
    """
    import pyarrow.parquet as pq

    path, entries = _manifest_entries(path, expected_tasks)
    if type(expected_episodes) is not int or expected_episodes < 1:
        raise ValueError("expected_episodes must be a positive integer")
    audits, episodes, train_states, train_actions = [], [], [], []
    reference_info = reference_signature = None
    frame_counts = {}
    for entry in entries:
        task = entry["task"]
        info, instruction, local_episodes, audit, _ = inspect_dataset(
            entry["dataset"], task=task, seed=seed, validation_fraction=validation_fraction)
        if len(local_episodes) != expected_episodes:
            raise ValueError(f"{task} must have exactly {expected_episodes} demonstrations")
        cameras = sorted(key for key, value in info["features"].items()
                         if key.startswith("observation.images.") and value.get("dtype") == "video")
        cameras = cameras or ["observation.images.front"]
        if cameras != ["observation.images.front"] or entry.get("expected_camera_count", 1) != 1:
            raise ValueError("joint SONIC data requires the same single front camera")
        signature = {key: info.get(key) for key in ("fps", "vla_protocol")}
        signature["features"] = {key: copy.deepcopy(info["features"][key])
                                 for key in ("observation.state", "action")}
        if reference_info is not None and signature != reference_signature:
            raise ValueError("joint tasks disagree on state/action feature names, units or protocol")
        reference_info, reference_signature = info, signature
        audits.append(audit)
        table_cache = {}
        frame_counts[task] = 0
        for episode in local_episodes:
            episode.update(task=task, instruction=instruction)
            episodes.append(episode)
            if episode["split"] != "train":
                continue
            parquet = episode["parquet"]
            if parquet not in table_cache:
                table_cache.clear()
                table_cache[parquet] = pq.read_table(parquet, columns=["observation.state", "action"])
            table = table_cache[parquet].take(episode["row_indices"])
            train_states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
            train_actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
            frame_counts[task] += len(table)
    statistics = fit_statistics(np.concatenate(train_states), np.concatenate(train_actions))
    audit = dict(format="zeta0.sonic_joint_training.audit.v1", tasks=audits,
        manifest_sha256=file_sha256(path), time_alignment=dict(TIME_ALIGNMENT),
        split_seed=seed, validation_fraction=validation_fraction,
        statistics_split="train", statistics_weighting="native_training_frames",
        statistics_train_frame_counts=frame_counts,
        sampling="equal_task_1_over_task_count", source_sampling_exponent=0.0,
        train_episodes=[f"{a['task']}/episode_{ep:06d}" for a in audits for ep in a["train_episodes"]],
        validation_episodes=[f"{a['task']}/episode_{ep:06d}" for a in audits for ep in a["validation_episodes"]])
    return reference_info, None, episodes, audit, statistics


def _read_data_config(payload):
    values = dict(payload["data"])
    for key in ("history_visual_offsets_s", "future_group_ends_s"):
        if key in values:
            values[key] = tuple(values[key])
    mapping = payload["runtime"].get("asset_path_map", {})
    for key in ("index_path", "feature_cache", "action_protocol_registry", "state_schema_registry",
                "measurement_registry", "canonical_future_capabilities"):
        values[key] = str(resolve_asset_path(values[key], mapping))
    return PredictiveWorldDataConfig(**values)


def _without_statistics(value):
    value = copy.deepcopy(value)
    value.pop("mean", None)
    value.pop("std", None)
    if value.get("canonical_body"):
        value["canonical_body"].pop("mean", None)
        value["canonical_body"].pop("std", None)
    return value


def shared_registries(registries_by_task, statistics, *, audit_identity):
    """Replace only common SONIC statistics; all other registry definitions agree."""
    tasks = sorted(registries_by_task)
    result = copy.deepcopy(registries_by_task[tasks[0]])
    for task in tasks:
        local = registries_by_task[task]
        for name in ("action_protocol_registry", "state_schema_registry", "measurement_registry"):
            if set(local[name]) != set(result[name]):
                raise ValueError(f"task registry keys differ: {name}")
            for key, value in local[name].items():
                common_sonic = key in {PROTOCOL, STATE, MEASUREMENT} or key.startswith("sonic_measured_")
                left = _without_statistics(value) if common_sonic else value
                right = _without_statistics(result[name][key]) if common_sonic else result[name][key]
                if left != right:
                    raise ValueError(f"task registry physical definitions differ: {name}/{key}")
    result["action_protocol_registry"][PROTOCOL].update(copy.deepcopy(statistics["action"]))
    result["state_schema_registry"][STATE].update(copy.deepcopy(statistics["state"]))
    for key, schema in result["measurement_registry"].items():
        if key != MEASUREMENT and not key.startswith("sonic_measured_"):
            continue
        canonical = schema.get("canonical_body")
        if canonical is None:
            continue
        indices = canonical["raw_indices"]
        canonical["mean"] = [statistics["body_mean"][i] if i is not None else 0. for i in indices]
        canonical["std"] = [statistics["body_std"][i] if i is not None else 1. for i in indices]
    capabilities = {}
    for task in tasks:
        local = registries_by_task[task]["capability_registry"]
        if set(local) != {task}:
            raise ValueError("each feature reference must contain exactly its declared task")
        capability = copy.deepcopy(local[task])
        capability["audit_revision"] = "sonic_joint_train_statistics_" + audit_identity
        capabilities[task] = capability
    result["capability_registry"] = capabilities
    return result


def _visual_query_identity(row):
    material = row["materialization"]
    return canonical_sha256({key: row[key] for key in (
        "sample_id", "source_id", "episode_id", "anchor_time", "anchor_ordinal",
        "future_visual_indices", "future_visual_valid", "future_visual_actual_times",
    )} | {"video_paths": material["video_paths"], "primary_camera_id": material["primary_camera_id"]})


def _link_shard(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(source, target):
            raise ValueError("joint feature shard points at another object")
        return
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        target.symlink_to(source)


def prepare_joint_release(dataset_manifest, reference_configs, output, *, seed=292285,
                          validation_fraction=0.0, expected_tasks=TASKS, expected_episodes=100,
                          verify_feature_hashes=True):
    """Build joint metadata/index and link verified immutable packed visual shards."""
    _, _, episodes, audit, statistics = inspect_task_manifest(dataset_manifest, seed=seed,
        validation_fraction=validation_fraction, expected_tasks=expected_tasks, expected_episodes=expected_episodes)
    tasks = sorted(expected_tasks)
    if set(reference_configs) != set(tasks):
        raise ValueError("reference configurations must cover every joint task exactly")
    output = Path(output).expanduser().resolve()
    references, registries, packed = {}, {}, {}
    reference_data = reference_temporal = feature_provenance = qwen_model = None
    for task in tasks:
        config_path = Path(reference_configs[task]).expanduser().resolve()
        payload = json.loads(config_path.read_text())
        data = _read_data_config(payload)
        artifacts = assert_predictive_world_training_data_contract(data, measurement_dim=payload["model"]["measurement_dim"])
        temporal = artifacts.index_meta["temporal_abi"]
        if (data.action_hz != 50 or data.state_history_count != 1 or data.action_horizon != 50
                or data.observed_visual_mode != "current_multiview"):
            raise ValueError("joint references require current state/RGB and 50-step SONIC at50Hz")
        manifest = verify_predictive_world_feature_manifest(artifacts.feature_manifest,
            verify_shard_hashes=verify_feature_hashes)
        if manifest["format"] != PACKED_FEATURE_FORMAT or manifest["groups"] != 3:
            raise ValueError("joint feature reuse requires packed three-group caches")
        local_qwen = str(resolve_asset_path(payload["model"]["qwen_model"], payload["runtime"].get("asset_path_map", {})))
        if reference_data is not None and (
                reference_temporal != temporal or feature_provenance != manifest["provenance"]
                or qwen_model != local_qwen or reference_data.current_visual_pooling != data.current_visual_pooling):
            raise ValueError("joint references disagree on temporal ABI, VLM, preprocessing or feature provenance")
        reference_data, reference_temporal = data, temporal
        feature_provenance, qwen_model = manifest["provenance"], local_qwen
        packed[task] = dict(path=str(artifacts.feature_manifest), manifest=manifest)
        references[task] = dict(config=str(config_path), config_sha256=file_sha256(config_path),
            index_path=data.index_path, index_binding=manifest["index_binding"],
            feature_manifest=str(artifacts.feature_manifest), feature_manifest_sha256=file_sha256(artifacts.feature_manifest))
        registries[task] = {name: json.loads(Path(getattr(data, name)).read_text()) for name in (
            "action_protocol_registry", "state_schema_registry", "measurement_registry")}
        registries[task]["capability_registry"] = json.loads(Path(data.canonical_future_capabilities).read_text())
    common = shared_registries(registries, statistics, audit_identity=canonical_sha256(audit))
    paths = {name: output / f"{name}.json" for name in common}
    for name, value in common.items():
        immutable_json(paths[name], value)
    immutable_json(output / "dataset_audit.json", audit)
    immutable_json(output / "statistics.json", statistics)
    schemas = load_measurement_registry(paths["measurement_registry"])
    capabilities = load_capability_registry(paths["capability_registry"], measurement_schemas=schemas,
        action_protocol_ids=set(common["action_protocol_registry"]))
    binding = {name + "_sha256": file_sha256(path) for name, path in paths.items()}
    provenance = dict(registry_binding=binding, dataset_audit_identity=canonical_sha256(audit),
        anchor_selection="every_native_frame_50hz_joint_v1", statistics_split="train",
        statistics_weighting="native_training_frames", time_alignment=dict(TIME_ALIGNMENT),
        source_releases=references, feature_reuse="identical_future_visual_queries_in_source_row_order")
    index = output / "predictive_world_index.jsonl"
    task_counts = {}
    def rows():
        for task in tasks:
            capability = capabilities[task]
            schema = schemas[capability.measurement_schema_id]
            def expected_rows():
                for episode in episodes:
                    if episode["task"] == task:
                        yield from episode_windows(episode, task=task, instruction=episode["instruction"],
                            temporal=reference_temporal, capability=capability, measurement_schema=schema,
                            measurement=schema.schema_id)
            task_counts[task] = 0
            with Path(references[task]["index_path"]).open() as source:
                recorded = (json.loads(line) for line in source if line.strip())
                for ordinal, (old, new) in enumerate(itertools.zip_longest(recorded, expected_rows())):
                    if old is None or new is None or old.get("feature_row_index") != ordinal:
                        raise ValueError(f"{task} source index row count/order disagrees with raw episodes")
                    if _visual_query_identity(old) != _visual_query_identity(new):
                        raise ValueError(f"{task} cached future visual query differs from audited raw data")
                    task_counts[task] += 1
                    yield new
            if task_counts[task] != packed[task]["manifest"]["rows"]:
                raise ValueError("packed feature rows disagree with source index")
    publish_predictive_world_index(index, rows=rows(), provenance=provenance,
        temporal_abi=reference_temporal, presorted=True)
    cache_dir, shards, offset = output / "feature_cache", [], 0
    for task in tasks:
        source = packed[task]
        for local in source["manifest"]["shards"]:
            relative = Path("shards") / f"{len(shards):05d}-{task}.npy"
            _link_shard(Path(source["path"]).parent / local["file"], cache_dir / relative)
            shards.append(dict(file=str(relative), rank=len(shards), start=offset + local["start"],
                stop=offset + local["stop"], nbytes=local["nbytes"], sha256=local["sha256"]))
        offset += source["manifest"]["rows"]
    feature_manifest = dict(format=PACKED_FEATURE_FORMAT, provenance=feature_provenance,
        cache_identity=canonical_sha256(feature_provenance), index_binding=predictive_world_index_binding(index),
        rows=offset, groups=3, feature_shape=[16, 2048], dtype="float16", shards=shards)
    feature_manifest["manifest_identity"] = canonical_sha256({key: feature_manifest[key] for key in (
        "cache_identity", "index_binding", "rows", "groups", "shards")})
    immutable_json(cache_dir / "feature_manifest.json", feature_manifest)
    data_values = dict(json.loads(Path(reference_configs[tasks[0]]).read_text())["data"])
    data_values.update({name: str(path) for name, path in paths.items() if name != "capability_registry"})
    data_values.update(canonical_future_capabilities=str(paths["capability_registry"]), index_path=str(index),
        feature_cache=str(cache_dir), source_sampling_exponent=0.0)
    # Use resolved data fields, even when a reference originated from a relocated export.
    data_values.update(current_visual_pooling=reference_data.current_visual_pooling)
    data_config = _read_data_config(dict(data=data_values, runtime={}))
    checked = assert_predictive_world_training_data_contract(data_config, measurement_dim=29)
    meta = verify_predictive_world_index(index, verify_content_hashes=False)
    if meta["rows"] != offset or set(checked.capabilities) != set(tasks):
        raise ValueError("joint release coverage differs from the selected tasks")
    result = dict(format=JOINT_DATA_FORMAT, data=data_values, model=dict(qwen_model=qwen_model),
        tasks=tasks, demonstrations_per_task=expected_episodes, task_sampling_probability=1 / len(tasks),
        source_sampling_exponent=0.0, train_episodes=len(audit["train_episodes"]),
        validation_episodes=len(audit["validation_episodes"]), split_seed=seed,
        statistics=str(output / "statistics.json"), statistics_sha256=file_sha256(output / "statistics.json"),
        dataset_audit=str(output / "dataset_audit.json"), index_binding=feature_manifest["index_binding"],
        feature_manifest_identity=feature_manifest["manifest_identity"],
        feature_content_verified=bool(verify_feature_hashes), reference_configs=references,
        dataset_roots={entry["task"]: entry["root"] for entry in audit["tasks"]},
        source_instructions={episode["task"]: episode["instruction"] for episode in episodes},
        camera_keys=["observation.images.front"],
        action_protocol_id=PROTOCOL, state_schema_id=STATE,
        source_rows={task: packed[task]["manifest"]["rows"] for task in tasks})
    result["identity"] = canonical_sha256(result)
    immutable_json(output / "joint_data.json", result)
    return result
