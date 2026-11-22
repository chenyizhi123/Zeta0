"""Joint statistics, cached-feature identity and deterministic multi-task sampling."""
from dataclasses import asdict
from itertools import islice
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi.config.predictive_world import PredictiveWorldDataConfig
from psi.data.predictive_world.cache import (
    PACKED_FEATURE_FORMAT, PackedFeatureStore, canonical_sha256, file_sha256,
    verify_predictive_world_feature_manifest,
)
from psi.data.predictive_world.dataset import PredictiveWorldOccurrenceDataset
from psi.data.predictive_world.index import predictive_world_index_binding, publish_predictive_world_index, temporal_abi_from_data_config
from psi.data.predictive_world.measurement_schema import load_capability_registry, load_measurement_registry
from psi.data.predictive_world.sonic_joint_training import inspect_task_manifest, prepare_joint_release
from psi.data.predictive_world.sonic_post_training import MEASUREMENT, episode_windows, extend_registries, fit_statistics, inspect_dataset
from test_sonic_post_training import dataset


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _task(root, task, offset, *, v3=True):
    raw = root / task / "raw"
    info = dataset(raw, v3=v3)
    for path in (raw / "data").rglob("*.parquet"):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["observation.state"] = [value + offset for value in row["observation.state"]]
            row["action"] = [value + offset for value in row["action"]]
        pq.write_table(pa.Table.from_pylist(rows), path)
    instruction = f"Perform {task}."
    if v3:
        pq.write_table(pa.Table.from_pylist([dict(task_index=0, task=instruction)]), raw / "meta/tasks.parquet")
    else:
        (raw / "meta/tasks.jsonl").write_text(json.dumps(dict(task_index=0, task=instruction)) + "\n")
    info, _, episodes, audit, statistics = inspect_dataset(raw, task=task)
    canonical = dict(layout_id="humanoid_upper_body_and_height_29_v1", slots=[f"slot{i}" for i in range(29)],
        raw_indices=[0] + [None] * 28, mean=[0.] * 29, std=[1.] * 29,
        normalization_space="per_source_registry_zscore_v1")
    source = dict(measurement_registry={"old": dict(width=64, world_action_eligible=True,
        fields=[dict(name="measured", start=0, stop=64, kind="measurement", unit="rad",
                     coordinate_frame="robot_native", time_key="timestamp")], canonical_body=canonical)},
        action_protocol_registry={}, state_schema_registry={})
    registries = extend_registries(source, info, statistics, task=task, audit_identity=canonical_sha256(audit))
    release = root / task / "reference"
    paths = {name: release / f"{name}.json" for name in registries}
    for name, value in registries.items():
        _write(paths[name], value)
    # The dummy legacy entry is never an active measurement source.
    schema = load_measurement_registry(paths["measurement_registry"])
    capabilities = load_capability_registry(paths["capability_registry"], measurement_schemas=schema,
        action_protocol_ids=set(registries["action_protocol_registry"]))
    data = PredictiveWorldDataConfig(state_history_count=1, observed_visual_mode="current_multiview",
        action_horizon=50, index_path=str(release / "index.jsonl"), feature_cache=str(release / "features"),
        action_protocol_registry=str(paths["action_protocol_registry"]), state_schema_registry=str(paths["state_schema_registry"]),
        measurement_registry=str(paths["measurement_registry"]), canonical_future_capabilities=str(paths["capability_registry"]))
    temporal = temporal_abi_from_data_config(data)
    rows = [row for ep in episodes for row in episode_windows(ep, task=task, instruction=instruction,
        temporal=temporal, capability=capabilities[task], measurement_schema=schema[MEASUREMENT])]
    publish_predictive_world_index(data.index_path, rows=rows,
        temporal_abi=temporal, provenance=dict(registry_binding={name + "_sha256": file_sha256(path) for name, path in paths.items()}))
    values = np.zeros((len(rows), 3, 16, 2048), dtype=np.float16)
    values[:, :, :, :] = offset
    values[:, 1] += 1
    shard = release / "features/shards/part.npy"
    shard.parent.mkdir(parents=True)
    np.save(shard, values)
    provenance = dict(encoder_id="fixture-vlm", processor_id="fixture-processor", image_preprocessing="aspect_v1",
        feature_layer="post_merger_content_area_overlap_4x4_v1", dtype="float16", feature_shape=[16, 2048], video_decoder="torchcodec")
    manifest = dict(format=PACKED_FEATURE_FORMAT, provenance=provenance, cache_identity=canonical_sha256(provenance),
        index_binding=predictive_world_index_binding(data.index_path), rows=len(rows), groups=3,
        feature_shape=[16, 2048], dtype="float16", shards=[dict(file="shards/part.npy", rank=0, start=0, stop=len(rows),
            sha256=file_sha256(shard), nbytes=shard.stat().st_size)])
    manifest["manifest_identity"] = canonical_sha256({key: manifest[key] for key in (
        "cache_identity", "index_binding", "rows", "groups", "shards")})
    _write(release / "features/feature_manifest.json", manifest)
    config = release / "config.json"
    _write(config, dict(data=asdict(data), model=dict(measurement_dim=29, qwen_model="fixture-vlm"), runtime={}))
    return dict(task=task, dataset=str(raw), config=config, values=values, manifest=release / "features/feature_manifest.json")


def _two_tasks(tmp_path, *, v3=True):
    entries = [_task(tmp_path, task, offset, v3=v3) for task, offset in (("task_a", 0), ("task_b", 10))]
    path = tmp_path / "datasets.json"
    _write(path, {entry["task"]: dict(dataset=entry["dataset"], expected_camera_count=1) for entry in entries})
    return path, entries


@pytest.mark.parametrize("v3", [False, True])
def test_shared_statistics_use_only_selected_raw_train_episode_rows(tmp_path, v3):
    manifest, entries = _two_tasks(tmp_path, v3=v3)
    _, _, episodes, audit, statistics = inspect_task_manifest(manifest, validation_fraction=.5,
        expected_tasks=("task_a", "task_b"), expected_episodes=2)
    states, actions = [], []
    for episode in episodes:
        if episode["split"] != "train":
            continue
        table = pq.read_table(episode["parquet"]).take(episode["row_indices"])
        states.extend(table["observation.state"].to_pylist())
        actions.extend(table["action"].to_pylist())
    assert statistics == fit_statistics(np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32))
    assert audit["statistics_train_frame_counts"] == {"task_a": 40, "task_b": 40}
    assert len(audit["train_episodes"]) == len(audit["validation_episodes"]) == 2
    assert set(audit["train_episodes"]).isdisjoint(audit["validation_episodes"])
    assert {episode["instruction"] for episode in episodes} == {"Perform task_a.", "Perform task_b."}
    task_means = [inspect_dataset(entry["dataset"], task=entry["task"])[4]["action"]["std"][0] for entry in entries]
    # Pooling raw distributions is different from averaging two small task ranges.
    assert statistics["action"]["std"][0] > np.mean(task_means) * 5
    if v3:
        assert any(episode["row_indices"][0] == 40 for episode in episodes)


def test_joint_release_preserves_packed_row_order_and_each_task_instruction(tmp_path):
    manifest, entries = _two_tasks(tmp_path)
    before = {entry["manifest"]: entry["manifest"].read_bytes() for entry in entries}
    release = prepare_joint_release(manifest, {entry["task"]: entry["config"] for entry in entries}, tmp_path / "joint",
        expected_tasks=("task_a", "task_b"), expected_episodes=2, validation_fraction=.5)
    assert release["source_sampling_exponent"] == 0 and release["task_sampling_probability"] == .5
    assert release["feature_content_verified"] is True
    assert release["train_episodes"] == release["validation_episodes"] == 2
    packed = PackedFeatureStore(tmp_path / "joint/feature_cache/feature_manifest.json")
    expected = np.concatenate([entry["values"] for entry in entries])
    np.testing.assert_array_equal(packed.get(0, 1), expected[0, 1])
    np.testing.assert_array_equal(packed.get(80, 1), expected[80, 1])
    rows = [json.loads(line) for line in (tmp_path / "joint/predictive_world_index.jsonl").read_text().splitlines()]
    assert [row["feature_row_index"] for row in rows] == list(range(160))
    assert {row["materialization"]["instruction"] for row in rows if row["source_id"] == "task_b"} == {"Perform task_b."}
    registry = json.loads((tmp_path / "joint/action_protocol_registry.json").read_text())
    statistics = json.loads((tmp_path / "joint/statistics.json").read_text())
    assert registry["sonic_semantic40_v31"] == statistics["action"]
    assert before == {path: path.read_bytes() for path in before}
    def reader(row, occurrence_index):
        return dict(source_id=row["source_id"], sample_id=row["sample_id"], occurrence=occurrence_index)
    def stream(rank, world_size, start=0):
        return PredictiveWorldOccurrenceDataset(tmp_path / "joint/predictive_world_index.jsonl", SimpleNamespace(read=reader),
            split="train", seed=0, rank=rank, world_size=world_size, start_occurrence=start,
            rank_batch_size=2, require_world_action=True, source_sampling_exponent=0.0)
    single = list(islice(stream(0, 1), 40))
    assert [item["source_id"] for item in single].count("task_a") == 20
    assert [item["source_id"] for item in single].count("task_b") == 20
    rank0, rank1 = list(islice(stream(0, 2), 20)), list(islice(stream(1, 2), 20))
    assert sorted(rank0 + rank1, key=lambda item: item["occurrence"]) == single
    assert list(islice(stream(0, 1, 20), 20)) == single[20:]
    # Reusing an identical prepared release is deterministic and never overwrites source objects.
    assert prepare_joint_release(manifest, {entry["task"]: entry["config"] for entry in entries}, tmp_path / "joint",
        expected_tasks=("task_a", "task_b"), expected_episodes=2, validation_fraction=.5) == release


@pytest.mark.parametrize("damage", ["feature_provenance", "cached_query", "tensor_content"])
def test_joint_release_rejects_unmatched_visual_cache(tmp_path, damage):
    manifest, entries = _two_tasks(tmp_path)
    entry = entries[1]
    feature = json.loads(entry["manifest"].read_text())
    if damage == "feature_provenance":
        feature["provenance"]["encoder_id"] = "another-vlm"
        feature["cache_identity"] = canonical_sha256(feature["provenance"])
        feature["manifest_identity"] = canonical_sha256({key: feature[key] for key in (
            "cache_identity", "index_binding", "rows", "groups", "shards")})
        _write(entry["manifest"], feature)
    elif damage == "tensor_content":
        path = entry["manifest"].parent / feature["shards"][0]["file"]
        with path.open("r+b") as stream:
            stream.seek(-2, 2)
            stream.write(b"\x01\x01")
    else:
        # This manifest is internally consistent but bound to another raw video query.
        config = json.loads(entry["config"].read_text())
        index = entry["config"].parent / "index.jsonl"
        records = [json.loads(line) for line in index.read_text().splitlines()]
        records[0]["future_visual_indices"][0] += 1
        old_meta = json.loads(index.with_suffix(".jsonl.meta.json").read_text())
        for path in (index, index.with_suffix(".jsonl.meta.json"), index.with_suffix(".jsonl.sqlite3")):
            path.unlink()
        publish_predictive_world_index(config["data"]["index_path"], rows=records,
            provenance=old_meta["provenance"], temporal_abi=old_meta["temporal_abi"])
        feature["index_binding"] = predictive_world_index_binding(index)
        feature["manifest_identity"] = canonical_sha256({key: feature[key] for key in (
            "cache_identity", "index_binding", "rows", "groups", "shards")})
        _write(entry["manifest"], feature)
        verify_predictive_world_feature_manifest(entry["manifest"])
    with pytest.raises(ValueError, match="provenance|visual query|content hash"):
        prepare_joint_release(manifest, {entry["task"]: entry["config"] for entry in entries}, tmp_path / "joint",
            expected_tasks=("task_a", "task_b"), expected_episodes=2)


def test_seven_task_contract_cannot_silently_accept_fewer_demonstrations(tmp_path):
    manifest, _ = _two_tasks(tmp_path)
    with pytest.raises(ValueError, match="expected distinct tasks"):
        inspect_task_manifest(manifest)
    with pytest.raises(ValueError, match="exactly 100 demonstrations"):
        inspect_task_manifest(manifest, expected_tasks=("task_a", "task_b"))
