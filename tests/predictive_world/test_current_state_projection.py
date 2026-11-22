import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_index import _capability, _episode, _provenance, _schema, _temporal

from psi.data.predictive_world.cache import (
    PACKED_FEATURE_FORMAT,
    canonical_sha256,
    file_sha256,
    verify_predictive_world_feature_manifest,
)
from psi.data.predictive_world.current_state import (
    STATE_FIELDS,
    project_current_state_row,
    publish_current_state_projection,
)
from psi.data.predictive_world.dataset import PredictiveWorldRowReader
from psi.data.predictive_world.index import (
    build_episode_windows,
    predictive_world_index_binding,
    publish_predictive_world_index,
    verify_predictive_world_index,
)


def _fixture(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    vectors = np.stack((np.arange(51), -np.arange(51)), axis=1).astype(np.float32)
    pq.write_table(pa.table({"observation.state": vectors.tolist(), "action": vectors.tolist()}),
                   root / "episode.parquet")
    temporal = _temporal(state_history_count=21)
    rows = []
    for episode_id, split, anchors in (("episode", "train", [0.0, .4]),
                                        ("heldout", "validation", [.4])):
        episode = {**_episode(), "episode_id": episode_id, "split": split,
                   "anchor_times": anchors}
        episode["materialization"].update(
            data_path=str(root / "episode.parquet"), state_columns=["observation.state"],
            action_columns=["action"], state_schema_id="state",
            action_normalization_space="raw", measurement_indices=[0, 1],
        )
        rows.extend(build_episode_windows(
            episode, history_visual_offsets_s=temporal["history_visual_offsets_s"],
            state_offsets_s=temporal["state_offsets_s"],
            future_group_ends_s=temporal["future_group_ends_s"],
            action_offsets_s=temporal["action_offsets_s"],
            future_measurement_offsets_s=temporal["future_measurement_offsets_s"],
            visual_max_age_s=.021, state_max_age_s=.021, action_max_age_s=.021,
            capability=_capability(), measurement_schema=_schema(),
        ))
    index = root / "index.jsonl"
    publish_predictive_world_index(index, rows=rows, provenance=_provenance(), temporal_abi=temporal)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    cache = root / "feature_cache"
    (cache / "shards").mkdir(parents=True)
    array = np.empty((len(rows), 3, 16, 2048), dtype=np.float16)
    for ordinal in range(len(rows)):
        for group in range(3):
            array[ordinal, group].fill(10 * ordinal + group)
    shard = cache / "shards/part.npy"
    np.save(shard, array)
    provenance = {"encoder_id": "test", "processor_id": "test", "dtype": "float16",
                  "image_preprocessing": "aspect_v1", "feature_shape": [16, 2048]}
    manifest = dict(
        format=PACKED_FEATURE_FORMAT, provenance=provenance,
        cache_identity=canonical_sha256(provenance), index_binding=predictive_world_index_binding(index),
        rows=len(rows), groups=3, feature_shape=[16, 2048], dtype="float16",
        shards=[dict(file="shards/part.npy", rank=0, start=0, stop=len(rows),
                     nbytes=shard.stat().st_size, sha256=file_sha256(shard))],
    )
    _write_manifest(cache / "feature_manifest.json", manifest)
    return index, rows, cache


def _write_manifest(path, manifest):
    manifest["manifest_identity"] = canonical_sha256({key: manifest[key] for key in (
        "cache_identity", "index_binding", "rows", "groups", "shards")})
    path.write_text(json.dumps(manifest))


def _prepare(source, output):
    index, _, cache = source
    return publish_current_state_projection(
        index, output / "index.jsonl", source_feature_manifest=cache,
        output_feature_cache=output / "feature_cache",
    )


def test_projection_changes_only_observed_state_and_rebinds_immutable_artifacts(tmp_path):
    source = _fixture(tmp_path)
    index, old_rows, cache = source
    source_hashes = {path: file_sha256(path) for path in (
        index, index.with_suffix(".jsonl.meta.json"), index.with_suffix(".jsonl.sqlite3"),
        cache / "feature_manifest.json", cache / "shards/part.npy",
    )}
    report = _prepare(source, tmp_path / "target")
    new_index = tmp_path / "target/index.jsonl"
    meta = verify_predictive_world_index(new_index)
    original = verify_predictive_world_index(index)
    assert meta["temporal_abi"] == {
        **original["temporal_abi"], "state_history_count": 1, "state_offsets_s": [0.0],
    }
    assert meta["provenance"]["registry_binding"] == original["provenance"]["registry_binding"]
    assert meta["semantic_identity"] != original["semantic_identity"]
    new_rows = [json.loads(line) for line in new_index.read_text().splitlines()]
    for old, new in zip(old_rows, new_rows, strict=True):
        assert canonical_sha256({k: v for k, v in old.items() if k not in STATE_FIELDS}) == canonical_sha256(
            {k: v for k, v in new.items() if k not in STATE_FIELDS})
        assert all(new[field] == old[field][-1:] for field in STATE_FIELDS)
    manifest = verify_predictive_world_feature_manifest(
        tmp_path / "target/feature_cache/feature_manifest.json", verify_shard_hashes=True)
    old_manifest = verify_predictive_world_feature_manifest(cache / "feature_manifest.json")
    assert manifest["index_binding"] == predictive_world_index_binding(new_index)
    assert manifest["provenance"] == old_manifest["provenance"]
    assert manifest["cache_identity"] == old_manifest["cache_identity"]
    assert manifest["shards"] == old_manifest["shards"]
    assert (tmp_path / "target/feature_cache/shards/part.npy").is_symlink()
    assert report["split_counts"] == {"train": 2, "validation": 1}
    assert report["feature_bytes_copied"] == 0
    assert all(file_sha256(path) == digest for path, digest in source_hashes.items())
    assert _prepare(source, tmp_path / "target") == report


def test_reader_uses_one_original_state_and_keeps_future_body_targets(tmp_path, monkeypatch):
    source = _fixture(tmp_path)
    _prepare(source, tmp_path / "target")
    monkeypatch.setattr(PredictiveWorldRowReader, "_qwen", lambda *args: {})

    def read(index, manifest):
        reader = PredictiveWorldRowReader(
            None, manifest, index_path=index, capabilities={"source": _capability()},
            measurement_schemas={"measurement": _schema()}, observed_visual_mode="current_multiview",
        )
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        return [reader.read(row, occurrence_index=ordinal) for ordinal, row in enumerate(rows)]

    old = read(source[0], source[2] / "feature_manifest.json")
    new = read(tmp_path / "target/index.jsonl", tmp_path / "target/feature_cache/feature_manifest.json")
    for before, after in zip(old, new, strict=True):
        assert before["raw_states"].shape == (21, 2)
        assert after["raw_states"].shape == (1, 2)
        np.testing.assert_array_equal(after["raw_states"], before["raw_states"][-1:])
        np.testing.assert_array_equal(after["state_valid"], before["state_valid"][-1:])
        for field in ("current_measurements", "current_measurement_valid", "future_measurements",
                      "future_measurement_valid", "future_measurement_times", "actions", "action_valid"):
            np.testing.assert_array_equal(after[field], before[field])
        assert after["future_measurements"].shape == (30, 2)
        assert after["future_measurement_valid"].all()
        for old_group, new_group in zip(before["future_patch_fields"], after["future_patch_fields"], strict=True):
            for old_field, new_field in zip(old_group, new_group, strict=True):
                torch.testing.assert_close(old_field["values"], new_field["values"], rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("state_indices", [0]), ("state_valid", [True]), ("state_actual_times", [0.0]),
    ("state_indices", [0] * 20 + [-1]), ("state_valid", [True] * 20 + [1]),
    ("state_actual_times", [0.] * 20 + [.02]), ("state_actual_times", [0.] * 20 + [float("nan")]),
])
def test_malformed_source_state_fails_closed(field, value):
    row = {"state_indices": [0] * 21, "state_valid": [True] * 21, "state_actual_times": [0.] * 21}
    row[field] = value
    with pytest.raises(ValueError, match="current-state projection source"):
        project_current_state_row(row, source_state_history_count=21)


def test_unavailable_current_state_keeps_mask_instead_of_fabricating_observation():
    row = {"state_indices": [2] * 21, "state_valid": [False] * 21,
           "state_actual_times": [float("nan")] * 21, "future_measurement_valid": [True] * 30}
    result = project_current_state_row(row, source_state_history_count=21)
    assert result["state_valid"] == [False]
    assert np.isnan(result["state_actual_times"][0])
    assert result["future_measurement_valid"] == [True] * 30


def test_source_index_tampering_is_rejected(tmp_path):
    source = _fixture(tmp_path)
    source[0].write_text(source[0].read_text() + "\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        _prepare(source, tmp_path / "target")
    assert not (tmp_path / "target/index.jsonl").exists()


def test_unrelated_feature_binding_and_partial_feature_request_are_rejected(tmp_path):
    source = _fixture(tmp_path)
    manifest_path = source[2] / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["index_binding"]["index_sha256"] = "f" * 64
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="different index"):
        _prepare(source, tmp_path / "target")
    with pytest.raises(ValueError, match="requires both"):
        publish_current_state_projection(source[0], tmp_path / "target/index.jsonl",
                                         source_feature_manifest=manifest_path)
    with pytest.raises(ValueError, match="new index"):
        publish_current_state_projection(source[0], source[0])


def test_conflicting_existing_target_is_not_overwritten(tmp_path):
    source = _fixture(tmp_path)
    _prepare(source, tmp_path / "target")
    manifest_path = tmp_path / "target/feature_cache/feature_manifest.json"
    modified = copy.deepcopy(json.loads(manifest_path.read_text()))
    modified["dtype"] = "float32"
    manifest_path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="immutable current-state artifact differs"):
        _prepare(source, tmp_path / "target")
    assert json.loads(manifest_path.read_text())["dtype"] == "float32"
