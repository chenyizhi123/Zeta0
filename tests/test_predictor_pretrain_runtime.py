import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from psi.data.predictor_pretrain.registry import canonical_sha256, write_manifest_rows
from psi.data.predictor_pretrain.runtime import (
    AnchorIndex, FixedOccurrenceDataset, LeRobotAnchorReader, LeRobotRuntimeSource,
    _clamp_opencv_endpoint_frame, build_anchor_index, decode_video_frames_opencv,
)
from psi.data.predictor_pretrain.lerobot_source import build_lerobot_anchor_rows


def test_opencv_exact_duration_maps_only_one_past_end_to_last_frame():
    assert _clamp_opencv_endpoint_frame(513, 513) == 512
    assert _clamp_opencv_endpoint_frame(512, 513) == 512
    assert _clamp_opencv_endpoint_frame(514, 513) == 514


def test_opencv_decoder_caps_threads_per_dataloader_worker(tmp_path):
    import cv2

    cv2.setNumThreads(20)
    with pytest.raises(RuntimeError, match="cannot open video"):
        decode_video_frames_opencv(tmp_path / "missing.mp4", [0.0], 0.01)
    assert cv2.getNumThreads() == 1


def test_anchor_index_is_bound_to_canonical_manifest_and_rejects_val(tmp_path):
    rows = [
        {"sample_id": "a", "source_id": "s", "split": "PT_train"},
        {"sample_id": "b", "source_id": "s", "split": "PT_val"},
    ]
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, rows)
    normalized = [json.loads(line) for line in manifest.read_text().splitlines()]
    index_path = build_anchor_index(manifest, tmp_path / "anchors.sqlite")
    index = AnchorIndex(index_path, expected_manifest_sha256=canonical_sha256(normalized))
    assert index.get("a")["split"] == "PT_train"
    with pytest.raises(ValueError, match="non-PT_train"):
        index.get("b")
    with pytest.raises(ValueError, match="different frozen manifest"):
        AnchorIndex(index_path, expected_manifest_sha256="0" * 64)


def test_lerobot_reader_routes_dynamic_schema_and_crops_stereo(tmp_path):
    table = pa.table({
        "arm": [[1.0, 2.0], [3.0, 4.0]],
        "leg": [[5.0], [6.0]],
    })
    data = tmp_path / "episode.parquet"
    pq.write_table(table, data)

    def decoder(path, timestamps, tolerance, backend):
        assert timestamps == [0.5]
        return torch.arange(3 * 4 * 8).reshape(1, 3, 4, 8)

    source = LeRobotRuntimeSource(
        "HE", None, state_columns=("arm", "leg"),
        state_schema_field="robot_type", image_crop="left_half",
    )
    row = {
        "source_id": "HE", "robot_type": "g1", "data_path": str(data),
        "video_path": str(tmp_path / "fake.mp4"), "current_video_timestamp": 0.5,
        "nominal_fps": 30, "state_row_indices": [0, 1],
        "state_history_mask": [True, True], "instruction": "walk",
    }
    result = LeRobotAnchorReader({"HE": source}, decoder=decoder).read(row)
    assert result["state_schema_id"] == "g1"
    np.testing.assert_array_equal(result["raw_states"], [[1, 2, 5], [3, 4, 6]])
    assert result["image"].shape == (3, 4, 4)
    assert result["state_present"] is True


def test_lerobot_reader_prefers_dedicated_video_row_identity_over_drifted_timestamp(tmp_path):
    pq.write_table(pa.table({"state": [[1.0]]}), tmp_path / "episode.parquet")

    def decoder(_path, timestamps, _tolerance, _backend):
        assert timestamps == [15 / 30]
        return torch.zeros(1, 3, 4, 4)

    source = LeRobotRuntimeSource(
        "HE", "schema", state_columns=("state",), video_backend="opencv"
    )
    row = {
        "source_id": "HE", "data_path": str(tmp_path / "episode.parquet"),
        "video_path": str(tmp_path / "episode.mp4"),
        "current_video_timestamp": 0.57, "current_row_index": 15,
        "nominal_fps": 30, "shared_data_file": False,
        "state_row_indices": [0], "state_history_mask": [True], "instruction": "walk",
    }
    assert LeRobotAnchorReader({"HE": source}, decoder=decoder).read(row)["image"].shape == (3, 4, 4)


def test_v3_freezes_physical_action_grid_and_routes_native_protocol(tmp_path):
    timestamps = np.arange(40, dtype=np.float64) / 30.0
    actions = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    data = tmp_path / "episode.parquet"
    (tmp_path / "video.mp4").touch()
    pq.write_table(pa.table({
        "timestamp": timestamps,
        "state": pa.array([[float(index)] for index in range(40)]),
        "action": pa.array(actions.tolist()),
    }), data)
    episode = {
        "sample_id": "source@rev:episode:0", "source_id": "source", "revision": "rev",
        "episode_id": "0", "group_id": "source|episode=0", "grouping_method": "episode_fallback",
        "lineage_ids": ["source@rev:episode:0"], "content_hashes": [],
        "paths": [str(data), str(tmp_path / "video.mp4")], "data_path": str(data),
        "video_path": str(tmp_path / "video.mp4"), "instruction": "move", "robot_type": "g1",
        "nominal_fps": 30.0, "primary_image_key": "head", "dataset_from_index": None,
        "dataset_to_index": None, "dataset_stop_exclusive": None, "shared_data_file": False,
        "data_file_global_start": None, "video_from_timestamp": None, "video_to_timestamp": None,
        "format_version": "predictor_pretrain_episode_v3", "input_image_keys": ["head"],
        "input_views": [{"image_key": "head", "camera_role": "primary", "video_path": str(tmp_path / "video.mp4"),
                         "available": True, "video_from_timestamp": None, "video_to_timestamp": None}],
        "target_image_key": "head", "split": "PT_train",
    }
    rows = build_lerobot_anchor_rows(
        [episode], protocol_version="v3", candidate_phases=1,
        anchor_hz=2, future_seconds=0.6, action_steps=30,
    )
    row = rows[0]
    assert row["format_version"] == "predictor_pretrain_anchor_v3"
    assert len(row["action_row_indices"]) == 30
    # 30-Hz source sampled on a 50-Hz grid necessarily repeats exact rows.
    assert len(set(row["action_row_indices"])) < 30
    source = LeRobotRuntimeSource(
        "source", "state1", state_columns=("state",), action_columns=("action",),
        action_protocol="native3",
    )
    reader = LeRobotAnchorReader(
        {"source": source}, decoder=lambda *_args: torch.zeros(1, 3, 4, 4)
    )
    materialized = reader.read(row)
    assert materialized["raw_actions"].shape == (30, 3)
    assert materialized["action_protocol_id"] == "native3"
    assert materialized["action_time_mask"].all()


def test_v3_causally_holds_explicit_action_clock_and_masks_uncovered_boundary(tmp_path):
    timestamps = np.arange(40, dtype=np.float64) / 30.0
    # The first command arrives after the first image; repeats model a lower
    # rate WBC stream causally held by the converter.
    action_timestamps = np.maximum(
        0.05, np.floor(np.maximum(timestamps - 0.05, 0.0) * 20.0) / 20.0 + 0.05
    )
    data = tmp_path / "episode.parquet"
    video = tmp_path / "video.mp4"
    video.touch()
    pq.write_table(pa.table({
        "timestamp": timestamps,
        "timestamp.wbc": action_timestamps,
        "state": pa.array([[float(index)] for index in range(40)]),
        "action": pa.array([[float(index)] for index in range(40)]),
    }), data)
    episode = {
        "sample_id": "source@rev:episode:0", "source_id": "source", "revision": "rev",
        "episode_id": "0", "group_id": "source|episode=0", "grouping_method": "episode_fallback",
        "lineage_ids": ["source@rev:episode:0"], "content_hashes": [],
        "paths": [str(data), str(video)], "data_path": str(data), "video_path": str(video),
        "instruction": "move", "robot_type": "g1", "nominal_fps": 30.0,
        "primary_image_key": "head", "dataset_from_index": None,
        "dataset_to_index": None, "dataset_stop_exclusive": None, "shared_data_file": False,
        "data_file_global_start": None, "video_from_timestamp": None,
        "format_version": "predictor_pretrain_episode_v3", "input_image_keys": ["head"],
        "input_views": [{"image_key": "head", "camera_role": "primary",
                         "video_path": str(video), "available": True,
                         "video_from_timestamp": None, "video_to_timestamp": None}],
        "target_image_key": "head", "split": "PT_train",
        "action_timestamp_column": "timestamp.wbc",
    }
    row = build_lerobot_anchor_rows(
        [episode], protocol_version="v3", candidate_phases=1,
        anchor_hz=2, future_seconds=0.6, action_steps=30,
    )[0]
    assert row["action_time_mask"][:3] == [False, False, False]
    assert all(row["action_time_mask"][3:])
    assert row["action_row_indices"][:3] == [0, 0, 0]
    valid = np.asarray(row["action_time_mask"], dtype=bool)
    requested = np.asarray(row["action_requested_timestamps"])[valid]
    actual = np.asarray(row["action_timestamps"])[valid]
    assert np.all(actual <= requested + 1e-9)


class _FixedStreamReader:
    def read(self, row):
        return {
            "raw_states": np.zeros((1, 0), dtype=np.float32),
            "state_history_mask": np.zeros(1, dtype=bool),
            "state_present": False,
        }


class _FixedStreamCache:
    def get(self, sample_id):
        return np.zeros((16, 2048), dtype=np.float16), 0.6


def _fixed_stream_dataset(tmp_path, *, start_occurrence):
    anchor_rows = [
        {
            "sample_id": f"s{index}", "source_id": "source", "split": "PT_train",
            "target_timestamp": 0.6,
        }
        for index in range(16)
    ]
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, anchor_rows)
    normalized = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest_sha = canonical_sha256(normalized)
    anchor_index = build_anchor_index(manifest, tmp_path / "anchors.sqlite")
    stream = tmp_path / "stream.jsonl"
    stream.write_text("".join(
        json.dumps({
            "occurrence_index": index,
            "source_id": "source",
            "sample_id": f"s{index}",
            "drop_state": False,
        }) + "\n"
        for index in range(16)
    ))
    meta = tmp_path / "stream.meta.json"
    meta.write_text(json.dumps({"dataset_manifest_sha256": manifest_sha}))
    return FixedOccurrenceDataset(
        stream, meta, anchor_index, _FixedStreamReader(), {"source": _FixedStreamCache()},
        expected_dataset_manifest_sha256=manifest_sha,
        rank=0, world_size=2, start_occurrence=start_occurrence,
        rank_batch_size=2, verify_stream_hash=False,
    )


@pytest.mark.parametrize(
    ("start_occurrence", "expected"),
    [(0, [0, 2, 4, 6, 8, 10, 12, 14]), (8, [8, 10, 12, 14])],
)
def test_fixed_stream_multiworker_preserves_order_and_resume(
    tmp_path, start_occurrence, expected
):
    loader = torch.utils.data.DataLoader(
        _fixed_stream_dataset(tmp_path, start_occurrence=start_occurrence),
        batch_size=2,
        num_workers=2,
    )
    observed = []
    for batch in loader:
        observed.extend(batch["occurrence_index"].tolist())
    assert observed == expected


def test_compact_v2_stream_multiworker_random_access_resume(tmp_path):
    anchor_rows = [
        {
            "sample_id": f"s{index}", "source_id": "source", "split": "PT_train",
            "target_timestamp": 0.6,
        }
        for index in range(16)
    ]
    manifest = tmp_path / "anchors.jsonl"
    write_manifest_rows(manifest, anchor_rows)
    normalized = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest_sha = canonical_sha256(normalized)
    anchor_index = build_anchor_index(manifest, tmp_path / "anchors.sqlite")
    stream = tmp_path / "stream.jsonl"
    compact_spec = {
        "format_version": "predictor_pretrain_compact_sample_stream_v2",
        "sampling_algorithm": "weighted_cycle_affine_permutation_v1",
        "dataset_manifest_sha256": manifest_sha,
        "source_weights": {"source": 1},
        "source_capacities": {"source": 16},
        "total_updates": 8,
        "global_batch_size": 4,
        "seed": 101,
        "state_dropout_probability": 0.0,
        "num_occurrences": 32,
    }
    stream.write_text(json.dumps(compact_spec) + "\n")
    meta = tmp_path / "stream.meta.json"
    meta.write_text(json.dumps({
        "dataset_manifest_sha256": manifest_sha,
        "stream_encoding": "algorithmic_v2",
        "sampling_algorithm": "weighted_cycle_affine_permutation_v1",
        "source_weights": {"source": 1},
        "source_capacities": {"source": 16},
        "total_updates": 8,
        "seed": 101,
        "state_dropout_probability": 0.0,
        "num_occurrences": 32,
        "global_batch_size": 4,
    }))
    dataset = FixedOccurrenceDataset(
        stream, meta, anchor_index, _FixedStreamReader(),
        {"source": _FixedStreamCache()}, expected_dataset_manifest_sha256=manifest_sha,
        rank=0, world_size=2, start_occurrence=8, rank_batch_size=2,
        verify_stream_hash=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)
    observed = []
    for batch in loader:
        observed.extend(batch["occurrence_index"].tolist())
    assert observed == list(range(8, 32, 2))
