import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from itertools import islice
from torch.utils.data import DataLoader

import psi.data.predictive_world.dataset as dataset_module
from psi.data.predictive_world.dataset import (
    PredictiveWorldOccurrenceDataset,
    assert_predictive_world_training_data_contract,
    validate_predictive_world_sample_feature_mapping,
)


class _OccurrenceReader:
    def read(self, row, *, occurrence_index):
        return occurrence_index


class _IdentityOccurrenceDataset(PredictiveWorldOccurrenceDataset):
    def __init__(self):
        self.reader = _OccurrenceReader()
        self.rank, self.world_size = 1, 2
        self.start_occurrence, self.rank_batch_size = 5, 3

    def _row(self, occurrence):
        return {'occurrence': occurrence}


@pytest.mark.parametrize('workers', [0, 2, 12])
def test_increasing_loader_workers_preserves_order_and_resume_cursor(workers):
    loader = DataLoader(_IdentityOccurrenceDataset(), batch_size=3, num_workers=workers,
                        **({'prefetch_factor': 1} if workers else {}))
    iterator = iter(loader)
    try:
        actual = [item for batch in islice(iterator, 17) for item in batch.tolist()]
        assert actual == list(range(5, 5 + 51 * 2, 2))
    finally:
        del iterator


def test_source_tempering_is_exact_and_deterministic_per_cycle():
    weights = PredictiveWorldOccurrenceDataset._integer_source_weights(
        (10_000, 2_500, 100), exponent=0.5, maximum_weight=100
    )
    assert weights == (100, 50, 10)
    schedule, offsets = PredictiveWorldOccurrenceDataset._source_schedule(weights)
    assert tuple(schedule.count(index) for index in range(3)) == weights
    assert sorted(offset for source, offset in zip(schedule, offsets) if source == 1) == list(
        range(weights[1])
    )


def _row(tmp_path: Path):
    head = str((tmp_path / "head.mp4").resolve())
    wrist = str((tmp_path / "wrist.mp4").resolve())
    return {
        "sample_id": "sample",
        "future_visual_indices": [10, 20],
        "future_visual_valid": [True, False],
        "future_visual_actual_times": [0.2, 0.4],
        "materialization": {"video_paths": {"head": head, "wrist": wrist}},
    }


def test_feature_mapping_requires_every_valid_camera_and_no_invalid_group(tmp_path):
    row = _row(tmp_path)
    head = row["materialization"]["video_paths"]["head"]
    wrist = row["materialization"]["video_paths"]["wrist"]
    records = [
        {
            "sample_id": "sample", "group": 0, "camera_id": "head",
            "actual_time": 0.2, "frame_id": f"{head}:10",
        },
        {
            "sample_id": "sample", "group": 0, "camera_id": "wrist",
            "actual_time": 0.2, "frame_id": f"{wrist}:10",
        },
    ]
    assert len(validate_predictive_world_sample_feature_mapping(row, records)) == 2
    with pytest.raises(ValueError, match="missing=.*wrist"):
        validate_predictive_world_sample_feature_mapping(row, records[:1])
    with pytest.raises(ValueError, match="extra"):
        validate_predictive_world_sample_feature_mapping(
            row,
            records + [{
                "sample_id": "sample", "group": 1, "camera_id": "head",
                "actual_time": 0.4, "frame_id": f"{head}:20",
            }],
        )


def test_feature_mapping_binds_exact_frame_and_relative_time(tmp_path):
    row = _row(tmp_path)
    head = row["materialization"]["video_paths"]["head"]
    wrist = row["materialization"]["video_paths"]["wrist"]
    records = [
        {
            "sample_id": "sample", "group": 0, "camera_id": "head",
            "actual_time": 0.21, "frame_id": f"{head}:10",
        },
        {
            "sample_id": "sample", "group": 0, "camera_id": "wrist",
            "actual_time": 0.2, "frame_id": f"{wrist}:10",
        },
    ]
    with pytest.raises(ValueError, match="frame/time"):
        validate_predictive_world_sample_feature_mapping(row, records)


def test_training_data_contract_returns_all_validated_artifacts(tmp_path, monkeypatch):
    paths = {
        name: tmp_path / name
        for name in ("measurements.json", "capabilities.json", "actions.json", "states.json")
    }
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    paths["actions.json"].write_text(json.dumps({"native": {}}), encoding="utf-8")
    index = tmp_path / "index.jsonl"
    index.write_text("", encoding="utf-8")
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    registry_binding = {
        "measurement_registry_sha256": dataset_module.file_sha256(paths["measurements.json"]),
        "capability_registry_sha256": dataset_module.file_sha256(paths["capabilities.json"]),
        "action_protocol_registry_sha256": dataset_module.file_sha256(paths["actions.json"]),
        "state_schema_registry_sha256": dataset_module.file_sha256(paths["states.json"]),
    }
    index_meta = {"provenance": {"registry_binding": registry_binding}}
    measurements = {"measurement": object()}
    capabilities = {"source": object()}
    binding = {"semantic_identity": "bound"}
    monkeypatch.setattr(
        dataset_module, "assert_predictive_world_index_temporal_abi",
        lambda *_args, **_kwargs: index_meta,
    )
    monkeypatch.setattr(dataset_module, "load_measurement_registry", lambda _path: measurements)
    monkeypatch.setattr(
        dataset_module, "load_capability_registry", lambda *_args, **_kwargs: capabilities,
    )
    monkeypatch.setattr(dataset_module, "assert_canonical_body_target_width", lambda *_a, **_k: None)
    monkeypatch.setattr(
        dataset_module, "verify_predictive_world_feature_manifest",
        lambda _path: {"index_binding": binding},
    )
    monkeypatch.setattr(dataset_module, "predictive_world_index_binding", lambda _path: binding)
    config = SimpleNamespace(
        index_path=str(index),
        feature_cache=str(manifest),
        measurement_registry=str(paths["measurements.json"]),
        canonical_future_capabilities=str(paths["capabilities.json"]),
        action_protocol_registry=str(paths["actions.json"]),
        state_schema_registry=str(paths["states.json"]),
    )

    artifacts = assert_predictive_world_training_data_contract(config, measurement_dim=29)

    assert artifacts.measurement_schemas is measurements
    assert artifacts.capabilities is capabilities
    assert artifacts.index_meta is index_meta
    assert artifacts.feature_manifest == manifest.resolve()
