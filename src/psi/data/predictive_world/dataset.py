"""LeRobot episode materialization for immutable predictive-world index rows."""

from __future__ import annotations

import json
import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from psi.data.lerobot.action_aligned_cache import preprocess_endpoint_vlm_image
from psi.data.predictive_world.cache import (
    PACKED_FEATURE_FORMAT,
    PackedFeatureStore,
    RaggedFeatureStore,
    canonical_sha256,
    file_sha256,
    verify_predictive_world_feature_manifest,
)
from psi.data.predictive_world.index import (
    INDEX_BINDING_FORMAT,
    assert_predictive_world_index_temporal_abi,
    predictive_world_index_binding,
    predictive_world_lookup_path,
    verify_predictive_world_index,
)
from psi.data.predictive_world.measurement_schema import (
    MeasurementSchema,
    SourceCapability,
    assert_canonical_body_target_width,
    load_capability_registry,
    load_measurement_registry,
    validate_predictive_world_row_capability,
)


def _vector(value: Any) -> np.ndarray:
    if hasattr(value, "as_py"):
        value = value.as_py()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _read_vectors(table: Any, indices: Iterable[int], columns: Iterable[str]) -> np.ndarray:
    """Gather numeric labels in Arrow, preserving indexed order and duplicates.

    Dense scalar/list columns avoid converting every element through Python.
    Unusual nested/ragged/null-list columns retain the scalar reader semantics.
    """
    import pyarrow as pa

    positions = np.asarray([int(index) for index in indices], dtype=np.int64)
    columns = tuple(columns)
    if not len(positions):
        raise ValueError("cannot materialize an empty vector selection")
    positions = np.where(positions < 0, positions + len(table), positions)
    if (positions < 0).any() or (positions >= len(table)).any():
        raise IndexError("numeric source row is out of range")
    selected = table.select(columns).take(pa.array(positions))
    vectors = []
    for name in columns:
        values = selected[name].combine_chunks()
        kind = values.type
        if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
            if values.null_count:
                break
            if not pa.types.is_fixed_size_list(kind):
                lengths = values.value_lengths().to_numpy(zero_copy_only=False)
                if not np.all(lengths == lengths[0]):
                    break
            values = values.flatten()
            kind = values.type
        if not (pa.types.is_floating(kind) or pa.types.is_integer(kind) or pa.types.is_boolean(kind)):
            break
        numeric = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
        vectors.append(numeric.reshape(len(positions), numeric.size // len(positions)))
    else:
        return np.concatenate(vectors, axis=1)
    return np.stack([
        np.concatenate([_vector(table[column][int(index)]) for column in columns])
        for index in positions
    ]).astype(np.float32, copy=False)


@dataclass(frozen=True)
class ValidatedPredictiveWorldDataArtifacts:
    measurement_schemas: Mapping[str, MeasurementSchema]
    capabilities: Mapping[str, SourceCapability]
    index_meta: Mapping[str, Any]
    feature_manifest: Path


def _feature_manifest_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    return resolved / "feature_manifest.json" if resolved.is_dir() else resolved


def assert_predictive_world_training_data_contract(
    data: Any,
    *,
    measurement_dim: int,
    verify_content_hashes: bool = True,
) -> ValidatedPredictiveWorldDataArtifacts:
    """Validate every config-to-index/cache binding before constructing a DataLoader."""

    index_meta = assert_predictive_world_index_temporal_abi(
        data.index_path, data, verify_content_hashes=verify_content_hashes
    )
    alignment = index_meta.get("provenance", {}).get("time_alignment", {}).get("future_component_alignment")
    if alignment != getattr(data, "future_component_alignment", None):
        raise ValueError("configured future component alignment disagrees with the audited index")
    registry_paths = {
        "measurement_registry_sha256": Path(data.measurement_registry).expanduser().resolve(),
        "capability_registry_sha256": Path(
            data.canonical_future_capabilities
        ).expanduser().resolve(),
        "action_protocol_registry_sha256": Path(
            data.action_protocol_registry
        ).expanduser().resolve(),
        "state_schema_registry_sha256": Path(data.state_schema_registry).expanduser().resolve(),
    }
    if any(not path.is_file() for path in registry_paths.values()):
        missing = sorted(str(path) for path in registry_paths.values() if not path.is_file())
        raise ValueError(f"predictive-world registry files are missing: {missing}")
    measurements = load_measurement_registry(registry_paths["measurement_registry_sha256"])
    protocols = json.loads(
        registry_paths["action_protocol_registry_sha256"].read_text(encoding="utf-8")
    )
    if not isinstance(protocols, dict) or not protocols:
        raise ValueError("action protocol registry must be a non-empty JSON object")
    capabilities = load_capability_registry(
        registry_paths["capability_registry_sha256"],
        measurement_schemas=measurements,
        action_protocol_ids=set(map(str, protocols)),
    )
    assert_canonical_body_target_width(
        capabilities, measurements, expected_width=measurement_dim
    )
    indexed_binding = index_meta.get("provenance", {}).get("registry_binding")
    if not isinstance(indexed_binding, Mapping):
        raise ValueError("predictive-world index has no immutable registry binding")
    for name, path in registry_paths.items():
        if indexed_binding.get(name) != file_sha256(path):
            raise ValueError(f"predictive-world index registry binding mismatch: {name}")
    manifest_path = _feature_manifest_path(data.feature_cache)
    manifest = verify_predictive_world_feature_manifest(manifest_path)
    # Reuse the metadata that was just validated instead of hashing a very
    # large index and lookup for a second time in the same startup path.
    if all(
        key in index_meta
        for key in ("semantic_identity", "index_sha256", "lookup_sha256", "temporal_abi")
    ):
        index_binding = {
            "format": INDEX_BINDING_FORMAT,
            "semantic_identity": str(index_meta["semantic_identity"]),
            "index_sha256": str(index_meta["index_sha256"]),
            "lookup_sha256": str(index_meta["lookup_sha256"]),
            "temporal_abi_sha256": canonical_sha256(index_meta["temporal_abi"]),
        }
    else:
        # Retained for small injected test doubles and legacy callers.
        index_binding = predictive_world_index_binding(data.index_path)
    if manifest.get("index_binding") != index_binding:
        raise ValueError("predictive-world feature manifest is bound to a different index")
    return ValidatedPredictiveWorldDataArtifacts(
        measurement_schemas=measurements,
        capabilities=capabilities,
        index_meta=index_meta,
        feature_manifest=manifest_path,
    )


def validate_predictive_world_sample_feature_mapping(
    row: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Require one cached frame for every valid index group/camera and no extras."""

    materialization = row.get("materialization")
    if not isinstance(materialization, Mapping):
        raise ValueError("index row lacks audited materialization metadata")
    videos = materialization.get("video_paths")
    if not isinstance(videos, Mapping) or not videos:
        raise ValueError("materialization needs non-empty camera video_paths")
    indices = row.get("future_visual_indices")
    valid = row.get("future_visual_valid")
    times = row.get("future_visual_actual_times")
    if not isinstance(indices, list) or not isinstance(valid, list) or not isinstance(times, list):
        raise ValueError("index row future visual mappings must be lists")
    if not (len(indices) == len(valid) == len(times)):
        raise ValueError("index row future visual mappings disagree")
    capability = row.get("source_capability", {})
    future_visual_keys = capability.get("future_visual_keys", tuple(videos))
    if not isinstance(future_visual_keys, (list, tuple)):
        raise ValueError("source capability future_visual_keys must be an array")
    expected = {
        (group, str(camera_id)): (
            float(times[group]),
            f"{Path(video_path).expanduser().resolve()}:{int(indices[group])}",
        )
        for group, present in enumerate(valid)
        if bool(present)
        for camera_id in sorted(map(str, future_visual_keys))
        for video_path in (videos[camera_id],)
    }
    record_list = list(records)
    actual = {
        (int(record["group"]), str(record["camera_id"])): record
        for record in record_list
    }
    if len(actual) != len(record_list) or set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "feature manifest is incomplete for index sample "
            f"{row.get('sample_id')}: missing={missing[:8]}, extra={extra[:8]}"
        )
    for feature_key, record in actual.items():
        expected_time, expected_frame_id = expected[feature_key]
        if (
            not math.isclose(
                float(record["actual_time"]), expected_time, rel_tol=0.0, abs_tol=1e-12
            )
            or str(record.get("frame_id", "")) != expected_frame_id
        ):
            raise ValueError("feature manifest mapping disagrees with index frame/time")
    return [actual[key] for key in sorted(actual)]


class PredictiveWorldRowReader:
    """Read only indices frozen into the new manifest; never search at runtime."""

    def __init__(
        self,
        processor: Any,
        feature_manifest: str | Path,
        *,
        index_path: str | Path,
        capabilities: Mapping[str, SourceCapability],
        measurement_schemas: Mapping[str, MeasurementSchema],
        index_already_verified: bool = False,
        parquet_cache_capacity: int = 2,
        video_cache_capacity: int = 8,
        observed_visual_mode: str = "history",
        current_visual_pooling: str = "canvas_v1",
    ) -> None:
        if parquet_cache_capacity <= 0 or video_cache_capacity <= 0:
            raise ValueError("reader cache capacities must be positive")
        if observed_visual_mode not in {"history", "current_multiview"}:
            raise ValueError("unsupported observed_visual_mode")
        self.observed_visual_mode = observed_visual_mode
        if current_visual_pooling not in {"canvas_v1", "content_area_v1"}:
            raise ValueError("unsupported current visual pooling")
        self.current_visual_pooling = current_visual_pooling
        self.processor = processor
        self.capabilities = dict(capabilities)
        self.measurement_schemas = dict(measurement_schemas)
        manifest_path = _feature_manifest_path(feature_manifest)
        payload = verify_predictive_world_feature_manifest(manifest_path)
        index_binding = (
            predictive_world_index_binding(index_path)
            if not index_already_verified
            else predictive_world_index_binding(
                index_path, verify_content_hashes=False
            )
        )
        if payload.get("index_binding") != index_binding:
            raise ValueError("predictive-world feature manifest is bound to a different index")
        self.video_decoder = str(payload["provenance"].get("video_decoder", "opencv"))
        if self.video_decoder not in {"opencv", "torchcodec"}:
            raise ValueError("unsupported feature video decoder")
        self.image_preprocessing = str(payload["provenance"].get("image_preprocessing", ""))
        if self.current_visual_pooling == "content_area_v1" and (
            self.image_preprocessing != "aspect_v1"
            or payload["provenance"].get("feature_layer") != "post_merger_content_area_overlap_4x4_v1"
        ):
            raise ValueError("content-area observations require matching aspect_v1 future feature geometry")
        if self.image_preprocessing not in {"legacy_v1", "aspect_v1"}:
            raise ValueError("feature provenance must declare image_preprocessing")
        self.packed_feature_store: PackedFeatureStore | None = None
        self.feature_store: RaggedFeatureStore | None = None
        self.features: dict[str, list[dict[str, Any]]] = {}
        if payload.get("format") == PACKED_FEATURE_FORMAT:
            self.packed_feature_store = PackedFeatureStore(manifest_path)
        else:
            self.feature_store = RaggedFeatureStore(
                manifest_path.parent, provenance=payload["provenance"]
            )
            if payload.get("cache_identity") != self.feature_store.identity:
                raise ValueError("feature manifest provenance identity mismatch")
            for record in payload["records"]:
                self.features.setdefault(str(record["sample_id"]), []).append(record)
        self._tables: OrderedDict[tuple[str, tuple[str, ...]], Any] = OrderedDict()
        self._parquet_cache_capacity = int(parquet_cache_capacity)
        self._captures: OrderedDict[str, Any] = OrderedDict()
        self._video_cache_capacity = int(video_cache_capacity)
        self._verified_feature_keys: set[str] = set()

    def _table(self, path: str, columns: tuple[str, ...]):
        key = (str(Path(path).resolve()), columns)
        if key in self._tables:
            self._tables.move_to_end(key)
            return self._tables[key]
        import pyarrow.parquet as pq

        self._tables[key] = pq.read_table(key[0], columns=list(columns))
        while len(self._tables) > self._parquet_cache_capacity:
            self._tables.popitem(last=False)
        return self._tables[key]

    def _capture(self, path: str):
        import cv2

        cv2.setNumThreads(1)
        key = str(Path(path).expanduser().resolve())
        capture = self._captures.get(key)
        if capture is not None:
            self._captures.move_to_end(key)
            return capture
        capture = cv2.VideoCapture(key)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open predictive-world video: {key}")
        self._captures[key] = capture
        while len(self._captures) > self._video_cache_capacity:
            _, old = self._captures.popitem(last=False)
            old.release()
        return capture

    def _decode_frames(self, path: str, indices: Iterable[int]) -> list[torch.Tensor]:
        if getattr(self, "video_decoder", "opencv") == "torchcodec":
            from psi.data.predictive_world.video import decode_torchcodec_frames
            return decode_torchcodec_frames(
                path, indices, cache_capacity=getattr(self, "_video_cache_capacity", 8)
            )
        import cv2

        capture = self._capture(path)
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        output = []
        for source_index in indices:
            index = int(source_index)
            if frame_count > 0 and index == frame_count:
                index = frame_count - 1
            if index < 0 or (frame_count > 0 and index >= frame_count):
                raise RuntimeError(
                    f"predictive-world video frame is out of range {path}:{source_index}"
                )
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"cannot decode predictive-world video frame {path}:{source_index}"
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255))
        return output

    def __del__(self) -> None:
        for capture in getattr(self, "_captures", {}).values():
            capture.release()

    def _qwen(self, row: Mapping[str, Any], materialization: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        primary_valid = [bool(value) for value in row["history_visual_valid"]]
        if not primary_valid or not primary_valid[-1]:
            raise ValueError("predictive-world sample lacks a current causal image")
        video_paths = materialization.get("video_paths")
        if not isinstance(video_paths, Mapping) or not video_paths:
            raise ValueError("materialization needs non-empty camera video_paths")
        primary_camera = materialization.get("primary_camera_id")
        if primary_camera is None:
            if len(video_paths) != 1:
                raise ValueError("multi-camera materialization requires primary_camera_id")
            primary_camera = next(iter(video_paths))
        if primary_camera not in video_paths:
            raise ValueError("primary_camera_id is absent from video_paths")
        images, labels, primary = [], [], None
        indices_by_camera = row.get("history_visual_indices_by_camera")
        valid_by_camera = row.get("history_visual_valid_by_camera")
        times_by_camera = row.get("history_visual_actual_times_by_camera")
        has_per_camera = all(
            isinstance(value, Mapping)
            for value in (indices_by_camera, valid_by_camera, times_by_camera)
        )
        for camera_id, video_path in sorted(video_paths.items()):
            camera_indices = (
                indices_by_camera[camera_id]
                if has_per_camera
                else row["history_visual_indices"]
            )
            camera_valid = (
                valid_by_camera[camera_id]
                if has_per_camera
                else row["history_visual_valid"]
            )
            camera_times = (
                times_by_camera[camera_id]
                if has_per_camera
                else row["history_visual_actual_times"]
            )
            requested = [
                (slot, int(frame_index), float(actual_time))
                for slot, (present, frame_index, actual_time) in enumerate(zip(
                    camera_valid, camera_indices, camera_times
                ))
                if present and (
                    self.observed_visual_mode == "history" or slot == len(primary_valid) - 1
                )
            ]
            decoded = self._decode_frames(
                str(video_path), (frame_index for _, frame_index, _ in requested)
            )
            for image, (slot, _frame_index, actual_time) in zip(decoded, requested):
                if slot == len(primary_valid) - 1 and camera_id == primary_camera:
                    primary = len(images)
                images.append(preprocess_endpoint_vlm_image(
                    image,
                    image_preprocessing=self.image_preprocessing,
                ))
                labels.append((str(camera_id), float(actual_time)))
        if primary is None:
            raise ValueError("materialization did not identify a primary current image")
        content = []
        for image, (camera, time) in zip(images, labels):
            content.extend((
                {"type": "text", "text": f"Camera: {camera}; observation time: {time:+.3f} seconds."},
                {"type": "image", "image": image},
            ))
        content.append({"type": "text", "text": str(materialization.get("instruction", ""))})
        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        encoded = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        # Qwen's processor already invokes image_processor with these images and
        # expands text tokens from the returned grid. Equivalence to the former
        # second image-only pass is checked at the real processor test boundary.
        encoded["image_counts"] = torch.tensor(len(images))
        encoded["primary_current_image_indices"] = torch.tensor(primary)
        if self.current_visual_pooling == "content_area_v1":
            from psi.models.action_aligned_image import content_boxes
            encoded["image_content_boxes"] = content_boxes(images)
        return {key: value[0] if key in {"input_ids", "attention_mask"} else value for key, value in encoded.items()}

    def read(self, row: Mapping[str, Any], *, occurrence_index: int) -> dict[str, Any]:
        materialization = row.get("materialization")
        if not isinstance(materialization, Mapping):
            raise ValueError("index row lacks audited materialization metadata")
        _, measurement_schema = validate_predictive_world_row_capability(
            row,
            capabilities=self.capabilities,
            measurement_schemas=self.measurement_schemas,
        )
        sample_records = (
            []
            if self.packed_feature_store is not None
            else validate_predictive_world_sample_feature_mapping(
                row, self.features.get(str(row["sample_id"]), ())
            )
        )
        state_columns = tuple(str(value) for value in materialization.get("state_columns", ()))
        action_columns = tuple(str(value) for value in materialization.get("action_columns", ()))
        if not state_columns or not action_columns:
            raise ValueError("materialization requires state/action columns")
        table = self._table(str(materialization["data_path"]), tuple(dict.fromkeys((*state_columns, *action_columns))))

        def read_vectors(indices, columns):
            return _read_vectors(table, indices, columns)

        raw_states = read_vectors(row["state_indices"], state_columns)
        if raw_states.shape[1] != measurement_schema.width:
            raise ValueError(
                "materialized state width disagrees with the audited measurement schema"
            )
        state_time_valid = np.asarray(row["state_valid"], dtype=bool)
        state_valid = np.isfinite(raw_states) & state_time_valid[:, None]
        raw_states = np.where(state_valid, raw_states, 0.0).astype(np.float32, copy=False)
        from psi.data.predictive_world.action_labels import materialize_action_labels
        actions, action_time_valid = materialize_action_labels(
            row, lambda indices: read_vectors(indices, action_columns))
        action_valid = np.isfinite(actions) & action_time_valid
        actions = np.where(action_valid, actions, 0.0).astype(np.float32, copy=False)
        measurement_indices = tuple(materialization.get("measurement_indices", ()))
        if not measurement_indices:
            raise ValueError("materialization requires audited canonical measurement indices")
        future_raw = read_vectors(row["future_measurement_indices"], state_columns)
        future_time_valid = np.asarray(row["future_measurement_valid"], dtype=bool)
        canonical = measurement_schema.canonical_body
        if canonical is None:
            dense_indices = np.asarray(measurement_indices, dtype=np.int64)
            future_measurements = future_raw[:, dense_indices]
            future_measurement_valid = (
                np.isfinite(future_measurements) & future_time_valid[:, None]
            )
            current = raw_states[-1, dense_indices]
            current_valid = state_valid[-1, dense_indices]
        else:
            width = len(canonical.raw_indices)
            future_measurements = np.zeros((len(future_raw), width), dtype=np.float32)
            future_measurement_valid = np.zeros_like(future_measurements, dtype=bool)
            current = np.zeros(width, dtype=np.float32)
            current_valid = np.zeros(width, dtype=bool)
            means = np.asarray(canonical.mean, dtype=np.float32)
            stds = np.asarray(canonical.std, dtype=np.float32)
            for target, raw_index in enumerate(canonical.raw_indices):
                if raw_index is None:
                    continue
                values = future_raw[:, raw_index]
                valid_values = np.isfinite(values) & future_time_valid
                future_measurements[:, target] = np.where(
                    valid_values, (values - means[target]) / stds[target], 0.0
                )
                future_measurement_valid[:, target] = valid_values
                if state_valid[-1, raw_index]:
                    current[target] = (
                        raw_states[-1, raw_index] - means[target]
                    ) / stds[target]
                    current_valid[target] = True
        future_measurements = np.where(
            future_measurement_valid, future_measurements, 0.0
        ).astype(np.float32, copy=False)
        fields = [[] for _ in row["future_visual_indices"]]
        if self.packed_feature_store is not None:
            camera_ids = tuple(row["source_capability"].get("future_visual_keys", ()))
            if len(camera_ids) != 1:
                raise ValueError(
                    "packed feature cache currently requires one audited future camera"
                )
            feature_row = int(row.get("feature_row_index", -1))
            for group, present in enumerate(row["future_visual_valid"]):
                if not present:
                    continue
                fields[group].append({
                    "values": self.packed_feature_store.get(feature_row, group),
                    "grid_hw": (4, 4),
                    "camera_id": camera_ids[0],
                    "actual_time": float(row["future_visual_actual_times"][group]),
                })
        else:
            assert self.feature_store is not None
            for record in sample_records:
                feature_key = (int(record["group"]), str(record["camera_id"]))
                expected_frame_id = str(record["frame_id"])
                key = str(record["key"])
                verify_hash = (
                    None if key in self._verified_feature_keys else record["object_sha256"]
                )
                field = self.feature_store.get(key, verify_hash=verify_hash)
                if field.frame_id != expected_frame_id or field.camera_id != feature_key[1]:
                    raise ValueError(
                        "cached feature object disagrees with manifest frame identity"
                    )
                self._verified_feature_keys.add(key)
                fields[int(record["group"])].append({
                    "values": field.values,
                    "grid_hw": field.grid_hw,
                    "camera_id": field.camera_id,
                    "actual_time": float(record["actual_time"]),
                })
        result = {
            "sample_id": str(row["sample_id"]),
            "source_id": str(row["source_id"]),
            "episode_id": str(row["episode_id"]),
            "occurrence_index": int(occurrence_index),
            "split": str(row["split"]),
            "qwen_inputs": self._qwen(row, materialization),
            "raw_states": raw_states,
            "state_valid": state_valid,
            "state_times": row["state_actual_times"],
            "state_schema_id": str(materialization["state_schema_id"]),
            "actions": actions,
            "action_valid": action_valid,
            "action_times": row["action_actual_times"],
            "action_protocol_id": str(materialization["action_protocol_id"]),
            "action_normalization_space": str(materialization["action_normalization_space"]),
            "world_action_group_valid": tuple(
                bool(value) for value in row.get("world_action_group_valid", ())
            ),
            "current_measurements": current,
            "current_measurement_valid": current_valid,
            "future_patch_fields": fields,
            "future_measurements": future_measurements,
            "future_measurement_valid": future_measurement_valid,
            "future_measurement_times": row["future_measurement_actual_times"],
            "future_measurement_source_indices": row["future_measurement_indices"],
        }
        return result


class PredictiveWorldOccurrenceDataset(IterableDataset):
    """Infinite deterministic, source-tempered stream sharded before materialization.

    A source with ``n`` eligible rows receives weight ``n ** exponent``.  The
    finite smooth-weighted schedule is exactly reproducible, and each source is
    sampled without replacement until its own eligible table wraps.  This keeps
    the large general corpus useful without starving smaller robot/protocol
    domains or making resume depend on DataLoader worker timing.
    """

    def __init__(
        self,
        index_path: str | Path,
        reader: PredictiveWorldRowReader,
        *,
        split: str,
        seed: int,
        rank: int,
        world_size: int,
        start_occurrence: int,
        rank_batch_size: int,
        require_world_action: bool = False,
        require_complete_future: bool = False,
        source_sampling_exponent: float = 0.5,
        index_already_verified: bool = False,
    ) -> None:
        super().__init__()
        if not index_already_verified:
            verify_predictive_world_index(index_path)
        if (
            not math.isfinite(source_sampling_exponent)
            or not 0.0 <= source_sampling_exponent <= 1.0
        ):
            raise ValueError("source sampling exponent must be in [0,1]")
        self.lookup_path = predictive_world_lookup_path(index_path)
        self.view = (
            "policy" if require_complete_future
            else "world" if require_world_action
            else "prior"
        )
        connection = sqlite3.connect(f"file:{self.lookup_path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                """SELECT source_id, COUNT(*)
                   FROM occurrences WHERE split=? AND view=?
                   GROUP BY source_id ORDER BY source_id""",
                (split, self.view),
            ).fetchall()
        finally:
            connection.close()
        self.source_ids = tuple(str(row[0]) for row in rows)
        self.source_counts = tuple(int(row[1]) for row in rows)
        self.row_count = sum(self.source_counts)
        if not self.row_count:
            qualifiers = []
            if require_world_action:
                qualifiers.append("world-action eligible")
            if require_complete_future:
                qualifiers.append("complete-future")
            suffix = f" {' '.join(qualifiers)}" if qualifiers else ""
            raise ValueError(f"predictive-world index has no{suffix} {split!r} rows")
        self.reader, self.seed, self.split = reader, int(seed), str(split)
        self.source_sampling_exponent = float(source_sampling_exponent)
        self.source_weights = self._integer_source_weights(
            self.source_counts, exponent=self.source_sampling_exponent
        )
        self.source_schedule, self.source_schedule_offsets = self._source_schedule(
            self.source_weights
        )
        self.schedule_size = len(self.source_schedule)
        self.source_seeds = tuple(
            int.from_bytes(
                hashlib.sha256(
                    f"predictive-world-source-v1|{self.seed}|{source}".encode()
                ).digest()[:8],
                "big",
            )
            for source in self.source_ids
        )
        self._connection: sqlite3.Connection | None = None
        self.rank, self.world_size = int(rank), int(world_size)
        self.start_occurrence, self.rank_batch_size = int(start_occurrence), int(rank_batch_size)
        if not 0 <= rank < world_size or start_occurrence < 0 or rank_batch_size <= 0:
            raise ValueError("invalid occurrence dataset sharding")

    @staticmethod
    def _integer_source_weights(
        counts: tuple[int, ...], *, exponent: float, maximum_weight: int = 4096
    ) -> tuple[int, ...]:
        if not counts or any(count <= 0 for count in counts):
            raise ValueError("source occurrence counts must be positive")
        raw = tuple(float(count) ** exponent for count in counts)
        scale = maximum_weight / max(raw)
        return tuple(max(1, int(round(value * scale))) for value in raw)

    @staticmethod
    def _source_schedule(
        weights: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Build one exact, smoothly interleaved weighted round-robin cycle."""

        total = sum(weights)
        current = [0] * len(weights)
        emitted = [0] * len(weights)
        schedule: list[int] = []
        offsets: list[int] = []
        for _ in range(total):
            for index, weight in enumerate(weights):
                current[index] += weight
            selected = max(range(len(weights)), key=lambda index: (current[index], -index))
            current[selected] -= total
            schedule.append(selected)
            offsets.append(emitted[selected])
            emitted[selected] += 1
        if tuple(emitted) != weights:
            raise AssertionError("source schedule does not preserve its exact weights")
        return tuple(schedule), tuple(offsets)

    @staticmethod
    def _permuted_position(*, size: int, position: int, epoch: int, seed: int) -> int:
        if size == 1:
            return 0
        digest = hashlib.sha256(
            f"predictive-world-affine-v1|{seed}|{epoch}".encode()
        ).digest()
        multiplier = int.from_bytes(digest[:8], "big") % size or 1
        while math.gcd(multiplier, size) != 1:
            multiplier = (multiplier + 1) % size or 1
        offset = int.from_bytes(digest[8:16], "big") % size
        return (multiplier * position + offset) % size

    def _database(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(
                f"file:{self.lookup_path}?mode=ro", uri=True, check_same_thread=False
            )
        return self._connection

    def _row(self, occurrence: int) -> Mapping[str, Any]:
        schedule_epoch, schedule_position = divmod(occurrence, self.schedule_size)
        shuffled_schedule_position = self._permuted_position(
            size=self.schedule_size,
            position=schedule_position,
            epoch=schedule_epoch,
            seed=self.seed,
        )
        source_index = self.source_schedule[shuffled_schedule_position]
        source_draw = (
            schedule_epoch * self.source_weights[source_index]
            + self.source_schedule_offsets[shuffled_schedule_position]
        )
        source_epoch, source_position = divmod(
            source_draw, self.source_counts[source_index]
        )
        source_position = self._permuted_position(
            size=self.source_counts[source_index],
            position=source_position,
            epoch=source_epoch,
            seed=self.source_seeds[source_index],
        )
        row = self._database().execute(
            """SELECT records.payload
               FROM occurrences JOIN records
                 ON records.ordinal = occurrences.record_ordinal
               WHERE occurrences.split=? AND occurrences.view=?
                 AND occurrences.source_id=? AND occurrences.source_position=?""",
            (
                self.split,
                self.view,
                self.source_ids[source_index],
                source_position,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("predictive-world lookup is missing an eligible occurrence")
        return json.loads(row[0])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id, workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        rank_start = self.start_occurrence + ((self.rank - self.start_occurrence) % self.world_size)
        local_batch = 0
        while True:
            if local_batch % workers == worker_id:
                begin = rank_start + local_batch * self.rank_batch_size * self.world_size
                for offset in range(self.rank_batch_size):
                    occurrence = begin + offset * self.world_size
                    yield self.reader.read(self._row(occurrence), occurrence_index=occurrence)
            local_batch += 1


__all__ = [
    "PredictiveWorldOccurrenceDataset",
    "PredictiveWorldRowReader",
    "ValidatedPredictiveWorldDataArtifacts",
    "assert_predictive_world_training_data_contract",
    "validate_predictive_world_sample_feature_mapping",
]
