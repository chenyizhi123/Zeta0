"""Runtime data plane for the fixed Stage-A occurrence stream.

The large JSONL artifacts remain the experiment source of truth.  A SQLite
index is only a lossless acceleration structure and is checked against the
manifest digest before it can be used.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .registry import (
    canonical_json_bytes,
    canonical_sha256,
    iter_manifest_rows,
    load_manifest_rows,
)
from .sample_stream import COMPACT_V2_ALGORITHM, CompactV2OccurrenceResolver
from .sparse_cache import SparseEndpointCache


ANCHOR_INDEX_VERSION = "predictor_pretrain_anchor_index_v4"
_LEGACY_ANCHOR_INDEX_VERSIONS = {
    "predictor_pretrain_anchor_index_v2",
    "predictor_pretrain_anchor_index_v3",
}
_SOURCE_POSITION_INDEX_VERSIONS = {
    "predictor_pretrain_anchor_index_v3",
    ANCHOR_INDEX_VERSION,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_model_directory(path: str | Path) -> str:
    root = Path(path).resolve()
    files = sorted(
        value for value in root.iterdir()
        if value.is_file() and (value.suffix in {".json", ".safetensors"})
    )
    if not files or not any(value.suffix == ".safetensors" for value in files):
        raise ValueError(f"teacher directory has no safetensors weights: {root}")
    return canonical_sha256({value.name: sha256_file(value) for value in files})


def build_anchor_index(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    progress_every_rows: int = 0,
    row_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> Path:
    """Build a transactional, bounded-memory sample/cache-position index."""

    manifest_path, output_path = Path(manifest_path).resolve(), Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute(
            "CREATE TABLE anchors(sample_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, "
            "split TEXT NOT NULL, source_position INTEGER, cache_position INTEGER, "
            "payload TEXT NOT NULL)"
        )
        digest = hashlib.sha256(b"[")
        first, num_rows = True, 0
        pending = []
        protocols: set[str] = set()
        history_seconds: set[float] = set()
        future_seconds: set[float] = set()
        action_steps: set[int] = set()
        source_revisions: dict[str, set[str]] = {}
        split_counts: Counter[tuple[str, str]] = Counter()
        for row in iter_manifest_rows(manifest_path):
            if row_observer is not None:
                row_observer(row)
            if not first:
                digest.update(b",")
            digest.update(canonical_json_bytes(row))
            first = False
            pending.append((
                str(row["sample_id"]), str(row["source_id"]), str(row["split"]),
                None, None, json.dumps(row, sort_keys=True),
            ))
            protocol = str(row.get("format_version", "predictor_pretrain_anchor_v1"))
            protocols.add(protocol)
            if protocol.endswith(("_v2", "_v3")):
                history_seconds.add(float(row["visual_history_seconds"]))
                future_seconds.add(float(row["future_seconds"]))
            if protocol.endswith("_v3"):
                action_steps.add(int(row.get("action_steps", -1)))
            source = str(row["source_id"])
            source_revisions.setdefault(source, set()).add(str(row.get("revision", "")))
            split_counts[(source, str(row["split"]))] += 1
            num_rows += 1
            if progress_every_rows and num_rows % progress_every_rows == 0:
                print(json.dumps({
                    "event": "anchor_index_progress",
                    "phase": "ingest",
                    "rows": num_rows,
                }), flush=True)
            if len(pending) == 2_000:
                connection.executemany(
                    "INSERT INTO anchors VALUES (?, ?, ?, ?, ?, ?)", pending,
                )
                pending.clear()
        if pending:
            connection.executemany(
                "INSERT INTO anchors VALUES (?, ?, ?, ?, ?, ?)", pending,
            )
        digest.update(b"]")
        manifest_sha = digest.hexdigest()
        if progress_every_rows:
            print(json.dumps({
                "event": "anchor_index_progress", "phase": "source_sort",
                "rows": num_rows,
            }), flush=True)
        connection.execute(
            "CREATE INDEX anchors_source_sample ON anchors(source_id, sample_id)"
        )
        cursor = connection.execute(
            "SELECT rowid,source_id,split FROM anchors ORDER BY source_id,sample_id"
        )
        current_source = None
        cache_position = train_position = 0
        while True:
            batch = cursor.fetchmany(10_000)
            if not batch:
                break
            updates = []
            for rowid, source, split in batch:
                if source != current_source:
                    current_source = source
                    cache_position = train_position = 0
                source_position = None
                if split == "PT_train":
                    source_position = train_position
                    train_position += 1
                updates.append((source_position, cache_position, rowid))
                cache_position += 1
            connection.executemany(
                "UPDATE anchors SET source_position=?,cache_position=? WHERE rowid=?",
                updates,
            )
        cursor.close()
        if progress_every_rows:
            print(json.dumps({
                "event": "anchor_index_progress", "phase": "secondary_indexes",
                "rows": num_rows,
            }), flush=True)
        connection.execute(
            "CREATE INDEX anchors_split_source ON anchors(split, source_id, sample_id)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX anchors_train_source_position "
            "ON anchors(source_id, source_position) WHERE split='PT_train'"
        )
        connection.execute(
            "CREATE UNIQUE INDEX anchors_source_cache_position "
            "ON anchors(source_id, cache_position)"
        )
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        summary = {
            "protocols": sorted(protocols),
            "visual_history_seconds": sorted(history_seconds),
            "future_seconds": sorted(future_seconds),
            "action_steps": sorted(action_steps),
            "source_revisions": {
                source: sorted(values) for source, values in sorted(source_revisions.items())
            },
            "split_counts": {
                source: {
                    split: split_counts[(source, split)]
                    for split in sorted({key[1] for key in split_counts if key[0] == source})
                }
                for source in sorted(source_revisions)
            },
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (("format_version", ANCHOR_INDEX_VERSION), ("manifest_sha256", manifest_sha),
             ("num_rows", str(num_rows)),
             ("summary_json", json.dumps(summary, sort_keys=True))),
        )
        connection.commit()
        if progress_every_rows:
            print(json.dumps({
                "event": "anchor_index_progress", "phase": "committed",
                "rows": num_rows,
            }), flush=True)
    finally:
        connection.close()
    temporary.replace(output_path)
    return output_path


class AnchorIndex:
    def __init__(self, path: str | Path, *, expected_manifest_sha256: str) -> None:
        self.path = Path(path).resolve()
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        metadata = dict(self.connection.execute("SELECT key,value FROM metadata"))
        self.metadata = metadata
        self.format_version = metadata.get("format_version")
        if self.format_version not in {ANCHOR_INDEX_VERSION, *_LEGACY_ANCHOR_INDEX_VERSIONS}:
            raise ValueError("anchor index format mismatch")
        if metadata.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("anchor index was built from a different frozen manifest")

    def frozen_summary(self) -> dict[str, Any]:
        """Return build-time metadata, deriving it cheaply for legacy V4 indexes.

        Older indexes predate ``summary_json`` but already bind every payload to
        the frozen manifest hash. Counts come from SQL and invariant protocol
        fields need only one representative payload per source.
        """

        encoded = self.metadata.get("summary_json")
        if encoded:
            return json.loads(encoded)
        split_counts: dict[str, dict[str, int]] = {}
        for source, split, count in self.connection.execute(
            "SELECT source_id,split,COUNT(*) FROM anchors GROUP BY source_id,split"
        ):
            split_counts.setdefault(str(source), {})[str(split)] = int(count)
        protocols: set[str] = set()
        history_seconds: set[float] = set()
        future_seconds: set[float] = set()
        action_steps: set[int] = set()
        source_revisions: dict[str, list[str]] = {}
        for source in sorted(split_counts):
            result = self.connection.execute(
                "SELECT payload FROM anchors WHERE source_id=? LIMIT 1", (source,)
            ).fetchone()
            if result is None:
                raise ValueError(f"anchor index source {source!r} has no payload")
            row = json.loads(result[0])
            protocol = str(row.get("format_version", "predictor_pretrain_anchor_v1"))
            protocols.add(protocol)
            if protocol.endswith(("_v2", "_v3")):
                history_seconds.add(float(row["visual_history_seconds"]))
                future_seconds.add(float(row["future_seconds"]))
            if protocol.endswith("_v3"):
                action_steps.add(int(row.get("action_steps", -1)))
            source_revisions[source] = [str(row.get("revision", ""))]
        return {
            "protocols": sorted(protocols),
            "visual_history_seconds": sorted(history_seconds),
            "future_seconds": sorted(future_seconds),
            "action_steps": sorted(action_steps),
            "source_revisions": source_revisions,
            "split_counts": split_counts,
            "derived_from_legacy_index": True,
        }

    def get(self, sample_id: str) -> dict[str, Any]:
        result = self.connection.execute(
            "SELECT payload FROM anchors WHERE sample_id=?", (str(sample_id),)
        ).fetchone()
        if result is None:
            raise KeyError(f"occurrence sample absent from frozen anchor index: {sample_id}")
        row = json.loads(result[0])
        if row.get("split") != "PT_train":
            raise ValueError(f"Stage A occurrence references non-PT_train sample {sample_id}")
        return row

    def get_train_source_position(self, source_id: str, position: int) -> dict[str, Any]:
        if self.format_version not in _SOURCE_POSITION_INDEX_VERSIONS:
            raise ValueError("compact V2 sampling requires a source-position anchor index")
        result = self.connection.execute(
            "SELECT payload FROM anchors WHERE source_id=? AND split='PT_train' "
            "AND source_position=?",
            (str(source_id), int(position)),
        ).fetchone()
        if result is None:
            raise KeyError(
                f"compact occurrence source position is absent: {source_id}[{position}]"
            )
        return json.loads(result[0])

    def iter_split(self, split: str) -> Iterator[dict[str, Any]]:
        if split not in {"PT_train", "PT_val", "PT_test"}:
            raise ValueError("evaluation split must be a PT split")
        cursor = self.connection.execute(
            "SELECT payload FROM anchors WHERE split=? ORDER BY source_id,sample_id", (split,)
        )
        for (payload,) in cursor:
            yield json.loads(payload)


@dataclass(frozen=True)
class LeRobotRuntimeSource:
    source_id: str
    state_schema: str | None
    state_columns: tuple[str, ...] = ()
    state_schema_field: str | None = None
    action_columns: tuple[str, ...] = ()
    action_protocol: str | None = None
    action_protocol_field: str | None = None
    action_protocol_map: Mapping[str, str] | None = None
    image_crop: str = "none"
    video_backend: str = "pyav"

    def __post_init__(self) -> None:
        if self.image_crop not in {"none", "left_half", "right_half"}:
            raise ValueError("image_crop must be none, left_half or right_half")
        if self.state_schema is not None and self.state_schema_field is not None:
            raise ValueError("declare a fixed state schema or a row schema field, not both")
        if self.state_schema is None and self.state_schema_field is None and self.state_columns:
            raise ValueError("missing-state sources cannot declare state columns")
        if self.action_protocol is not None and self.action_protocol_field is not None:
            raise ValueError("declare a fixed action protocol or a row protocol field, not both")
        if self.action_columns and self.action_protocol is None and self.action_protocol_field is None:
            raise ValueError("action columns require a fixed protocol or row protocol field")
        if not self.action_columns and (self.action_protocol is not None or self.action_protocol_field is not None):
            raise ValueError("an action protocol requires action columns")


class _ParquetLRU:
    def __init__(self, capacity: int = 2) -> None:
        self.capacity = capacity
        self.tables: OrderedDict[tuple[str, tuple[str, ...]], Any] = OrderedDict()

    def read(self, path: str, columns: Sequence[str]):
        key = (path, tuple(columns))
        if key in self.tables:
            self.tables.move_to_end(key)
            return self.tables[key]
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=list(columns))
        self.tables[key] = table
        while len(self.tables) > self.capacity:
            self.tables.popitem(last=False)
        return table


def _vector(value: Any) -> np.ndarray:
    if hasattr(value, "as_py"):
        value = value.as_py()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _clamp_opencv_endpoint_frame(frame_index: int, frame_count: int) -> int:
    """Map an exact duration timestamp to the last zero-based video frame."""

    if frame_count > 0 and frame_index == frame_count:
        return frame_count - 1
    return frame_index


def decode_video_frames_opencv(
    path: Path, timestamps: list[float], tolerance: float, _backend: str = "opencv"
) -> torch.Tensor:
    """Decode fixed-FPS frames with an explicit timestamp tolerance check."""

    import cv2
    # OpenCV defaults to the host core count in each DataLoader process.  With
    # 64+ workers per node that silently creates thousands of runnable threads.
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"video reports invalid FPS: {path}")
    output = []
    try:
        for timestamp in timestamps:
            frame_index = int(round(float(timestamp) * fps))
            actual = frame_index / fps
            if abs(actual - float(timestamp)) > tolerance:
                raise RuntimeError(f"video timestamp misses tolerance: requested={timestamp}, actual={actual}")
            # Some LeRobot episodes timestamp their final parquet row at the
            # exact video duration.  Rounding then produces frame_count even
            # though the corresponding final video frame is frame_count - 1.
            # Clamp only this single, well-defined endpoint case; any larger
            # mismatch must still fail as corrupt/inconsistent source data.
            frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
            frame_index = _clamp_opencv_endpoint_frame(frame_index, frame_count)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode {path} frame {frame_index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            output.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0))
    finally:
        capture.release()
    return torch.stack(output)


class _OpenCVDecoderLRU:
    def __init__(self, capacity: int = 2) -> None:
        self.capacity = capacity
        self.captures: OrderedDict[str, Any] = OrderedDict()

    def __call__(self, path: Path, timestamps: list[float], tolerance: float, _backend: str):
        import cv2
        cv2.setNumThreads(1)
        key = str(path)
        capture = self.captures.get(key)
        if capture is None:
            capture = cv2.VideoCapture(key)
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {path}")
            self.captures[key] = capture
            while len(self.captures) > self.capacity:
                _, old = self.captures.popitem(last=False)
                old.release()
        else:
            self.captures.move_to_end(key)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"video reports invalid FPS: {path}")
        output = []
        for timestamp in timestamps:
            frame_index = int(round(float(timestamp) * fps))
            actual = frame_index / fps
            if abs(actual - float(timestamp)) > tolerance:
                raise RuntimeError(f"video timestamp misses tolerance: requested={timestamp}, actual={actual}")
            frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
            frame_index = _clamp_opencv_endpoint_frame(frame_index, frame_count)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode {path} frame {frame_index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            output.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0))
        return torch.stack(output)


class LeRobotAnchorReader:
    """Materialize manifest-declared RGB views/history and aligned state."""

    def __init__(
        self,
        sources: Mapping[str, LeRobotRuntimeSource],
        *,
        decoder: Callable[[Path, list[float], float, str], torch.Tensor] | None = None,
    ) -> None:
        self.sources = dict(sources)
        self.parquet = _ParquetLRU()
        self.decoder = decoder
        self.opencv_decoder = _OpenCVDecoderLRU()

    def _decode(self, row: Mapping[str, Any], config: LeRobotRuntimeSource) -> torch.Tensor:
        decoder = self.decoder
        if decoder is None:
            if config.video_backend == "opencv":
                decoder = self.opencv_decoder
            else:
                try:
                    from lerobot.common.datasets.video_utils import decode_video_frames
                except ImportError:
                    try:
                        from lerobot.datasets.video_utils import decode_video_frames
                    except ImportError as error:
                        raise RuntimeError(
                            "pyav/torchcodec decoding requires the LeRobot environment; "
                            "use video_backend=opencv in this environment"
                        ) from error
                decoder = decode_video_frames
        nominal_fps = float(row["nominal_fps"])
        video_timestamp = float(row["current_video_timestamp"])
        # Dedicated LeRobot episode videos have one frame per parquet row.
        # Prefer that exact identity over a timestamp that may drift by one or
        # more frames near the end of an episode. Shared/segmented videos keep
        # using their explicit video timestamp and offset.
        if (
            config.video_backend == "opencv"
            and not bool(row.get("shared_data_file"))
            and row.get("current_row_index") is not None
        ):
            video_timestamp = int(row["current_row_index"]) / nominal_fps
        tolerance = 0.5 / nominal_fps + 0.001
        frames = decoder(
            Path(str(row["video_path"])), [video_timestamp],
            tolerance, config.video_backend,
        )
        image = torch.as_tensor(frames)
        while image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3:
            raise RuntimeError(f"video decoder returned invalid image shape {tuple(image.shape)}")
        # LeRobot decoders return CHW float tensors.  Crop on the width axis.
        if config.image_crop != "none":
            width_axis = 2 if image.shape[0] in (1, 3, 4) else 1
            width = image.shape[width_axis]
            if width % 2:
                raise ValueError("stereo image width must be even")
            sl = [slice(None)] * 3
            sl[width_axis] = slice(0, width // 2) if config.image_crop == "left_half" else slice(width // 2, width)
            image = image[tuple(sl)]
        return image

    def _schema(self, row: Mapping[str, Any], config: LeRobotRuntimeSource) -> str | None:
        if config.state_schema_field is None:
            return config.state_schema
        value = str(row.get(config.state_schema_field, "")).strip()
        if not value:
            raise ValueError(f"row lacks state schema field {config.state_schema_field!r}")
        return value

    def _state(self, row: Mapping[str, Any], config: LeRobotRuntimeSource) -> tuple[np.ndarray, np.ndarray]:
        indices = [int(value) for value in row["state_row_indices"]]
        history_valid = np.asarray(row["state_history_mask"], dtype=bool)
        if self._schema(row, config) is None:
            return np.zeros((len(indices), 0), np.float32), history_valid & False
        table = self.parquet.read(str(row["data_path"]), config.state_columns)
        vectors = []
        for index in indices:
            pieces = [_vector(table[column][index]) for column in config.state_columns]
            vectors.append(np.concatenate(pieces))
        states = np.stack(vectors).astype(np.float32, copy=False)
        return states, history_valid

    def _action_protocol(self, row: Mapping[str, Any], config: LeRobotRuntimeSource) -> str | None:
        if config.action_protocol_field is None:
            return config.action_protocol
        value = str(row.get(config.action_protocol_field, "")).strip()
        if not value:
            raise ValueError(f"row lacks action protocol field {config.action_protocol_field!r}")
        mapping = dict(config.action_protocol_map or {})
        if value not in mapping:
            raise ValueError(f"unmapped action protocol value {value!r} for {config.source_id}")
        return mapping[value]

    def _action(self, row: Mapping[str, Any], config: LeRobotRuntimeSource) -> tuple[np.ndarray, np.ndarray]:
        protocol = self._action_protocol(row, config)
        if protocol is None:
            return np.zeros((0, 0), np.float32), np.zeros((0,), dtype=bool)
        if row.get("format_version") != "predictor_pretrain_anchor_v3":
            raise ValueError("action-supervised Stage A requires a V3 anchor with frozen action indices")
        indices = np.asarray(row["action_row_indices"], dtype=np.int64)
        valid = np.asarray(row["action_time_mask"], dtype=bool)
        if indices.ndim != 1 or valid.shape != indices.shape:
            raise ValueError("invalid frozen action indices/mask")
        columns = tuple(dict.fromkeys((*config.action_columns,)))
        table = self.parquet.read(str(row["data_path"]), columns)
        vectors = []
        for index in indices.tolist():
            pieces = [_vector(table[column][index]) for column in config.action_columns]
            vectors.append(np.concatenate(pieces))
        actions = np.stack(vectors).astype(np.float32, copy=False)
        finite = np.isfinite(actions).all(axis=1)
        return np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0), valid & finite

    def read(self, row: Mapping[str, Any]) -> dict[str, Any]:
        source_id = str(row["source_id"])
        if source_id not in self.sources:
            raise KeyError(f"no runtime source configuration for {source_id!r}")
        config = self.sources[source_id]
        schema = self._schema(row, config)
        action_protocol = self._action_protocol(row, config)
        states, history_mask = self._state(row, config)
        actions, action_time_mask = self._action(row, config)
        result = {
            "instruction": str(row.get("instruction", "")),
            "raw_states": states,
            "state_history_mask": history_mask,
            "state_schema_id": schema or "missing",
            "state_present": schema is not None and bool(history_mask[-1]),
            "raw_actions": actions,
            "action_time_mask": action_time_mask,
            "action_protocol_id": action_protocol or "missing",
            "action_present": action_protocol is not None and bool(action_time_mask.any()),
        }
        if row.get("format_version") not in {
            "predictor_pretrain_anchor_v2", "predictor_pretrain_anchor_v3",
        }:
            result["image"] = self._decode(row, config)
            return result
        slots = row.get("visual_inputs")
        if not slots:
            raise ValueError("V2 anchor is missing visual_inputs")
        images, present_slots = [], []
        primary_index = None
        for slot_index, slot in enumerate(slots):
            if not slot["available"]:
                continue
            if not Path(str(slot["video_path"])).is_file():
                raise FileNotFoundError(f"manifest-declared input video disappeared: {slot['video_path']}")
            frame = {
                **dict(row), "video_path": slot["video_path"],
                "current_video_timestamp": slot["video_timestamp"],
                "current_row_index": slot["row_index"],
            }
            view_config = config if slot["image_key"] == row["primary_image_key"] else replace(config, image_crop="none")
            images.append(self._decode(frame, view_config))
            present_slots.append({**dict(slot), "slot_index": slot_index})
            if slot["is_primary_current"]:
                if primary_index is not None:
                    raise ValueError("V2 anchor has multiple primary-current images")
                primary_index = len(images) - 1
        if primary_index is None:
            raise ValueError("V2 anchor lacks its primary-current image")
        result.update({
            "image": images[primary_index], "images": images,
            "image_slots": present_slots,
            "image_slot_mask": [bool(slot["available"]) for slot in slots],
            "primary_current_image_index": primary_index,
        })
        return result


class FixedOccurrenceDataset(IterableDataset):
    """Stream exact global occurrences, then shard by DDP rank and worker."""

    def __init__(
        self,
        stream_path: str | Path,
        stream_meta_path: str | Path,
        anchor_index_path: str | Path,
        reader: LeRobotAnchorReader,
        caches: Mapping[str, SparseEndpointCache],
        *,
        expected_dataset_manifest_sha256: str,
        rank: int = 0,
        world_size: int = 1,
        start_occurrence: int = 0,
        rank_batch_size: int | None = None,
        verify_stream_hash: bool = True,
    ) -> None:
        super().__init__()
        self.stream_path = Path(stream_path)
        meta = json.loads(Path(stream_meta_path).read_text())
        if meta.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
            raise ValueError("sample stream and frozen dataset manifest disagree")
        if verify_stream_hash and sha256_file(self.stream_path) != meta.get("stream_file_sha256"):
            raise ValueError("sample stream content hash mismatch")
        if meta.get("stream_encoding") == "algorithmic_v2":
            lines = [line for line in self.stream_path.read_text(encoding="utf-8").splitlines() if line]
            if len(lines) != 1:
                raise ValueError("compact V2 sample stream must contain one descriptor")
            descriptor = json.loads(lines[0])
            expected = {
                "format_version": "predictor_pretrain_compact_sample_stream_v2",
                "sampling_algorithm": COMPACT_V2_ALGORITHM,
                "dataset_manifest_sha256": expected_dataset_manifest_sha256,
                "source_weights": meta.get("source_weights"),
                "source_capacities": meta.get("source_capacities"),
                "total_updates": meta.get("total_updates"),
                "global_batch_size": meta.get("global_batch_size"),
                "seed": meta.get("seed"),
                "state_dropout_probability": meta.get("state_dropout_probability"),
                "num_occurrences": meta.get("num_occurrences"),
            }
            if descriptor != expected:
                raise ValueError("compact V2 sample stream descriptor and metadata disagree")
        self.stream_meta = meta
        self.anchor_index_path = Path(anchor_index_path)
        self.reader, self.caches = reader, dict(caches)
        self.expected_manifest_sha = expected_dataset_manifest_sha256
        self.rank, self.world_size = int(rank), int(world_size)
        self.start_occurrence = int(start_occurrence)
        self.rank_batch_size = None if rank_batch_size is None else int(rank_batch_size)
        if not 0 <= self.rank < self.world_size or self.start_occurrence < 0:
            raise ValueError("invalid rank/world_size/start_occurrence")
        if self.rank_batch_size is not None and self.rank_batch_size <= 0:
            raise ValueError("rank_batch_size must be positive")

    def _materialize_occurrence(
        self,
        occurrence: Mapping[str, Any],
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        if str(row["source_id"]) != str(occurrence["source_id"]):
            raise ValueError("sample stream source_id does not match anchor manifest")
        materialized = self.reader.read(row)
        cache = self.caches[str(row["source_id"])]
        target, cached_timestamp = cache.get(str(row["sample_id"]))
        if abs(cached_timestamp - float(row["target_timestamp"])) > 1e-6:
            raise ValueError("future target timestamp does not match frozen anchor")
        if bool(occurrence.get("drop_state")):
            materialized["raw_states"] = np.zeros_like(materialized["raw_states"])
            materialized["state_history_mask"] = np.zeros_like(
                materialized["state_history_mask"]
            )
            materialized["state_present"] = False
        return {
            **occurrence,
            **materialized,
            "sample_id": str(row["sample_id"]),
            "episode_id": str(row.get("episode_id", "")),
            "intent_target_vlm_state": target,
        }

    def _iter_compact_v2(
        self,
        index: AnchorIndex,
        *,
        worker_id: int,
        workers: int,
    ) -> Iterator[dict[str, Any]]:
        meta = self.stream_meta
        if meta.get("sampling_algorithm") not in (None, COMPACT_V2_ALGORITHM):
            raise ValueError("unsupported compact V2 sampling algorithm")
        resolver = CompactV2OccurrenceResolver(
            source_weights=meta["source_weights"],
            source_capacities=meta["source_capacities"],
            seed=int(meta["seed"]),
            state_dropout_probability=float(meta["state_dropout_probability"]),
        )
        total = int(meta["num_occurrences"])
        first_rank_line = self.start_occurrence + (
            self.rank - self.start_occurrence % self.world_size
        ) % self.world_size
        if first_rank_line >= total:
            return
        rank_items = (total - 1 - first_rank_line) // self.world_size + 1
        batch_size = self.rank_batch_size or 1
        rank_batches = (rank_items + batch_size - 1) // batch_size
        for local_batch in range(worker_id, rank_batches, workers):
            begin = local_batch * batch_size
            stop = min(begin + batch_size, rank_items)
            for local_position in range(begin, stop):
                occurrence_index = first_rank_line + local_position * self.world_size
                occurrence = resolver.occurrence(occurrence_index)
                source_position = int(occurrence.pop("source_position"))
                row = index.get_train_source_position(
                    str(occurrence["source_id"]), source_position
                )
                occurrence["update"] = occurrence_index // int(meta["global_batch_size"])
                occurrence["position_in_global_batch"] = (
                    occurrence_index % int(meta["global_batch_size"])
                )
                yield self._materialize_occurrence(occurrence, row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id, workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        if workers > 1 and self.rank_batch_size is None:
            raise RuntimeError(
                "multi-worker fixed-occurrence loading requires rank_batch_size "
                "to preserve exact global batches"
            )
        index = AnchorIndex(self.anchor_index_path, expected_manifest_sha256=self.expected_manifest_sha)
        if self.stream_meta.get("stream_encoding") == "algorithmic_v2":
            yield from self._iter_compact_v2(
                index, worker_id=worker_id, workers=workers
            )
            return
        first_rank_line = self.start_occurrence + (
            self.rank - self.start_occurrence % self.world_size
        ) % self.world_size
        with self.stream_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line_number < self.start_occurrence or line_number % self.world_size != self.rank:
                    continue
                # DataLoader assigns whole batches to workers in round-robin
                # order.  Shard rank-local *batches*, not individual rows, so
                # multi-worker loading yields exactly the same occurrence
                # order as num_workers=0 and resume remains bit-exact.
                local_position = (line_number - first_rank_line) // self.world_size
                local_batch = local_position // (self.rank_batch_size or 1)
                if local_batch % workers != worker_id:
                    continue
                occurrence = json.loads(line)
                occurrence_index = int(occurrence["occurrence_index"])
                if occurrence_index != line_number:
                    raise ValueError("sample stream occurrence indices are not contiguous")
                row = index.get(str(occurrence["sample_id"]))
                yield self._materialize_occurrence(occurrence, row)


class PredictorEvaluationDataset(IterableDataset):
    """One deterministic pass over unique PT_val/PT_test anchors."""

    def __init__(self, anchor_index_path: str | Path, reader: LeRobotAnchorReader,
                 caches: Mapping[str, SparseEndpointCache], *, split: str,
                 expected_dataset_manifest_sha256: str, rank: int = 0, world_size: int = 1,
                 max_samples: int | None = None) -> None:
        super().__init__()
        self.anchor_index_path, self.reader, self.caches = Path(anchor_index_path), reader, dict(caches)
        self.split, self.expected_sha = split, expected_dataset_manifest_sha256
        self.rank, self.world_size, self.max_samples = rank, world_size, max_samples

    def __iter__(self):
        index = AnchorIndex(self.anchor_index_path, expected_manifest_sha256=self.expected_sha)
        for global_index, row in enumerate(index.iter_split(self.split)):
            if self.max_samples is not None and global_index >= self.max_samples:
                break
            if global_index % self.world_size != self.rank:
                continue
            item = self.reader.read(row)
            target, timestamp = self.caches[str(row["source_id"])].get(str(row["sample_id"]))
            if abs(timestamp - float(row["target_timestamp"])) > 1e-6:
                raise ValueError("evaluation target timestamp mismatch")
            yield {"occurrence_index": global_index, "sample_id": row["sample_id"],
                   "source_id": row["source_id"], "episode_id": str(row.get("episode_id", "")), **item,
                   "intent_target_vlm_state": target}


__all__ = [
    "ANCHOR_INDEX_VERSION", "AnchorIndex", "FixedOccurrenceDataset", "PredictorEvaluationDataset",
    "LeRobotAnchorReader", "LeRobotRuntimeSource", "build_anchor_index",
    "decode_video_frames_opencv", "fingerprint_model_directory", "sha256_file",
]
