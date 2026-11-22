"""Immutable content-addressed stores for ragged Qwen patches and world latents."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


FEATURE_FORMAT = "oraclevla.predictive_world.features.v2"
PACKED_FEATURE_FORMAT = "oraclevla.predictive_world.packed_features.v1"
FEATURE_RECORDS_FORMAT = "oraclevla.predictive_world.feature_records.v1"
LATENT_FORMAT = "oraclevla.predictive_world.latents.v1"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_records_metadata_path(path: str | Path) -> Path:
    records = Path(path).expanduser().resolve()
    return records.with_suffix(records.suffix + ".meta.json")


def verify_predictive_world_feature_records(path: str | Path) -> dict[str, Any]:
    records = Path(path).expanduser().resolve()
    metadata_path = feature_records_metadata_path(records)
    if not records.is_file() or not metadata_path.is_file():
        raise ValueError("feature records publication is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != FEATURE_RECORDS_FORMAT:
        raise ValueError("feature records metadata format mismatch")
    if metadata.get("records_sha256") != file_sha256(records):
        raise ValueError("feature records content hash mismatch")
    binding = metadata.get("index_binding")
    if not isinstance(binding, Mapping) or any(
        not binding.get(key)
        for key in (
            "format", "semantic_identity", "index_sha256", "lookup_sha256",
            "temporal_abi_sha256",
        )
    ):
        raise ValueError("feature records lack a complete index binding")
    with records.open("r", encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    if count != int(metadata.get("records", -1)):
        raise ValueError("feature records row count mismatch")
    return metadata


def verify_predictive_world_feature_manifest(
    path: str | Path, *, verify_shard_hashes: bool = False
) -> dict[str, Any]:
    manifest = Path(path).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("format") == PACKED_FEATURE_FORMAT:
        provenance = payload.get("provenance")
        binding = payload.get("index_binding")
        shards = payload.get("shards")
        shape = payload.get("feature_shape")
        if (
            not isinstance(provenance, Mapping)
            or payload.get("cache_identity") != canonical_sha256(dict(provenance))
            or not isinstance(binding, Mapping)
            or not isinstance(shards, list)
            or not shards
            or shape != [16, 2048]
            or payload.get("dtype") != "float16"
            or int(payload.get("groups", 0)) <= 0
            or int(payload.get("rows", 0)) <= 0
        ):
            raise ValueError("packed feature manifest is incomplete")
        expected_start = 0
        for shard_index, shard in enumerate(shards):
            if not isinstance(shard, Mapping):
                raise ValueError("packed feature shard descriptor must be an object")
            start, stop = int(shard.get("start", -1)), int(shard.get("stop", -1))
            relative = Path(str(shard.get("file", "")))
            digest = str(shard.get("sha256", ""))
            if (
                int(shard.get("rank", -1)) != shard_index
                or start != expected_start
                or stop <= start
                or relative.is_absolute()
                or ".." in relative.parts
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("packed feature shard ranges/identity are invalid")
            shard_path = manifest.parent / relative
            if not shard_path.is_file() or shard_path.stat().st_size != int(
                shard.get("nbytes", -1)
            ):
                raise ValueError(f"packed feature shard is missing/truncated: {shard_path}")
            if verify_shard_hashes and file_sha256(shard_path) != digest:
                raise ValueError(f"packed feature shard content hash mismatch: {shard_path}")
            expected_start = stop
        if expected_start != int(payload["rows"]):
            raise ValueError("packed feature shards do not cover every index row")
        identity = canonical_sha256({
            "cache_identity": payload["cache_identity"],
            "index_binding": dict(binding),
            "rows": int(payload["rows"]),
            "groups": int(payload["groups"]),
            "shards": [dict(value) for value in shards],
        })
        if payload.get("manifest_identity") != identity:
            raise ValueError("packed feature manifest identity mismatch")
        return payload
    if payload.get("format") != FEATURE_FORMAT:
        raise ValueError("feature manifest format mismatch")
    provenance = payload.get("provenance")
    records = payload.get("records")
    binding = payload.get("index_binding")
    if not isinstance(provenance, Mapping) or not isinstance(records, list):
        raise ValueError("feature manifest is incomplete")
    if payload.get("cache_identity") != canonical_sha256(dict(provenance)):
        raise ValueError("feature manifest provenance identity mismatch")
    if not isinstance(binding, Mapping):
        raise ValueError("feature manifest lacks an index binding")
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("feature manifest record must be an object")
        try:
            identity = (
                str(record["sample_id"]),
                int(record["group"]),
                str(record["camera_id"]),
            )
            actual_time = float(record["actual_time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("feature manifest record is incomplete") from error
        if (
            identity in seen
            or not identity[0]
            or identity[1] < 0
            or not identity[2]
            or not str(record.get("frame_id", ""))
            or not str(record.get("key", ""))
            or len(str(record.get("object_sha256", ""))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(record.get("object_sha256", ""))
            )
            or not np.isfinite(actual_time)
            or actual_time <= 0
        ):
            raise ValueError("feature manifest record identity/hash/time is invalid")
        seen.add(identity)
    mapping_sha256 = canonical_sha256({"records": records})
    if payload.get("mapping_sha256") != mapping_sha256:
        raise ValueError("feature manifest mapping hash mismatch")
    identity = canonical_sha256({
        "cache_identity": payload["cache_identity"],
        "index_binding": dict(binding),
        "feature_records_sha256": payload.get("feature_records_sha256"),
        "mapping_sha256": mapping_sha256,
    })
    if payload.get("manifest_identity") != identity:
        raise ValueError("feature manifest identity mismatch")
    return payload


class PackedFeatureStore:
    """Lazy mmap reader for fixed 4x4 Qwen target fields sharded by index row."""

    def __init__(self, manifest: str | Path) -> None:
        self.manifest_path = Path(manifest).expanduser().resolve()
        self.manifest = verify_predictive_world_feature_manifest(self.manifest_path)
        if self.manifest.get("format") != PACKED_FEATURE_FORMAT:
            raise ValueError("PackedFeatureStore requires a packed feature manifest")
        self.shards = tuple(self.manifest["shards"])
        self.stops = tuple(int(value["stop"]) for value in self.shards)
        self._arrays: dict[int, np.ndarray] = {}

    def _array(self, shard_index: int) -> np.ndarray:
        if shard_index not in self._arrays:
            descriptor = self.shards[shard_index]
            value = np.load(
                self.manifest_path.parent / descriptor["file"],
                mmap_mode="r",
                allow_pickle=False,
            )
            expected = (
                int(descriptor["stop"]) - int(descriptor["start"]),
                int(self.manifest["groups"]),
                *tuple(int(item) for item in self.manifest["feature_shape"]),
            )
            if value.shape != expected or value.dtype != np.float16:
                raise ValueError("packed feature shard tensor metadata mismatch")
            self._arrays[shard_index] = value
        return self._arrays[shard_index]

    def get(self, row_index: int, group: int) -> Tensor:
        import bisect

        row_index, group = int(row_index), int(group)
        if not 0 <= row_index < int(self.manifest["rows"]):
            raise IndexError("packed feature row is out of range")
        if not 0 <= group < int(self.manifest["groups"]):
            raise IndexError("packed feature group is out of range")
        shard_index = bisect.bisect_right(self.stops, row_index)
        descriptor = self.shards[shard_index]
        local = row_index - int(descriptor["start"])
        # Copying 64 KiB makes the worker tensor writable and detaches it from
        # mmap lifetime without loading any other sample.
        return torch.from_numpy(np.array(self._array(shard_index)[local, group], copy=True))


def feature_identity(
    *,
    frame_id: str,
    actual_time: float,
    camera_id: str,
    grid_hw: Sequence[int],
    provenance: Mapping[str, Any],
) -> str:
    """Identity excludes training knobs and includes every feature-determining input."""

    payload = {
        "frame_id": str(frame_id),
        "camera_id": str(camera_id),
        "grid_hw": [int(value) for value in grid_hw],
        "provenance": dict(provenance),
    }
    if (
        not payload["frame_id"]
        or not payload["camera_id"]
        or not np.isfinite(float(actual_time))
        or min(payload["grid_hw"]) <= 0
    ):
        raise ValueError("feature identity needs a finite time and positive grid")
    return canonical_sha256(payload)


@dataclass(frozen=True)
class CachedPatchField:
    key: str
    values: Tensor
    grid_hw: tuple[int, int]
    actual_time: float
    camera_id: str
    frame_id: str


class RaggedFeatureStore:
    """One immutable safetensors object per unique frame with an atomic manifest."""

    def __init__(self, root: str | Path, *, provenance: Mapping[str, Any]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.provenance = dict(provenance)
        required = {
            "encoder_id", "processor_id", "image_preprocessing", "feature_layer", "dtype"
        }
        if (
            required - self.provenance.keys()
            or any("REQUIRED_" in str(value) for value in self.provenance.values())
        ):
            raise ValueError("feature cache requires complete encoder/processor provenance")
        if self.provenance["image_preprocessing"] not in {"legacy_v1", "aspect_v1"}:
            raise ValueError("feature cache image preprocessing is unsupported")
        self.identity = canonical_sha256(self.provenance)
        self.objects = self.root / "objects"
        self.manifest_path = self.root / "feature_manifest.json"

    def put(
        self,
        *,
        frame_id: str,
        values: Tensor,
        grid_hw: Sequence[int],
        actual_time: float,
        camera_id: str,
    ) -> str:
        from safetensors.torch import save_file

        height, width = (int(value) for value in grid_hw)
        if values.ndim != 2 or values.shape[0] != height * width:
            raise ValueError("patch values must be [grid_h*grid_w,D]")
        if not torch.isfinite(values).all():
            raise ValueError("cached patch values must be finite")
        key = feature_identity(
            frame_id=frame_id,
            actual_time=actual_time,
            camera_id=camera_id,
            grid_hw=(height, width),
            provenance=self.provenance,
        )
        self.objects.mkdir(parents=True, exist_ok=True)
        destination = self.objects / f"{key}.safetensors"
        metadata = {
            "format": FEATURE_FORMAT,
            "cache_identity": self.identity,
            "frame_id": str(frame_id),
            "actual_time": repr(float(actual_time)),
            "camera_id": str(camera_id),
            "grid_hw": f"{height},{width}",
        }
        if destination.exists():
            current = self.get(key)
            if not torch.equal(current.values, values.detach().cpu().to(current.values.dtype)):
                raise ValueError("content-addressed feature object exists with different values")
            return key
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        save_file({"patches": values.detach().cpu().contiguous()}, str(temporary), metadata=metadata)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
        published = self.get(key)
        if not torch.equal(published.values, values.detach().cpu().to(published.values.dtype)):
            raise ValueError("concurrent feature publication produced different values")
        return key

    def get(self, key: str, *, verify_hash: str | None = None) -> CachedPatchField:
        from safetensors import safe_open

        path = self.objects / f"{key}.safetensors"
        if not path.is_file():
            raise KeyError(f"missing predictive-world feature object {key}")
        if verify_hash is not None and file_sha256(path) != verify_hash:
            raise ValueError("feature object content hash mismatch")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            values = handle.get_tensor("patches")
        if metadata.get("format") != FEATURE_FORMAT or metadata.get("cache_identity") != self.identity:
            raise ValueError("feature object provenance/format mismatch")
        height, width = (int(value) for value in metadata["grid_hw"].split(","))
        if values.ndim != 2 or values.shape[0] != height * width or not torch.isfinite(values).all():
            raise ValueError("feature object tensor is corrupt")
        return CachedPatchField(
            key=key,
            values=values,
            grid_hw=(height, width),
            actual_time=float(metadata["actual_time"]),
            camera_id=metadata["camera_id"],
            frame_id=metadata["frame_id"],
        )

    def publish(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        index_binding: Mapping[str, Any],
        feature_records_sha256: str,
    ) -> Path:
        """Publish only references to complete immutable objects; never overwrite a release."""

        normalized = []
        seen: set[str] = set()
        for record in records:
            sample_id, group, key = str(record["sample_id"]), int(record["group"]), str(record["key"])
            row_key = f"{sample_id}:{group}:{record.get('camera_id', '')}"
            if row_key in seen:
                raise ValueError(f"duplicate feature manifest row {row_key}")
            seen.add(row_key)
            field = self.get(key)
            supplied_camera = str(record.get("camera_id", ""))
            supplied_frame = str(record.get("frame_id", ""))
            if supplied_camera != field.camera_id or supplied_frame != field.frame_id:
                raise ValueError("feature manifest record disagrees with cached frame identity")
            actual_time = float(record.get("actual_time", field.actual_time))
            if not np.isfinite(actual_time):
                raise ValueError("feature manifest actual_time must be finite")
            normalized.append({
                "sample_id": sample_id,
                "group": group,
                "camera_id": field.camera_id,
                "frame_id": field.frame_id,
                "key": key,
                "object_sha256": file_sha256(self.objects / f"{key}.safetensors"),
                "actual_time": actual_time,
                "grid_hw": list(field.grid_hw),
            })
        if not normalized:
            raise ValueError("feature manifest cannot be empty")
        normalized = sorted(
            normalized,
            key=lambda row: (row["sample_id"], row["group"], row["camera_id"]),
        )
        binding = dict(index_binding)
        if not binding or any("REQUIRED_" in str(value) for value in binding.values()):
            raise ValueError("feature manifest requires a complete immutable index binding")
        records_digest = str(feature_records_sha256)
        if len(records_digest) != 64:
            raise ValueError("feature manifest requires the feature-records SHA-256")
        mapping_sha256 = canonical_sha256({"records": normalized})
        payload = {
            "format": FEATURE_FORMAT,
            "cache_identity": self.identity,
            "provenance": self.provenance,
            "index_binding": binding,
            "feature_records_sha256": records_digest,
            "mapping_sha256": mapping_sha256,
            "records": normalized,
        }
        payload["manifest_identity"] = canonical_sha256({
            "cache_identity": self.identity,
            "index_binding": binding,
            "feature_records_sha256": records_digest,
            "mapping_sha256": mapping_sha256,
        })
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            if json.loads(self.manifest_path.read_text(encoding="utf-8")) != payload:
                raise ValueError("immutable feature manifest already exists with different content")
            return self.manifest_path
        temporary = self.root / f".{self.manifest_path.name}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)
        return self.manifest_path


class LatentCache:
    """Immutable sample-keyed raw latent cache bound to a frozen reference ID."""

    def __init__(
        self,
        path: str | Path,
        *,
        reference_id: str,
        latent_shape: Sequence[int],
        input_fingerprint: str,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.reference_id = str(reference_id)
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.input_fingerprint = str(input_fingerprint)
        if (
            not self.reference_id or "REQUIRED_" in self.reference_id
            or not self.input_fingerprint or "REQUIRED_" in self.input_fingerprint
            or len(self.latent_shape) != 3
        ):
            raise ValueError("latent cache requires reference/input IDs and [G,S,D] shape")

    def publish(self, sample_ids: Sequence[str], values: Tensor, valid: Tensor) -> Path:
        from safetensors.torch import save_file

        ids = tuple(str(value) for value in sample_ids)
        if len(ids) != len(set(ids)) or values.shape != (len(ids), *self.latent_shape):
            raise ValueError("latent cache IDs/shape mismatch")
        if valid.shape != values.shape[:2] or not torch.isfinite(values[valid[:, :, None, None].expand_as(values)]).all():
            raise ValueError("latent cache mask or valid values are invalid")
        metadata = {
            "format": LATENT_FORMAT,
            "reference_id": self.reference_id,
            "input_fingerprint": self.input_fingerprint,
            "latent_shape": json.dumps(self.latent_shape),
            "sample_ids": json.dumps(ids),
        }
        if self.path.exists():
            raise FileExistsError(f"latent cache already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        save_file(
            {"latent_raw": values.detach().cpu().contiguous(), "group_valid": valid.detach().cpu().bool()},
            str(temporary), metadata=metadata,
        )
        temporary.replace(self.path)
        return self.path

    def load(self) -> tuple[tuple[str, ...], Tensor, Tensor]:
        from safetensors import safe_open

        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            values = handle.get_tensor("latent_raw")
            valid = handle.get_tensor("group_valid")
        if (
            metadata.get("format") != LATENT_FORMAT
            or metadata.get("reference_id") != self.reference_id
            or metadata.get("input_fingerprint") != self.input_fingerprint
            or tuple(json.loads(metadata.get("latent_shape", "[]"))) != self.latent_shape
        ):
            raise ValueError("latent cache format/reference/shape mismatch")
        ids = tuple(json.loads(metadata["sample_ids"]))
        if values.shape != (len(ids), *self.latent_shape) or valid.shape != values.shape[:2]:
            raise ValueError("latent cache tensor metadata mismatch")
        return ids, values, valid.bool()


__all__ = [
    "CachedPatchField",
    "FEATURE_FORMAT",
    "PACKED_FEATURE_FORMAT",
    "FEATURE_RECORDS_FORMAT",
    "LATENT_FORMAT",
    "LatentCache",
    "PackedFeatureStore",
    "RaggedFeatureStore",
    "canonical_sha256",
    "feature_identity",
    "feature_records_metadata_path",
    "file_sha256",
    "verify_predictive_world_feature_manifest",
    "verify_predictive_world_feature_records",
]
