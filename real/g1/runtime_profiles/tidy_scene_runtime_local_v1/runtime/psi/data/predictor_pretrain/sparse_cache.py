"""Sparse sample-id keyed future-field cache with atomic publication."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from psi.models.action_aligned_vlm_features import VLM_TARGET_DIM, VLM_TARGET_TOKENS


SPARSE_CACHE_VERSION = "action_aligned_sparse_endpoint_v1"
SPARSE_CACHE_BUILD_VERSION = "action_aligned_sparse_endpoint_build_v1"
SPARSE_CACHE_COMPOSITE_VERSION = "action_aligned_sparse_endpoint_composite_v1"


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_identity(
    sample_ids: Sequence[str],
    target_timestamps: np.ndarray,
    provenance: Mapping[str, Any],
) -> str:
    """Hash every input that determines the row-to-feature mapping."""

    digest = hashlib.sha256()
    for sample_id in sample_ids:
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update(np.asarray(target_timestamps, dtype="<f8").tobytes())
    digest.update(
        json.dumps(
            provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return digest.hexdigest()


class SparseEndpointCacheWriter:
    """Stream a sparse cache with durable, fail-closed resume checkpoints.

    The unpublished array lives in a stable build directory.  A checkpoint
    first flushes/fsyncs feature data and only then atomically advances the
    committed-row bitmap, so a killed job may redo rows but can never trust
    rows whose feature bytes were not durable.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        sample_ids: Sequence[str],
        target_timestamps: Sequence[float],
        provenance: Mapping[str, Any],
        resume: bool = False,
    ) -> None:
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.ids = [str(value) for value in sample_ids]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("sparse cache sample_ids must be unique")
        self.timestamps = np.asarray(target_timestamps, dtype=np.float64)
        if self.timestamps.shape != (len(self.ids),) or not np.isfinite(self.timestamps).all():
            raise ValueError("target_timestamps must be one finite value per sample")
        self.provenance = dict(provenance)
        self.shape = (len(self.ids), VLM_TARGET_TOKENS, VLM_TARGET_DIM)
        self.identity = _build_identity(self.ids, self.timestamps, self.provenance)
        self.build_dir = self.output / ".sparse_endpoint_build"
        self.array_tmp = self.build_dir / "endpoint_fields.npy"
        self.progress_path = self.build_dir / "committed.npy"
        self.metadata_path = self.build_dir / "build.json"
        self.lock_path = self.output / ".sparse_endpoint_build.lock"
        self._lock_handle = self.lock_path.open("a+b")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_handle.close()
            raise RuntimeError(
                f"another sparse cache builder holds {self.lock_path}"
            ) from error

        published_manifest = self.output / "sparse_endpoint_manifest.json"
        if published_manifest.is_file():
            try:
                published = json.loads(published_manifest.read_text(encoding="utf-8"))
                array_path = self.output / str(published["array_file"])
                index_path = self.output / str(published["index_file"])
                published_matches = (
                    published.get("format_version") == SPARSE_CACHE_VERSION
                    and published.get("build_identity_sha256") == self.identity
                    and published.get("shape") == list(self.shape)
                    and published.get("dtype") == "float16"
                    and array_path.is_file()
                    and index_path.is_file()
                    and array_path.stat().st_size == published.get("array_nbytes")
                    and index_path.stat().st_size == published.get("index_nbytes")
                )
            except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                published_matches = False
            if published_matches:
                self.completed_manifest: Path | None = published_manifest
                self.fields = None
                self.written = np.ones(len(self.ids), dtype=bool)
                self._release_lock()
                return
        self.completed_manifest = None

        expected_metadata = {
            "format_version": SPARSE_CACHE_BUILD_VERSION,
            "identity_sha256": self.identity,
            "num_samples": len(self.ids),
            "shape": list(self.shape),
            "dtype": "float16",
        }
        try:
            if self.build_dir.exists():
                if not resume:
                    raise RuntimeError(
                        f"unfinished sparse cache exists at {self.build_dir}; "
                        "enable resume or remove it explicitly"
                    )
                try:
                    actual_metadata = json.loads(
                        self.metadata_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"invalid sparse cache build metadata: {self.metadata_path}"
                    ) from error
                if actual_metadata != expected_metadata:
                    raise RuntimeError(
                        "refusing to resume an incompatible sparse cache build; "
                        f"expected identity {self.identity}, got "
                        f"{actual_metadata.get('identity_sha256')}"
                    )
                self.fields = np.lib.format.open_memmap(self.array_tmp, mode="r+")
                if self.fields.shape != self.shape or self.fields.dtype != np.float16:
                    raise RuntimeError("resumable sparse cache array metadata mismatch")
                committed = np.load(self.progress_path, allow_pickle=False)
                if committed.shape != (len(self.ids),) or committed.dtype != np.bool_:
                    raise RuntimeError("resumable sparse cache progress metadata mismatch")
                self.written = committed.copy()
            else:
                self.build_dir.mkdir()
                metadata_tmp = self.build_dir / ".build.json.tmp"
                metadata_tmp.write_text(
                    json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                metadata_tmp.replace(self.metadata_path)
                self.fields = np.lib.format.open_memmap(
                    self.array_tmp,
                    mode="w+",
                    dtype=np.float16,
                    shape=self.shape,
                )
                self.written = np.zeros(len(self.ids), dtype=bool)
                self.checkpoint()
                _fsync_directory(self.build_dir)
        except BaseException:
            self._release_lock()
            raise

    @property
    def num_written(self) -> int:
        return int(self.written.sum())

    def missing_offsets(self) -> np.ndarray:
        return np.flatnonzero(~self.written)

    def write(self, offsets: Sequence[int], fields: np.ndarray) -> None:
        if self.completed_manifest is not None:
            raise RuntimeError("cannot write to an already-published sparse cache")
        offsets_array = np.asarray(offsets, dtype=np.int64)
        values = np.asarray(fields)
        expected = (len(offsets_array), VLM_TARGET_TOKENS, VLM_TARGET_DIM)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"cache batch must be finite with shape {expected}")
        if (
            (offsets_array < 0).any()
            or (offsets_array >= len(self.ids)).any()
            or len(np.unique(offsets_array)) != len(offsets_array)
            or self.written[offsets_array].any()
        ):
            raise ValueError("cache offsets are out of range, repeated or already written")
        self.fields[offsets_array] = values.astype(np.float16, copy=False)
        self.written[offsets_array] = True

    def invalidate(self, offsets: Sequence[int]) -> int:
        """Mark previously committed rows for recomputation before publication."""

        if self.completed_manifest is not None:
            raise RuntimeError("cannot invalidate an already-published sparse cache")
        offsets_array = np.asarray(offsets, dtype=np.int64)
        if offsets_array.size == 0:
            return 0
        if (offsets_array < 0).any() or (offsets_array >= len(self.ids)).any():
            raise ValueError("cache invalidation offsets are out of range")
        offsets_array = np.unique(offsets_array)
        count = int(self.written[offsets_array].sum())
        self.written[offsets_array] = False
        return count

    def checkpoint(self) -> None:
        """Durably commit all rows written so far."""

        if self.completed_manifest is not None:
            return
        assert self.fields is not None
        self.fields.flush()
        _fsync_file(self.array_tmp)
        progress_tmp = self.build_dir / ".committed.npy.tmp"
        with progress_tmp.open("wb") as handle:
            np.save(handle, self.written, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        progress_tmp.replace(self.progress_path)
        _fsync_directory(self.build_dir)

    def close(self) -> None:
        self._close_fields()
        self._release_lock()

    def _close_fields(self) -> None:
        mmap = getattr(getattr(self, "fields", None), "_mmap", None)
        if mmap is not None:
            mmap.close()

    def _release_lock(self) -> None:
        handle = getattr(self, "_lock_handle", None)
        if handle is not None and not handle.closed:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def publish(self) -> Path:
        if self.completed_manifest is not None:
            return self.completed_manifest
        if not self.written.all():
            raise RuntimeError(f"cannot publish sparse cache with {(~self.written).sum()} missing rows")
        self.checkpoint()
        self._close_fields()
        nonce = uuid.uuid4().hex
        self.array_name = f"endpoint_fields.{nonce}.npy"
        self.index_name = f"endpoint_index.{nonce}.jsonl"
        array_path = self.output / self.array_name
        index_path = self.output / self.index_name
        index_tmp = self.output / f".{self.index_name}.tmp"
        # A hard link publishes the immutable bytes without copying a cache
        # that may be tens of GiB, while retaining resumability until the
        # manifest pointer itself has been atomically replaced.
        os.link(self.array_tmp, array_path)
        with index_tmp.open("w", encoding="utf-8") as handle:
            for offset, (sample_id, timestamp) in enumerate(zip(self.ids, self.timestamps.tolist())):
                handle.write(
                    json.dumps(
                        {"sample_id": sample_id, "offset": offset, "target_timestamp": timestamp},
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        index_tmp.replace(index_path)
        expected = self.shape
        manifest = {
            "format_version": SPARSE_CACHE_VERSION,
            "num_samples": len(self.ids),
            "shape": list(expected),
            "dtype": "float16",
            "array_file": self.array_name,
            "array_sha256": _sha256_file(array_path),
            "array_nbytes": array_path.stat().st_size,
            "index_file": self.index_name,
            "index_sha256": _sha256_file(index_path),
            "index_nbytes": index_path.stat().st_size,
            "build_identity_sha256": self.identity,
            "provenance": self.provenance,
        }
        manifest_path = self.output / "sparse_endpoint_manifest.json"
        manifest_tmp = self.output / f".{manifest_path.name}.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _fsync_file(manifest_tmp)
        manifest_tmp.replace(manifest_path)
        _fsync_directory(self.output)
        shutil.rmtree(self.build_dir)
        self._release_lock()
        return manifest_path


def write_sparse_endpoint_cache(
    output_dir: str | Path,
    *,
    sample_ids: Sequence[str],
    target_timestamps: Sequence[float],
    fields: np.ndarray,
    provenance: Mapping[str, Any],
) -> Path:
    """Write arrays first and atomically publish their content-addressed manifest."""

    values = np.asarray(fields)
    expected = (len(sample_ids), VLM_TARGET_TOKENS, VLM_TARGET_DIM)
    if values.shape != expected:
        raise ValueError(f"cache fields must have shape {expected}")
    writer = SparseEndpointCacheWriter(
        output_dir,
        sample_ids=sample_ids,
        target_timestamps=target_timestamps,
        provenance=provenance,
    )
    writer.write(range(len(sample_ids)), values)
    return writer.publish()


def publish_sparse_endpoint_composite(
    output_dir: str | Path,
    *,
    sample_ids: Sequence[str],
    target_timestamps: Sequence[float],
    provenance: Mapping[str, Any],
    shard_manifests: Sequence[str | Path],
) -> Path:
    """Atomically publish one logical source cache backed by immutable shards.

    Shard builders never share an mmap or progress bitmap.  This coordinator
    verifies every shard (including its content hashes), proves that the
    stable source partition is complete and disjoint, and only then replaces
    the source-level manifest consumed by training.
    """

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids = [str(value) for value in sample_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("composite cache sample_ids must be unique")
    timestamps = np.asarray(target_timestamps, dtype=np.float64)
    if timestamps.shape != (len(ids),) or not np.isfinite(timestamps).all():
        raise ValueError("composite cache needs one finite timestamp per sample")
    manifests = [Path(value).resolve() for value in shard_manifests]
    if not manifests:
        raise ValueError("composite cache requires at least one shard")

    lock_path = output / ".sparse_endpoint_build.lock"
    lock_handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another sparse cache publisher holds {lock_path}") from error
        if (output / ".sparse_endpoint_build").exists():
            raise RuntimeError(
                "cannot publish a composite over an unfinished monolithic cache build"
            )

        expected_provenance = dict(provenance)
        shard_descriptors = []
        expected_shards = len(manifests)
        seen_indices: set[int] = set()
        for manifest_path in manifests:
            cache = SparseEndpointCache(manifest_path, verify_hashes=True)
            if cache.manifest.get("format_version") != SPARSE_CACHE_VERSION:
                raise ValueError("composite inputs must be regular sparse cache shards")
            actual_provenance = dict(cache.manifest.get("provenance", {}))
            shard = actual_provenance.pop("cache_shard", None)
            if actual_provenance != expected_provenance or not isinstance(shard, Mapping):
                raise ValueError("cache shard provenance differs from the composite release")
            shard_index = int(shard.get("index", -1))
            num_shards = int(shard.get("count", -1))
            start = int(shard.get("start", -1))
            stop = int(shard.get("stop", -1))
            expected_start = len(ids) * shard_index // expected_shards
            expected_stop = len(ids) * (shard_index + 1) // expected_shards
            if (
                num_shards != expected_shards
                or not 0 <= shard_index < expected_shards
                or shard_index in seen_indices
                or int(shard.get("source_num_samples", -1)) != len(ids)
                or start != expected_start
                or stop != expected_stop
                or not str(shard.get("selection_order", ""))
            ):
                raise ValueError("cache shard partition metadata is invalid")
            seen_indices.add(shard_index)
            expected_ids = ids[start:stop]
            if len(cache.index) != len(expected_ids) or set(cache.index) != set(expected_ids):
                raise ValueError("cache shard sample ids do not equal its stable partition")
            expected_timestamps = dict(zip(expected_ids, timestamps[start:stop].tolist()))
            for sample_id, (_offset, timestamp) in cache.index.items():
                if abs(float(timestamp) - expected_timestamps[sample_id]) > 1e-9:
                    raise ValueError("cache shard target timestamp differs from anchor index")
            shard_descriptors.append(
                {
                    "shard_index": shard_index,
                    "manifest_file": os.path.relpath(manifest_path, output),
                    "manifest_sha256": _sha256_file(manifest_path),
                    "build_identity_sha256": cache.manifest.get("build_identity_sha256"),
                    "num_samples": len(cache.index),
                }
            )
        if seen_indices != set(range(expected_shards)):
            raise ValueError("composite cache is missing one or more shard indices")
        shard_descriptors.sort(key=lambda value: int(value["shard_index"]))
        manifest = {
            "format_version": SPARSE_CACHE_COMPOSITE_VERSION,
            "storage_layout": "source_partitioned_sparse_manifests_v1",
            "num_samples": len(ids),
            "shape": [len(ids), VLM_TARGET_TOKENS, VLM_TARGET_DIM],
            "dtype": "float16",
            "build_identity_sha256": _build_identity(ids, timestamps, expected_provenance),
            "provenance": expected_provenance,
            "shards": shard_descriptors,
        }
        manifest_path = output / "sparse_endpoint_manifest.json"
        manifest_tmp = output / f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _fsync_file(manifest_tmp)
        manifest_tmp.replace(manifest_path)
        _fsync_directory(output)
        return manifest_path
    finally:
        if not lock_handle.closed:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


class SparseEndpointCache:
    def __init__(self, manifest_path: str | Path, *, verify_hashes: bool = True) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        format_version = self.manifest.get("format_version")
        if format_version == SPARSE_CACHE_COMPOSITE_VERSION:
            self._load_composite(verify_hashes=verify_hashes)
            return
        if format_version != SPARSE_CACHE_VERSION:
            raise ValueError("unsupported sparse endpoint cache format")
        self._composite_shards = None
        root = self.manifest_path.parent
        self.array_path = root / str(self.manifest["array_file"])
        self.index_path = root / str(self.manifest["index_file"])
        if verify_hashes:
            if _sha256_file(self.array_path) != self.manifest.get("array_sha256"):
                raise ValueError("sparse endpoint array hash mismatch")
            if _sha256_file(self.index_path) != self.manifest.get("index_sha256"):
                raise ValueError("sparse endpoint index hash mismatch")
        self.fields = np.load(self.array_path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(value) for value in self.manifest["shape"])
        if self.fields.shape != expected_shape or self.fields.dtype != np.float16:
            raise ValueError("sparse endpoint array metadata mismatch")
        self.index: dict[str, tuple[int, float]] = {}
        seen_offsets: set[int] = set()
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in self.index:
                raise ValueError("duplicate sample_id in sparse endpoint index")
            offset, timestamp = int(row["offset"]), float(row["target_timestamp"])
            if (
                offset < 0
                or offset >= int(self.manifest["num_samples"])
                or offset in seen_offsets
                or not np.isfinite(timestamp)
            ):
                raise ValueError("sparse endpoint index has invalid offset/timestamp")
            seen_offsets.add(offset)
            self.index[sample_id] = (offset, timestamp)
        if len(self.index) != int(self.manifest["num_samples"]):
            raise ValueError("sparse endpoint index length mismatch")
        if len(seen_offsets) != int(self.manifest["num_samples"]):
            raise ValueError("sparse endpoint index offsets are not a bijection")

    def _load_composite(self, *, verify_hashes: bool) -> None:
        expected_shape = tuple(int(value) for value in self.manifest.get("shape", ()))
        if (
            len(expected_shape) != 3
            or expected_shape[1:] != (VLM_TARGET_TOKENS, VLM_TARGET_DIM)
            or self.manifest.get("dtype") != "float16"
        ):
            raise ValueError("composite sparse endpoint metadata mismatch")
        descriptors = self.manifest.get("shards")
        if not isinstance(descriptors, list) or not descriptors:
            raise ValueError("composite sparse endpoint cache has no shards")
        root = self.manifest_path.parent
        self.array_path = None
        self.index_path = None
        self._composite_shards: list[SparseEndpointCache] = []
        self.fields = []
        self.index: dict[str, tuple[int, int, float]] = {}
        seen_indices = set()
        selection_order = None
        for descriptor in descriptors:
            shard_index = int(descriptor.get("shard_index", -1))
            if shard_index in seen_indices or shard_index != len(self._composite_shards):
                raise ValueError("composite shard descriptors are not contiguous and ordered")
            seen_indices.add(shard_index)
            shard_manifest = (root / str(descriptor["manifest_file"])).resolve()
            if _sha256_file(shard_manifest) != descriptor.get("manifest_sha256"):
                raise ValueError("composite shard manifest hash mismatch")
            shard_cache = SparseEndpointCache(shard_manifest, verify_hashes=verify_hashes)
            if shard_cache.manifest.get("format_version") != SPARSE_CACHE_VERSION:
                raise ValueError("nested composite sparse caches are unsupported")
            shard_provenance = dict(shard_cache.manifest.get("provenance", {}))
            shard_spec = shard_provenance.pop("cache_shard", None)
            shard_order = shard_spec.get("selection_order") if isinstance(shard_spec, Mapping) else None
            expected_start = expected_shape[0] * shard_index // len(descriptors)
            expected_stop = expected_shape[0] * (shard_index + 1) // len(descriptors)
            if (
                shard_provenance != self.manifest.get("provenance", {})
                or not isinstance(shard_spec, Mapping)
                or int(shard_spec.get("index", -1)) != shard_index
                or int(shard_spec.get("count", -1)) != len(descriptors)
                or int(shard_spec.get("start", -1)) != expected_start
                or int(shard_spec.get("stop", -1)) != expected_stop
                or int(shard_spec.get("source_num_samples", -1)) != expected_shape[0]
                or not str(shard_order or "")
                or (selection_order is not None and shard_order != selection_order)
                or len(shard_cache.index) != int(descriptor.get("num_samples", -1))
                or shard_cache.manifest.get("build_identity_sha256")
                != descriptor.get("build_identity_sha256")
            ):
                raise ValueError("composite shard descriptor metadata mismatch")
            selection_order = shard_order
            self._composite_shards.append(shard_cache)
            self.fields.append(shard_cache.fields)
            for sample_id, (offset, timestamp) in shard_cache.index.items():
                if sample_id in self.index:
                    raise ValueError("duplicate sample_id across composite cache shards")
                self.index[sample_id] = (shard_index, int(offset), float(timestamp))
        if len(self.index) != int(self.manifest.get("num_samples", -1)):
            raise ValueError("composite sparse endpoint index length mismatch")
        if expected_shape[0] != len(self.index):
            raise ValueError("composite sparse endpoint logical shape mismatch")

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self.index

    def get(self, sample_id: str) -> tuple[np.ndarray, float]:
        try:
            location = self.index[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"sample_id absent from sparse endpoint cache: {sample_id}") from error
        if self._composite_shards is not None:
            shard_index, offset, timestamp = location
            return np.asarray(self._composite_shards[shard_index].fields[offset]), timestamp
        offset, timestamp = location
        return np.asarray(self.fields[offset]), timestamp


__all__ = [
    "SPARSE_CACHE_VERSION",
    "SPARSE_CACHE_COMPOSITE_VERSION",
    "SparseEndpointCache",
    "SparseEndpointCacheWriter",
    "publish_sparse_endpoint_composite",
    "write_sparse_endpoint_cache",
]
