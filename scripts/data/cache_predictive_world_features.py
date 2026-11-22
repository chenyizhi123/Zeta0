#!/usr/bin/env python3
"""Build a resumable, row-sharded frozen-Qwen future feature cache.

The cache is deliberately indexed by the immutable predictive-world row
ordinal.  A torchrun rank owns one contiguous range and one NPY file, so a
production release creates tens of large files instead of millions of tiny
objects.  Committed progress is journaled separately and can be resumed after
pre-emption.  When an exactly compatible +0.6 s endpoint exists in the legacy
pretraining cache it is copied byte-for-byte; earlier future groups are built
incrementally with the same frozen Qwen encoder.
"""

from __future__ import annotations

import argparse
import bisect
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any, Iterable, Mapping
import uuid

import numpy as np
import torch
import torch.distributed as dist

from psi.data.lerobot.action_aligned_cache import extract_endpoint_vlm_features
from psi.data.predictive_world.cache import (
    PACKED_FEATURE_FORMAT,
    canonical_sha256,
    file_sha256,
    verify_predictive_world_feature_manifest,
)
from psi.data.predictive_world.index import (
    INDEX_BINDING_FORMAT,
    predictive_world_lookup_path,
    verify_predictive_world_index,
)
from psi.data.predictive_world.video import decode_frame_job, initialize_decode_worker


FEATURE_SHAPE = (16, 2048)
FrameKey = tuple[str, int]


class _RecentFrameFeatures:
    """Bounded LRU for immutable frozen-Qwen features within one cache shard."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("recent feature capacity must be positive")
        self.capacity = int(capacity)
        self.values: OrderedDict[FrameKey, np.ndarray] = OrderedDict()

    def get(self, key: FrameKey) -> np.ndarray | None:
        value = self.values.pop(key, None)
        if value is not None:
            self.values[key] = value
        return value

    def put(self, key: FrameKey, value: np.ndarray) -> None:
        self.values.pop(key, None)
        # Do not keep a reference to a whole Qwen batch just for one row.
        self.values[key] = np.asarray(value, dtype=np.float16).copy()
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-artifact-root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-rows", type=int, default=1024)
    parser.add_argument("--recent-feature-capacity", type=int, default=1024)
    parser.add_argument("--decode-workers", type=int, default=8,
                        help="CPU decoder processes (shared total with --decode-pool); 0 selects synchronous decoding")
    parser.add_argument("--decode-pool", help="Private node-local Unix socket for a shared CPU decoder pool")
    parser.add_argument("--decode-batch-size", type=int, default=8,
                        help="Unique frames per CPU work item, independent of Qwen batch size")
    parser.add_argument("--decode-prefetch-batches", type=int, default=0,
                        help="Bound on submitted decode batches; 0 selects twice the worker count")
    parser.add_argument("--decoder-cache-capacity", type=int, default=8)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _initialize_distributed(enabled: bool) -> tuple[int, int, torch.device]:
    if not enabled:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return 0, 1, device
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError(f"--distributed requires torchrun variables: {missing}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("distributed feature caching requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    return rank, world_size, torch.device("cuda", local_rank)


def _barrier(enabled: bool) -> None:
    if enabled:
        dist.barrier()


def _index_binding(index: Path, *, verify: bool) -> tuple[dict[str, Any], int, int]:
    meta_path = index.with_suffix(index.suffix + ".meta.json")
    if verify:
        meta = verify_predictive_world_index(index)
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lookup = predictive_world_lookup_path(index)
    temporal = meta.get("temporal_abi", {})
    groups = len(temporal.get("future_group_ends_s", ()))
    rows = int(meta.get("rows", 0))
    if rows <= 0 or groups <= 0 or not lookup.is_file():
        raise ValueError("predictive-world index publication is incomplete")
    return ({
        "format": INDEX_BINDING_FORMAT,
        "semantic_identity": str(meta["semantic_identity"]),
        "index_sha256": str(meta["index_sha256"]),
        "lookup_sha256": str(meta["lookup_sha256"]),
        "temporal_abi_sha256": canonical_sha256(temporal),
    }, rows, groups)


def _decode_frames(
    path: str, frame_indices: Iterable[int], backend: str = "opencv"
) -> list[torch.Tensor]:
    indices = [int(index) for index in frame_indices]
    decoded, _ = decode_frame_job([(path, index) for index in indices], backend)
    return [torch.from_numpy(decoded[path, index]).float().div_(255) for index in indices]


def _decode_frame(path: str, frame_index: int, backend: str = "opencv") -> torch.Tensor:
    return _decode_frames(path, [frame_index], backend)[0]


def _decode_frame_batch(
    keys: Iterable[FrameKey], *, backend: str
) -> dict[FrameKey, torch.Tensor]:
    """Decode distinct requested frames in one backend call per source video."""

    by_path: OrderedDict[str, list[FrameKey]] = OrderedDict()
    for path, frame_index in keys:
        by_path.setdefault(path, []).append((path, frame_index))
    decoded: dict[FrameKey, torch.Tensor] = {}
    for path, requests in by_path.items():
        frames = _decode_frames(path, [frame_index for _, frame_index in requests], backend)
        if len(frames) != len(requests):
            raise RuntimeError("video decoder returned a different number of frames than requested")
        decoded.update(zip(requests, frames))
    return decoded


class _LegacyEndpointCache:
    """Read legacy endpoint arrays through their audited global cache position."""

    def __init__(self, root: Path, provenance: Mapping[str, Any]) -> None:
        self.root = root.resolve()
        self.connection = sqlite3.connect(
            f"file:{self.root / 'anchor_index.sqlite'}?mode=ro", uri=True
        )
        self.expected_teacher = str(provenance.get("legacy_teacher_sha256", ""))
        self.sources: dict[str, tuple[list[int], list[tuple[int, int, Path]]]] = {}
        self.arrays: dict[Path, np.ndarray] = {}
        for manifest_path in sorted((self.root / "caches").glob("*/sparse_endpoint_manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_id = str(manifest.get("provenance", {}).get("source_id", ""))
            if not source_id:
                continue
            descriptors: list[tuple[int, int, Path]] = []
            if manifest.get("format_version") == "action_aligned_sparse_endpoint_composite_v1":
                for shard in manifest.get("shards", ()):
                    child_path = (manifest_path.parent / str(shard["manifest_file"])).resolve()
                    child = json.loads(child_path.read_text(encoding="utf-8"))
                    cache_shard = child.get("provenance", {}).get("cache_shard", {})
                    descriptors.append((
                        int(cache_shard["start"]),
                        int(cache_shard["stop"]),
                        (child_path.parent / str(child["array_file"])).resolve(),
                    ))
                    self._validate_provenance(child.get("provenance", {}))
            else:
                descriptors.append((
                    0,
                    int(manifest["num_samples"]),
                    (manifest_path.parent / str(manifest["array_file"])).resolve(),
                ))
                self._validate_provenance(manifest.get("provenance", {}))
            descriptors.sort()
            starts = [value[0] for value in descriptors]
            if not descriptors or starts[0] != 0 or any(
                left[1] != right[0] for left, right in zip(descriptors, descriptors[1:])
            ):
                raise ValueError(f"legacy endpoint cache for {source_id!r} is not contiguous")
            self.sources[source_id] = ([value[1] for value in descriptors], descriptors)

    def _validate_provenance(self, value: Mapping[str, Any]) -> None:
        teacher = str(value.get("teacher_sha256", ""))
        preprocessing = value.get("image_preprocessing", {})
        if self.expected_teacher and teacher != self.expected_teacher:
            raise ValueError("legacy endpoint cache uses a different frozen Qwen checkpoint")
        if not isinstance(preprocessing, Mapping) or preprocessing.get("version") != "aspect_v1":
            raise ValueError("legacy endpoint cache preprocessing is incompatible")

    def get(self, source_id: str, sample_id: str) -> np.ndarray | None:
        source = self.sources.get(str(source_id))
        if source is None:
            return None
        row = self.connection.execute(
            "SELECT source_id, cache_position FROM anchors WHERE sample_id=?",
            (str(sample_id),),
        ).fetchone()
        if row is None or str(row[0]) != str(source_id) or row[1] is None:
            return None
        position = int(row[1])
        stops, descriptors = source
        shard_index = bisect.bisect_right(stops, position)
        if shard_index >= len(descriptors):
            raise ValueError("legacy endpoint cache position is out of range")
        start, stop, array_path = descriptors[shard_index]
        if not start <= position < stop:
            raise ValueError("legacy endpoint cache shard lookup is inconsistent")
        if array_path not in self.arrays:
            value = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if value.dtype != np.float16 or tuple(value.shape[1:]) != FEATURE_SHAPE:
                raise ValueError("legacy endpoint tensor ABI mismatch")
            self.arrays[array_path] = value
        return np.asarray(self.arrays[array_path][position - start])

    def close(self) -> None:
        self.connection.close()


def _iter_rows(lookup: Path, start: int, stop: int) -> Iterable[tuple[int, dict[str, Any]]]:
    connection = sqlite3.connect(f"file:{lookup}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            "SELECT ordinal, payload FROM records WHERE ordinal>=? AND ordinal<? ORDER BY ordinal",
            (int(start), int(stop)),
        )
        for ordinal, payload in cursor:
            yield int(ordinal), json.loads(payload)
    finally:
        connection.close()


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable cache metadata exists with different content: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _rank_paths(output: Path, rank: int, world_size: int) -> tuple[Path, Path, Path]:
    stem = f"rank-{rank:05d}-of-{world_size:05d}"
    return (
        output / "shards" / f"{stem}.npy",
        output / "work" / f"{stem}.progress.npy",
        output / "shards" / f"{stem}.json",
    )


def _validate_rank_sidecar(
    path: Path,
    *,
    rank: int,
    world_size: int,
    start: int,
    stop: int,
    cache_identity: str,
    index_binding: Mapping[str, Any],
    verify_hash: bool = True,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "rank": rank,
        "world_size": world_size,
        "start": start,
        "stop": stop,
        "cache_identity": cache_identity,
        "index_binding": dict(index_binding),
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise ValueError(f"completed feature shard has a different identity: {path}")
    array_path = path.parent / str(value.get("file", ""))
    if not array_path.is_file() or array_path.stat().st_size != int(value.get("nbytes", -1)):
        raise ValueError(f"completed feature shard is corrupt: {array_path}")
    if verify_hash and file_sha256(array_path) != value.get("sha256"):
        raise ValueError(f"completed feature shard hash mismatch: {array_path}")
    return value


def main() -> None:
    args = _parser().parse_args()
    if (
        args.batch_size <= 0
        or args.checkpoint_rows <= 0
        or args.recent_feature_capacity <= 0
        or args.decode_workers < 0
        or args.decode_batch_size <= 0
        or args.decode_prefetch_batches < 0
        or args.decoder_cache_capacity <= 0
    ):
        raise ValueError("batch sizes and capacities must be positive; workers/prefetch must be nonnegative")
    # Validate the actual encoder/runtime even when reusing a completed cache.
    # Pure metadata estimate/verify without --model remains a read-only audit.
    if args.model and not args.estimate_only:
        from psi.data.predictive_world.provenance import verify_qwen_provenance
        verify_qwen_provenance(args.model, json.loads(Path(args.provenance).read_text()))
    rank, world_size, device = _initialize_distributed(args.distributed)
    try:
        index = Path(args.index).expanduser().resolve()
        provenance_path = Path(args.provenance).expanduser().resolve()
        output = Path(args.output).expanduser().resolve()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        required = {"encoder_id", "processor_id", "image_preprocessing", "feature_layer", "dtype"}
        if (
            not isinstance(provenance, dict)
            or required - provenance.keys()
            or provenance.get("image_preprocessing") != "aspect_v1"
            or provenance.get("dtype") != "float16"
        ):
            raise ValueError("feature provenance does not declare the fixed packed-feature ABI")
        # Rank zero performs the expensive full content verification once.  All
        # ranks then consume the immutable metadata and the torchrun barrier
        # propagates a failure before any shard is opened.
        if rank == 0:
            _index_binding(index, verify=True)
        _barrier(args.distributed)
        binding, rows, groups = _index_binding(index, verify=False)
        cache_identity = canonical_sha256(provenance)
        manifest_path = output / "feature_manifest.json"
        if manifest_path.is_file():
            payload = verify_predictive_world_feature_manifest(
                manifest_path, verify_shard_hashes=args.verify_only
            )
            if (
                payload.get("cache_identity") != cache_identity
                or payload.get("index_binding") != binding
                or int(payload.get("rows", -1)) != rows
                or int(payload.get("groups", -1)) != groups
            ):
                raise ValueError("existing packed feature release has a different identity")
            if rank == 0:
                print(json.dumps({
                    "event": "predictive_world_feature_cache_reused",
                    "manifest": str(manifest_path),
                    "rows": rows,
                }, sort_keys=True), flush=True)
            return
        maximum_bytes = rows * groups * FEATURE_SHAPE[0] * FEATURE_SHAPE[1] * 2
        if rank == 0:
            print(json.dumps({
                "event": "predictive_world_feature_estimate",
                "rows": rows,
                "groups": groups,
                "bytes_fp16": maximum_bytes,
                "world_size": world_size,
            }, sort_keys=True), flush=True)
        if args.estimate_only:
            return
        if args.verify_only:
            raise FileNotFoundError(f"feature manifest does not exist: {manifest_path}")
        if not args.model:
            raise ValueError("--model is required when building the feature cache")

        output.mkdir(parents=True, exist_ok=True)
        (output / "shards").mkdir(parents=True, exist_ok=True)
        (output / "work").mkdir(parents=True, exist_ok=True)
        start = rows * rank // world_size
        stop = rows * (rank + 1) // world_size
        array_path, progress_path, sidecar_path = _rank_paths(output, rank, world_size)
        completed = _validate_rank_sidecar(
            sidecar_path,
            rank=rank,
            world_size=world_size,
            start=start,
            stop=stop,
            cache_identity=cache_identity,
            index_binding=binding,
        )
        if completed is None:
            local_rows = stop - start
            expected_shape = (local_rows, groups, *FEATURE_SHAPE)
            if array_path.exists():
                features = np.load(array_path, mmap_mode="r+", allow_pickle=False)
                if features.shape != expected_shape or features.dtype != np.float16:
                    raise ValueError(f"partial feature shard has a different ABI: {array_path}")
            else:
                features = np.lib.format.open_memmap(
                    array_path, mode="w+", dtype=np.float16, shape=expected_shape
                )
            if progress_path.exists():
                progress = np.load(progress_path, mmap_mode="r+", allow_pickle=False)
                if progress.shape != (local_rows, groups) or progress.dtype != np.bool_:
                    raise ValueError("feature shard progress journal has a different ABI")
            else:
                progress = np.lib.format.open_memmap(
                    progress_path, mode="w+", dtype=np.bool_, shape=(local_rows, groups)
                )
                progress[:] = False
                progress.flush()

            legacy = None
            if args.legacy_artifact_root:
                legacy = _LegacyEndpointCache(
                    Path(args.legacy_artifact_root).expanduser(), provenance
                )
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            model = None
            processor = None
            pending: list[tuple[int, int]] = []
            image_batch: list[torch.Tensor] = []
            batch_keys: list[FrameKey] = []
            batch_slot_groups: list[list[tuple[int, int]]] = []
            batch_slots_by_key: dict[FrameKey, list[tuple[int, int]]] = {}
            batch_shape: tuple[int, ...] | None = None
            decode_queue: OrderedDict[FrameKey, list[tuple[int, int]]] = OrderedDict()
            recent_features = _RecentFrameFeatures(args.recent_feature_capacity)
            reused = 0
            encoded = 0
            recent_reused = 0
            visited_rows = 0
            started = perf_counter()
            timings = dict(decode_worker_seconds=0.0, decode_wait_seconds=0.0,
                           preprocess_encode_seconds=0.0, model_load_seconds=0.0,
                           feature_write_seconds=0.0, commit_seconds=0.0)
            committed_groups = 0
            prefetch_batches = args.decode_prefetch_batches or max(1, 2 * args.decode_workers)
            decode_jobs = deque()
            executor = None
            print(json.dumps({
                "event": "predictive_world_feature_cache_pipeline", "rank": rank,
                "decode_workers": args.decode_workers, "decode_batch_size": args.decode_batch_size,
                "decode_pool": args.decode_pool,
                "decode_prefetch_batches": prefetch_batches,
                "max_prefetched_frames": prefetch_batches * args.decode_batch_size,
                "decoder_cache_capacity": args.decoder_cache_capacity,
                "qwen_batch_size": args.batch_size,
            }, sort_keys=True), flush=True)

            def commit() -> None:
                nonlocal pending, committed_groups
                if not pending:
                    return
                tick = perf_counter()
                features.flush()
                for local_index, group_index in pending:
                    progress[local_index, group_index] = True
                progress.flush()
                committed_groups += len(pending)
                pending = []
                timings["commit_seconds"] += perf_counter() - tick

            def write_feature(slot, value) -> None:
                tick = perf_counter()
                features[slot] = value
                pending.append(slot)
                timings["feature_write_seconds"] += perf_counter() - tick

            def flush_images() -> None:
                nonlocal model, processor, image_batch, batch_keys, batch_slot_groups, encoded, batch_shape
                if not image_batch:
                    return
                if model is None:
                    tick = perf_counter()
                    model = Qwen3VLForConditionalGeneration.from_pretrained(
                        args.model,
                        local_files_only=True,
                        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                        attn_implementation="sdpa",
                    ).to(device).eval()
                    processor = AutoProcessor.from_pretrained(
                        args.model, local_files_only=True
                    )
                    timings["model_load_seconds"] += perf_counter() - tick
                tick = perf_counter()
                values = extract_endpoint_vlm_features(
                    model,
                    processor,
                    torch.stack(image_batch),
                    device=device,
                    image_preprocessing="aspect_v1",
                ).numpy()
                # The extractor returns CPU features, so this includes completed
                # GPU work, CPU preprocessing, H2D and D2H (not just kernel launch).
                timings["preprocess_encode_seconds"] += perf_counter() - tick
                for key, slots, value in zip(batch_keys, batch_slot_groups, values):
                    for slot in slots:
                        write_feature(slot, value)
                    recent_features.put(key, value)
                    del batch_slots_by_key[key]
                encoded += len(image_batch)
                image_batch = []
                batch_keys = []
                batch_slot_groups = []
                batch_shape = None

            def consume_decoded(queued, decoded) -> None:
                nonlocal batch_shape
                for key, slots in queued.items():
                    # Transport uint8, then reproduce the synchronous decoder's
                    # exact float32 [0,1] values at the consumer boundary.
                    image = torch.from_numpy(decoded[key]).float().div_(255)
                    shape = tuple(image.shape)
                    if image_batch and (shape != batch_shape or len(image_batch) >= args.batch_size):
                        flush_images()
                    batch_shape = shape
                    image_batch.append(image)
                    batch_keys.append(key)
                    batch_slot_groups.append(slots)
                    if len(image_batch) >= args.batch_size:
                        flush_images()

            def consume_decode_job() -> None:
                queued, future = decode_jobs.popleft()
                tick = perf_counter()
                decoded, worker_seconds = future.result(timeout=600)
                timings["decode_wait_seconds"] += perf_counter() - tick
                timings["decode_worker_seconds"] += worker_seconds
                consume_decoded(queued, decoded)

            def flush_decode_queue() -> None:
                if not decode_queue:
                    return
                queued = decode_queue.copy()
                decode_queue.clear()
                # Include queued/in-flight work in de-duplication before waiting
                # for a worker; later rows append slots to the same live lists.
                batch_slots_by_key.update(queued)
                backend = provenance.get("video_decoder", "opencv")
                if executor is None:
                    decoded, worker_seconds = decode_frame_job(
                        list(queued), backend, args.decoder_cache_capacity
                    )
                    timings["decode_worker_seconds"] += worker_seconds
                    timings["decode_wait_seconds"] += worker_seconds
                    consume_decoded(queued, decoded)
                else:
                    decode_jobs.append((queued, executor.submit(
                        decode_frame_job, list(queued), backend, args.decoder_cache_capacity
                    )))
                    if len(decode_jobs) >= prefetch_batches or decode_jobs[0][1].done():
                        consume_decode_job()

            def drain() -> None:
                flush_decode_queue()
                while decode_jobs:
                    consume_decode_job()
                flush_images()

            try:
                if args.decode_workers:
                    if args.decode_pool:
                        from psi.data.predictive_world.decode_pool import SharedDecodeClient
                        executor = SharedDecodeClient(
                            args.decode_pool, args.decode_workers, prefetch_batches
                        )
                    else:
                        executor = ProcessPoolExecutor(
                            max_workers=args.decode_workers,
                            mp_context=multiprocessing.get_context("spawn"),
                            initializer=initialize_decode_worker,
                        )
                lookup = predictive_world_lookup_path(index)
                legacy_group = int(provenance.get("legacy_endpoint_group", groups - 1))
                for ordinal, row in _iter_rows(lookup, start, stop):
                    local = ordinal - start
                    visited_rows += 1
                    valid = tuple(bool(value) for value in row["future_visual_valid"])
                    indices = tuple(int(value) for value in row["future_visual_indices"])
                    video_paths = row["materialization"]["video_paths"]
                    cameras = tuple(row["source_capability"].get("future_visual_keys", ()))
                    if len(cameras) != 1:
                        raise ValueError("packed cache requires exactly one audited future camera")
                    camera = cameras[0]
                    for group in range(groups):
                        if bool(progress[local, group]):
                            continue
                        if not valid[group]:
                            write_feature((local, group), 0)
                            continue
                        legacy_value = None
                        if legacy is not None and group == legacy_group:
                            base = str(row["materialization"].get("legacy_episode_sample_id", ""))
                            legacy_source = str(
                                row["materialization"].get("legacy_source_id", row["source_id"])
                            )
                            if base:
                                legacy_id = (
                                    f"{base}:v3:frame:{int(row['history_visual_indices'][-1])}"
                                    f":target:{indices[group]}"
                                )
                                legacy_value = legacy.get(legacy_source, legacy_id)
                        if legacy_value is not None:
                            write_feature((local, group), legacy_value)
                            reused += 1
                            continue
                        key = (str(video_paths[camera]), indices[group])
                        cached = recent_features.get(key)
                        if cached is not None:
                            write_feature((local, group), cached)
                            recent_reused += 1
                            continue
                        inflight = batch_slots_by_key.get(key)
                        if inflight is not None:
                            inflight.append((local, group))
                            recent_reused += 1
                            continue
                        queued = decode_queue.get(key)
                        if queued is not None:
                            queued.append((local, group))
                            recent_reused += 1
                            continue
                        decode_queue[key] = [(local, group)]
                        if len(decode_queue) >= args.decode_batch_size:
                            flush_decode_queue()
                    if visited_rows % args.checkpoint_rows == 0:
                        drain()
                        commit()
                        print(json.dumps({
                            "event": "predictive_world_feature_cache_progress",
                            "rank": rank,
                            "rows_done": visited_rows,
                            "rows_total": local_rows,
                            "legacy_reused": reused,
                            "qwen_encoded": encoded,
                            "recent_frame_reused": recent_reused,
                            "elapsed_seconds": perf_counter() - started,
                            "session_committed_groups": committed_groups,
                            **timings,
                        }, sort_keys=True), flush=True)
                drain()
                commit()
                if not bool(np.asarray(progress).all()):
                    raise RuntimeError("feature shard finished with uncommitted rows")
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                if legacy is not None:
                    legacy.close()
            print(json.dumps({
                "event": "predictive_world_feature_cache_timing", "rank": rank,
                "elapsed_seconds": perf_counter() - started,
                "session_committed_groups": committed_groups, **timings,
            }, sort_keys=True), flush=True)
            features.flush()
            shard_sha = file_sha256(array_path)
            completed = {
                "format": "zeta0.predictive_world.packed_feature_shard.v1",
                "rank": rank,
                "world_size": world_size,
                "start": start,
                "stop": stop,
                "file": array_path.name,
                "nbytes": array_path.stat().st_size,
                "sha256": shard_sha,
                "cache_identity": cache_identity,
                "index_binding": binding,
                "legacy_reused": reused,
                "qwen_encoded": encoded,
                "recent_frame_reused": recent_reused,
            }
            _immutable_json(sidecar_path, completed)
        print(json.dumps({
            "event": "predictive_world_feature_shard_complete",
            "rank": rank,
            "start": start,
            "stop": stop,
        }, sort_keys=True), flush=True)
        _barrier(args.distributed)
        if rank == 0:
            shards = []
            for shard_rank in range(world_size):
                shard_start = rows * shard_rank // world_size
                shard_stop = rows * (shard_rank + 1) // world_size
                shard_array, _, shard_sidecar = _rank_paths(output, shard_rank, world_size)
                value = _validate_rank_sidecar(
                    shard_sidecar,
                    rank=shard_rank,
                    world_size=world_size,
                    start=shard_start,
                    stop=shard_stop,
                    cache_identity=cache_identity,
                    index_binding=binding,
                    # Every owner rank has already checked (or just created and
                    # hashed) its shard before this barrier.  Re-hashing every
                    # multi-GB shard serially on rank zero would turn manifest
                    # publication into hours of redundant I/O.
                    verify_hash=False,
                )
                if value is None:
                    raise RuntimeError(f"feature shard did not publish its sidecar: {shard_rank}")
                shards.append({
                    "rank": shard_rank,
                    "start": shard_start,
                    "stop": shard_stop,
                    "file": str(shard_array.relative_to(output)),
                    "nbytes": int(value["nbytes"]),
                    "sha256": str(value["sha256"]),
                    "legacy_reused": int(value.get("legacy_reused", 0)),
                    "qwen_encoded": int(value.get("qwen_encoded", 0)),
                    "recent_frame_reused": int(value.get("recent_frame_reused", 0)),
                })
            manifest = {
                "format": PACKED_FEATURE_FORMAT,
                "provenance": provenance,
                "cache_identity": cache_identity,
                "index_binding": binding,
                "rows": rows,
                "groups": groups,
                "feature_shape": list(FEATURE_SHAPE),
                "dtype": "float16",
                "shards": shards,
            }
            manifest["manifest_identity"] = canonical_sha256({
                "cache_identity": cache_identity,
                "index_binding": binding,
                "rows": rows,
                "groups": groups,
                "shards": shards,
            })
            _immutable_json(manifest_path, manifest)
            verify_predictive_world_feature_manifest(manifest_path)
            print(json.dumps({
                "event": "predictive_world_feature_cache_published",
                "manifest": str(manifest_path),
                "rows": rows,
                "legacy_reused": sum(value["legacy_reused"] for value in shards),
                "qwen_encoded": sum(value["qwen_encoded"] for value in shards),
                "recent_frame_reused": sum(value["recent_frame_reused"] for value in shards),
            }, sort_keys=True), flush=True)
        _barrier(args.distributed)
    finally:
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
