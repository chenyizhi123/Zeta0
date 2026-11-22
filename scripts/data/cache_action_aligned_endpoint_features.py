#!/usr/bin/env python
"""Build the frame-major action-aligned endpoint Qwen3-VL cache.

For a large video dataset, launch one worker per visible GPU with
``torchrun ... --distributed``.  Each rank decodes only its contiguous row
shard and rank 0 publishes the same single-file ABI after an atomic merge.
"""

from __future__ import annotations
from psi.models.action_aligned_image import IMAGE_PREPROCESSING_VERSIONS

import argparse
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from psi.data.lerobot.action_aligned_cache import (
    build_action_aligned_endpoint_manifest,
    build_action_aligned_vlm_checkpoint_fingerprint,
    build_action_aligned_vlm_processor_fingerprint,
    extract_endpoint_vlm_features,
    sha256_file,
)
from psi.data.lerobot.transition_cache import build_lerobot_dataset_fingerprint
from psi.models.action_aligned_vlm_features import (
    VLM_TARGET_DIM,
    VLM_TARGET_TOKENS,
    endpoint_vlm_feature_shape,
    resolve_qwen3vl_visual_feature_spec,
)


def publish_feature_array(
    features: np.ndarray,
    output_dir: str | os.PathLike[str],
    *,
    episode_data_index,
    primary_image_key: str,
    vlm_model: str,
    vlm_feature_spec,
    dataset_fingerprint=None,
    vlm_checkpoint_fingerprint=None,
    vlm_processor_fingerprint=None,
    feature_name: str | None = None,
    image_preprocessing: str = "legacy_v1",
) -> Path:
    """Atomically publish an array and then its generation-pointer manifest."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features)
    if features.ndim != 3 or features.shape[1:] != (VLM_TARGET_TOKENS, VLM_TARGET_DIM):
        raise ValueError("endpoint features must have shape [N,16,2048]")
    if features.dtype != np.float16:
        features = features.astype(np.float16)
    feature_name = feature_name or f"action_aligned_vlm_endpoint_features.{uuid.uuid4().hex}.npy"
    feature_rel = Path(feature_name)
    if (
        feature_rel.is_absolute()
        or ".." in feature_rel.parts
        or not feature_rel.name.startswith("action_aligned_vlm_endpoint_features.")
        or feature_rel.suffix != ".npy"
    ):
        raise ValueError("feature_name must be a local action_aligned_vlm_endpoint_features.<uuid>.npy name")
    feature_name = feature_rel.name
    feature_path = output_dir / feature_name
    if feature_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing endpoint feature array: {feature_path}"
        )
    # NPY is written to a sibling temporary file and renamed before the
    # manifest is published.  A partially written feature array is never the
    # generation pointer.
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".endpoint.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, features, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, feature_path)
        manifest = build_action_aligned_endpoint_manifest(
            num_frames=len(features),
            episode_data_index=episode_data_index,
            primary_image_key=primary_image_key,
            vlm_model=vlm_model,
            vlm_feature_spec=vlm_feature_spec,
            dataset_fingerprint=dataset_fingerprint,
            vlm_checkpoint_fingerprint=vlm_checkpoint_fingerprint,
            vlm_processor_fingerprint=vlm_processor_fingerprint,
            feature_file=feature_path.name,
            feature_nbytes=feature_path.stat().st_size,
            feature_sha256=sha256_file(feature_path),
            feature_mtime_ns=feature_path.stat().st_mtime_ns,
            image_preprocessing=image_preprocessing,
        )
        manifest_path = output_dir / "action_aligned_vlm_endpoint_manifest.json"
        with tempfile.NamedTemporaryFile(
            dir=output_dir, prefix=".endpoint-manifest.", suffix=".tmp", mode="w", encoding="utf-8", delete=False
        ) as handle:
            temporary_manifest = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, manifest_path)
    finally:
        for path in (locals().get("temporary"), locals().get("temporary_manifest")):
            if path is not None:
                Path(path).unlink(missing_ok=True)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-preprocessing", choices=IMAGE_PREPROCESSING_VERSIONS, default="legacy_v1")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="LeRobot repository root containing meta/info.json (not its parent task directory).",
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-image-key", required=True)
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Pinned local Qwen3-VL Ψ0 checkpoint used for the future feature targets.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=32,
        help=(
            "Contiguous video frames decoded by one worker at once. Multiple "
            "chunks are collated into one GPU --batch-size so decode and VLM "
            "encoding run in parallel (default: 32)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Video decode workers used for contiguous per-episode batches.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help=(
            "Run one cache worker per torchrun rank. Each rank encodes a "
            "contiguous frame shard and rank 0 atomically merges them."
        ),
    )
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        help="torch.distributed backend used with --distributed (default: nccl).",
    )
    return parser


def _init_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    """Initialize torchrun state and return ``(rank, world_size, local_rank)``."""

    if not args.distributed:
        return 0, 1, 0
    if not dist.is_available():
        raise RuntimeError("--distributed requires torch.distributed")
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if rank < 0 or world_size <= 1 or local_rank < 0:
        raise RuntimeError(
            "--distributed must be launched by torchrun with RANK, WORLD_SIZE, "
            "and LOCAL_RANK set"
        )
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--distributed with a CUDA device requires CUDA")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA devices "
                f"({torch.cuda.device_count()})"
            )
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    return rank, world_size, local_rank


def _shard_bounds(num_frames: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return contiguous, non-overlapping frame bounds for one rank."""

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid shard rank/world_size: {rank}/{world_size}")
    return num_frames * rank // world_size, num_frames * (rank + 1) // world_size


def _column_to_numpy(column, *, dtype) -> np.ndarray:
    """Convert a LeRobot scalar column to a dense one-dimensional array."""

    if isinstance(column, torch.Tensor):
        return column.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
    try:
        array = np.asarray(column, dtype=dtype)
        if array.ndim == 1:
            return array
    except (TypeError, ValueError):
        pass
    values = []
    for value in column:
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif hasattr(value, "item"):
            try:
                value = value.item()
            except ValueError:
                pass
        values.append(value)
    return np.asarray(values, dtype=dtype).reshape(-1)


class _EndpointVideoBatchDataset(torch.utils.data.Dataset):
    """Decode contiguous rows from each episode in one video query.

    The endpoint ABI is still one raw feature per dense dataset row.  Batching
    only avoids constructing a decoder and seeking once for every row; it does
    not change row order, frame selection or preprocessing.
    """

    def __init__(
        self,
        dataset,
        image_key: str,
        decode_chunk_size: int,
        *,
        shard_start: int = 0,
        shard_end: int | None = None,
    ) -> None:
        if decode_chunk_size <= 0:
            raise ValueError("decode_chunk_size must be positive")
        self.dataset = dataset
        self.image_key = image_key
        timestamps = _column_to_numpy(
            dataset.hf_dataset["timestamp"], dtype=np.dtype(np.float64)
        )
        episode_indices = _column_to_numpy(
            dataset.hf_dataset["episode_index"], dtype=np.dtype(np.int64)
        )
        if len(timestamps) != len(dataset) or len(episode_indices) != len(dataset):
            raise ValueError("LeRobot timestamp/episode columns do not match dataset length")
        self.timestamps = timestamps
        self.episode_indices = episode_indices
        starts = _column_to_numpy(
            dataset.episode_data_index["from"], dtype=np.dtype(np.int64)
        )
        ends = _column_to_numpy(
            dataset.episode_data_index["to"], dtype=np.dtype(np.int64)
        )
        if starts.shape != ends.shape:
            raise ValueError("LeRobot episode_data_index has mismatched boundaries")
        if shard_end is None:
            shard_end = len(dataset)
        shard_start, shard_end = int(shard_start), int(shard_end)
        if shard_start < 0 or shard_end <= shard_start or shard_end > len(dataset):
            raise ValueError(
                f"invalid endpoint cache shard [{shard_start}, {shard_end}) "
                f"for {len(dataset)} rows"
            )
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.chunks: list[tuple[int, int, int]] = []
        for episode_start, episode_end in zip(starts, ends):
            episode_start = max(int(episode_start), shard_start)
            episode_end = min(int(episode_end), shard_end)
            if episode_start >= episode_end:
                continue
            episode_index = int(self.episode_indices[episode_start])
            if not np.all(
                self.episode_indices[episode_start:episode_end] == episode_index
            ):
                raise ValueError("episode_index changes inside a declared episode")
            for start in range(episode_start, episode_end, decode_chunk_size):
                self.chunks.append(
                    (
                        start,
                        min(start + decode_chunk_size, episode_end),
                        episode_index,
                    )
                )
        if not self.chunks:
            raise ValueError(
                f"no endpoint video chunks generated for shard "
                f"[{shard_start}, {shard_end})"
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int):
        start, end, episode_index = self.chunks[index]
        query_videos = getattr(self.dataset, "_query_videos", None)
        if not callable(query_videos):
            raise RuntimeError("installed LeRobot dataset has no _query_videos method")
        frames = query_videos(
            {self.image_key: self.timestamps[start:end].tolist()}, episode_index
        )[self.image_key]
        if frames.ndim == 3 and end - start == 1:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[0] != end - start:
            raise RuntimeError(
                f"video decoder returned {tuple(frames.shape)} for rows [{start},{end})"
            )
        return torch.arange(start, end, dtype=torch.int64), frames


def _collate_endpoint_chunks(chunks):
    """Join independently decoded contiguous chunks into one GPU batch."""

    if not chunks:
        raise ValueError("endpoint cache collator received no chunks")
    rows = torch.cat([torch.as_tensor(item[0], dtype=torch.int64) for item in chunks])
    frames = torch.cat([torch.as_tensor(item[1]) for item in chunks], dim=0)
    if rows.ndim != 1 or frames.ndim != 4 or frames.shape[0] != rows.shape[0]:
        raise RuntimeError("endpoint cache chunks have inconsistent row/frame shapes")
    if rows.numel() > 1 and not torch.equal(rows[1:], rows[:-1] + 1):
        raise RuntimeError("endpoint cache chunks must remain globally contiguous")
    return rows, frames


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    array.flush()
    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def _write_manifest_atomically(output_dir: Path, manifest: dict) -> Path:
    manifest_path = output_dir / "action_aligned_vlm_endpoint_manifest.json"
    temporary = output_dir / f".endpoint-manifest.{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def _merge_distributed_shards(
    *,
    output_dir: Path,
    build_dir: Path,
    feature_name: str,
    feature_shape: tuple[int, int, int],
    world_size: int,
) -> Path:
    """Merge rank-local endpoint arrays without materializing a second copy."""

    feature_path = output_dir / feature_name
    temporary = output_dir / f".{feature_name}.{os.getpid()}.tmp.npy"
    merged: np.memmap | None = None
    try:
        merged = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=feature_shape
        )
        for rank in range(world_size):
            start, end = _shard_bounds(feature_shape[0], rank, world_size)
            shard_path = build_dir / f"rank-{rank}.npy"
            if not shard_path.is_file():
                raise RuntimeError(f"missing endpoint cache shard: {shard_path}")
            shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            expected_shape = (end - start, *feature_shape[1:])
            if tuple(shard.shape) != expected_shape or shard.dtype != np.float16:
                raise RuntimeError(
                    f"endpoint cache shard {shard_path} has shape/dtype "
                    f"{tuple(shard.shape)}/{shard.dtype}, expected "
                    f"{expected_shape}/float16"
                )
            for offset in range(0, end - start, 4096):
                block_end = min(offset + 4096, end - start)
                merged[start + offset:start + block_end] = shard[offset:block_end]
            mmap_handle = getattr(shard, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        _close_memmap(merged)
        merged = None
        os.replace(temporary, feature_path)
    except BaseException:
        if merged is not None:
            try:
                _close_memmap(merged)
            except (AttributeError, OSError, ValueError):
                pass
        temporary.unlink(missing_ok=True)
        raise
    return feature_path


def _validate_published_cache(
    output_dir: Path, *, episode_data_index, num_frames: int
) -> None:
    """Exercise the exact mmap reader used by training before returning."""

    from psi.data.lerobot.action_aligned_cache import ActionAlignedEndpointCache

    cache = ActionAlignedEndpointCache(
        output_dir,
        episode_data_index=episode_data_index,
        # The manifest SHA-256 was just computed immediately before this
        # reader smoke test.  Exercise the mmap ABI without scanning the full
        # 4.6-GiB array a second time.
        verify_feature_sha256=False,
    )
    try:
        for row in sorted({0, num_frames // 2, num_frames - 1}):
            value, valid = cache.get_endpoint(row, 30)
            if value.shape != (VLM_TARGET_TOKENS, VLM_TARGET_DIM) or value.dtype != torch.float16:
                raise RuntimeError(
                    f"endpoint reader returned {tuple(value.shape)}/{value.dtype} "
                    "instead of [16,2048]/float16"
                )
            if not torch.isfinite(value).all():
                raise RuntimeError(f"endpoint reader returned non-finite row {row}")
            if row < num_frames and not isinstance(valid, bool):
                raise RuntimeError("endpoint reader validity flag is not bool")
    finally:
        cache.close()


def main() -> None:
    args = build_parser().parse_args()
    rank, world_size, local_rank = _init_distributed(args)
    distributed = world_size > 1
    try:
        _run(
            args,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _run(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
) -> None:
    distributed = world_size > 1
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from psi.data.lerobot.compat import LeRobotDataset
    except Exception as error:
        raise RuntimeError("cache generation requires transformers and LeRobot") from error
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.decode_chunk_size <= 0:
        raise ValueError("--decode-chunk-size must be positive")
    if args.batch_size % args.decode_chunk_size:
        raise ValueError("--batch-size must be divisible by --decode-chunk-size")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is unavailable")

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps={args.primary_image_key: [0.0]},
        image_transforms=None,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_start, shard_end = _shard_bounds(len(dataset), rank, world_size)
    if distributed and args.device.startswith("cuda"):
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, local_files_only=True
    )
    vlm_feature_spec = resolve_qwen3vl_visual_feature_spec(model)

    # Hashing the complete VLM checkpoint and all dataset source files is
    # intentionally fail-closed, but doing it independently on every GPU
    # would turn a two-GPU cache build into duplicate multi-gigabyte I/O.
    # Rank 0 creates the immutable provenance once and broadcasts it before
    # any feature shards are published.
    provenance_box: list[dict | None] = [None]
    if rank == 0:
        provenance_box[0] = {
            "dataset_fingerprint": build_lerobot_dataset_fingerprint(dataset),
            "vlm_checkpoint_fingerprint": build_action_aligned_vlm_checkpoint_fingerprint(
                args.model_name_or_path, model.config, include_content_hash=True
            ),
            "vlm_processor_fingerprint": build_action_aligned_vlm_processor_fingerprint(
                processor
            ),
        }
    if distributed:
        dist.broadcast_object_list(provenance_box, src=0)
    provenance = provenance_box[0]
    if provenance is None:
        raise RuntimeError("rank 0 did not publish VLM cache provenance")
    dataset_fingerprint = provenance["dataset_fingerprint"]
    vlm_checkpoint_fingerprint = provenance["vlm_checkpoint_fingerprint"]
    vlm_processor_fingerprint = provenance["vlm_processor_fingerprint"]

    # Rank 0 chooses the generation name and broadcasts it so no rank can
    # publish a mismatched manifest.
    feature_name_box = [
        f"action_aligned_vlm_endpoint_features.{uuid.uuid4().hex}.npy"
        if rank == 0
        else ""
    ]
    if distributed:
        dist.broadcast_object_list(feature_name_box, src=0)
    feature_name = str(feature_name_box[0])
    feature_shape = endpoint_vlm_feature_shape(len(dataset))

    cache_dataset = _EndpointVideoBatchDataset(
        dataset,
        args.primary_image_key,
        args.decode_chunk_size,
        shard_start=shard_start,
        shard_end=shard_end,
    )
    loader = DataLoader(
        cache_dataset,
        # Four 32-frame chunks are decoded by independent workers before a
        # 128-frame Qwen visual batch is launched.  The old one-chunk-per-GPU
        # batch design left RTX 4090s idle while PyAV sought a long clip.
        batch_size=args.batch_size // args.decode_chunk_size,
        collate_fn=_collate_endpoint_chunks,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    if not distributed or rank == 0:
        print(
            f"Endpoint cache decode mode: batched-video; "
            f"decode_chunks_on_rank={len(cache_dataset)}; "
            f"gpu_batches_on_rank={len(loader)}; "
            f"decode_chunk_size={args.decode_chunk_size}; "
            f"gpu_batch_size={args.batch_size}"
            + (f"; world_size={world_size}" if distributed else "")
        )

    local_array: np.memmap | None = None
    written: np.ndarray | None = None
    if distributed:
        build_dir = output_dir / f".{Path(feature_name).stem}.shards"
        if rank == 0:
            if build_dir.exists():
                raise RuntimeError(
                    f"refusing to reuse stale distributed endpoint build directory: {build_dir}"
                )
            build_dir.mkdir(parents=False)
        dist.barrier()
        temporary = build_dir / f".rank-{rank}.tmp.npy"
        local_path = build_dir / f"rank-{rank}.npy"
        local_array = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=(shard_end - shard_start, *feature_shape[1:]),
        )
        written = np.zeros(shard_end - shard_start, dtype=bool)
    else:
        # A full VLM cache is roughly 4.6 GiB for the Sofa dataset.  Stream
        # every batch into an NPY memmap instead of retaining all rows in RAM.
        temporary = output_dir / f".{feature_name}.{os.getpid()}.tmp.npy"
        local_path = output_dir / feature_name
        if local_path.exists() or temporary.exists():
            raise FileExistsError(
                f"refusing to overwrite endpoint cache generation {local_path}"
            )
        local_array = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=(shard_end - shard_start, *feature_shape[1:]),
        )
        written = np.zeros(shard_end - shard_start, dtype=bool)

    next_row = shard_start
    try:
        for row_indices, images in tqdm(
            loader,
            total=len(loader),
            desc=(
                "Caching action-aligned endpoint features"
                if not distributed
                else f"Caching endpoint features rank={rank}/{world_size}"
            ),
            disable=distributed and rank != 0,
        ):
            row_indices = torch.as_tensor(row_indices, dtype=torch.int64)
            expected = torch.arange(
                next_row, next_row + len(row_indices), dtype=torch.int64
            )
            if not torch.equal(row_indices, expected):
                raise RuntimeError(
                    f"video loader emitted rows {row_indices.tolist()}, "
                    f"expected {expected.tolist()}"
                )
            encoded = extract_endpoint_vlm_features(
                model, processor, torch.as_tensor(images), device=device,
                spec=vlm_feature_spec,
                image_preprocessing=args.image_preprocessing,
            )
            assert local_array is not None and written is not None
            local_indices = row_indices.numpy() - shard_start
            if np.any(local_indices < 0) or np.any(
                local_indices >= shard_end - shard_start
            ):
                raise RuntimeError("endpoint loader emitted a row outside its rank shard")
            if np.unique(local_indices).size != local_indices.size or written[local_indices].any():
                raise RuntimeError("endpoint loader emitted a row more than once")
            local_array[local_indices] = encoded.numpy()
            written[local_indices] = True
            next_row += len(row_indices)
    except BaseException:
        if local_array is not None:
            try:
                _close_memmap(local_array)
            except (AttributeError, OSError, ValueError):
                pass
        temporary.unlink(missing_ok=True)
        raise

    if next_row != shard_end:
        raise RuntimeError(
            f"endpoint cache loader emitted rows [{shard_start}, {next_row}), "
            f"expected [{shard_start}, {shard_end})"
        )
    if distributed:
        assert local_array is not None and written is not None
        if not written.all():
            missing = np.flatnonzero(~written)
            raise RuntimeError(f"endpoint cache missed {missing.size} local rows")
        _close_memmap(local_array)
        local_array = None
        temporary.replace(local_path)
        dist.barrier()
        if rank == 0:
            feature_path = _merge_distributed_shards(
                output_dir=output_dir,
                build_dir=build_dir,
                feature_name=feature_name,
                feature_shape=feature_shape,
                world_size=world_size,
            )
            manifest = build_action_aligned_endpoint_manifest(
                num_frames=len(dataset),
                episode_data_index=dataset.episode_data_index,
                primary_image_key=args.primary_image_key,
                vlm_model=args.model_name_or_path,
                vlm_feature_spec=vlm_feature_spec,
                dataset_fingerprint=dataset_fingerprint,
                vlm_checkpoint_fingerprint=vlm_checkpoint_fingerprint,
                vlm_processor_fingerprint=vlm_processor_fingerprint,
                feature_file=feature_name,
                feature_nbytes=feature_path.stat().st_size,
                feature_sha256=sha256_file(feature_path),
                feature_mtime_ns=feature_path.stat().st_mtime_ns,
                image_preprocessing=args.image_preprocessing,
            )
            _write_manifest_atomically(output_dir, manifest)
            _validate_published_cache(
                output_dir,
                episode_data_index=dataset.episode_data_index,
                num_frames=len(dataset),
            )
            shutil.rmtree(build_dir)
            print(f"Wrote {len(dataset)} endpoint rows using {world_size} GPUs: {feature_path}")
        dist.barrier()
        return

    assert local_array is not None and written is not None
    if not written.all():
        missing = np.flatnonzero(~written)
        raise RuntimeError(f"endpoint cache missed {missing.size} local rows")
    _close_memmap(local_array)
    local_array = None
    temporary.replace(local_path)
    manifest = build_action_aligned_endpoint_manifest(
        num_frames=len(dataset),
        episode_data_index=dataset.episode_data_index,
        primary_image_key=args.primary_image_key,
        vlm_model=args.model_name_or_path,
        vlm_feature_spec=vlm_feature_spec,
        dataset_fingerprint=dataset_fingerprint,
        vlm_checkpoint_fingerprint=vlm_checkpoint_fingerprint,
        vlm_processor_fingerprint=vlm_processor_fingerprint,
        feature_file=feature_name,
        feature_nbytes=local_path.stat().st_size,
        feature_sha256=sha256_file(local_path),
        feature_mtime_ns=local_path.stat().st_mtime_ns,
        image_preprocessing=args.image_preprocessing,
    )
    manifest_path = _write_manifest_atomically(output_dir, manifest)
    _validate_published_cache(
        output_dir,
        episode_data_index=dataset.episode_data_index,
        num_frames=len(dataset),
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
