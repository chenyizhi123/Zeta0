#!/usr/bin/env python3
"""Cache frozen-Qwen future fields for unique Stage-A anchors only."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from psi.data.lerobot.action_aligned_cache import (
    build_action_aligned_vlm_processor_fingerprint,
    extract_endpoint_vlm_features,
)
from psi.data.predictor_pretrain.cache_index import (
    load_anchor_rows_by_sample_ids,
    load_source_anchor_shard,
)
from psi.data.predictor_pretrain.registry import canonical_sha256, load_manifest_rows
from psi.data.predictor_pretrain.runtime import (
    LeRobotAnchorReader, LeRobotRuntimeSource, fingerprint_model_directory,
)
from psi.data.predictor_pretrain.sparse_cache import SparseEndpointCache, SparseEndpointCacheWriter
from psi.models.action_aligned_vlm_features import resolve_qwen3vl_visual_feature_spec
from psi.models.action_aligned_image import IMAGE_PREPROCESSING_VERSIONS, image_preprocessing_spec


class _Targets(Dataset):
    def __init__(self, rows, offsets, source: LeRobotRuntimeSource) -> None:
        self.rows = rows
        self.offsets = offsets
        self.reader = LeRobotAnchorReader({source.source_id: source})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        target_row = dict(row)
        target_row["current_video_timestamp"] = row["target_video_timestamp"]
        target_row["current_row_index"] = row["target_row_index"]
        return self.offsets[index], self.reader._decode(
            target_row, self.reader.sources[str(row["source_id"])]
        )


def _collate(rows):
    return torch.tensor([row[0] for row in rows]), torch.stack([row[1] for row in rows])


def reuse_endpoint_fields(
    writer,
    rows,
    previous_rows,
    previous_cache,
    *,
    previous_manifest_sha256=None,
):
    """Copy identical frozen target fields into a new cache, never mutate V1.

    Input camera/history changes do not affect the image-only future target.
    Reuse requires an unchanged source revision, physical frame, crop and
    frozen teacher/processor, not merely a matching old sample id.
    """
    prior = previous_cache.manifest.get("provenance", {})
    for key in ("source_id", "teacher_sha256", "teacher_processor_fingerprint", "image_crop", "image_preprocessing"):
        if prior.get(key) != writer.provenance.get(key):
            raise ValueError(f"cannot reuse endpoint fields with different {key}")
    previous_sha = (
        str(previous_manifest_sha256)
        if previous_manifest_sha256 is not None
        else canonical_sha256(previous_rows)
    )
    if prior.get("dataset_manifest_sha256") != previous_sha:
        raise ValueError("reuse manifest does not match the previous cache provenance")
    def identity(row):
        return (str(row["source_id"]), str(row["revision"]), str(row["video_path"]),
                str(row["primary_image_key"]), int(row["target_row_index"]),
                float(row["target_video_timestamp"]), float(row["target_timestamp"]))
    def trustworthy_frame(row):
        # V1 caches predating the dedicated-video row-index fix lack a decoder
        # ABI marker. Do not inherit an ambiguous timestamp-drifted target.
        return (bool(row.get("shared_data_file")) or row.get("nominal_fps") is None
                or int(round(float(row["target_video_timestamp"]) * float(row["nominal_fps"])))
                == int(row["target_row_index"]))
    old = {identity(row): str(row["sample_id"]) for row in previous_rows
           if str(row["source_id"]) == writer.provenance["source_id"] and trustworthy_frame(row)}
    count = 0
    offsets, fields = [], []
    for offset in writer.missing_offsets().tolist():
        previous_id = old.get(identity(rows[offset]))
        if previous_id is None or previous_id not in previous_cache:
            continue
        field, timestamp = previous_cache.get(previous_id)
        if abs(timestamp - float(rows[offset]["target_timestamp"])) > 1e-9:
            raise ValueError("reused cache target timestamp does not match the physical frame")
        offsets.append(offset)
        fields.append(field)
        if len(offsets) == 64:
            writer.write(offsets, np.stack(fields))
            count += len(offsets)
            offsets, fields = [], []
    if offsets:
        writer.write(offsets, np.stack(fields))
        count += len(offsets)
    if count:
        writer.checkpoint()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-preprocessing", choices=IMAGE_PREPROCESSING_VERSIONS, default="legacy_v1")
    parser.add_argument("--anchor-manifest", required=True)
    parser.add_argument("--anchor-index")
    parser.add_argument("--dataset-freeze", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--image-crop", choices=("none", "left_half", "right_half"), default="none")
    parser.add_argument("--video-backend", default="opencv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every-batches", type=int, default=100)
    parser.add_argument("--reuse-anchor-manifest")
    parser.add_argument("--reuse-anchor-index")
    parser.add_argument("--reuse-cache-manifest")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if bool(args.reuse_cache_manifest) != bool(
        args.reuse_anchor_index or args.reuse_anchor_manifest
    ):
        raise ValueError("reuse requires a previous cache and anchor index/manifest")
    if args.reuse_anchor_index and args.reuse_anchor_manifest:
        raise ValueError("reuse accepts an anchor index or manifest, not both")
    if args.reuse_cache_manifest and Path(args.output_dir).resolve() == Path(args.reuse_cache_manifest).resolve().parent:
        raise ValueError("new cache must not overwrite the previous cache directory")
    if args.checkpoint_every_batches <= 0:
        raise ValueError("--checkpoint-every-batches must be positive")
    if args.batch_size <= 0 or args.num_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError("batch size/prefetch factor must be positive and workers non-negative")
    freeze = json.loads(Path(args.dataset_freeze).read_text())
    manifest_sha = str(freeze.get("manifest_sha256", ""))
    if not manifest_sha:
        raise ValueError("dataset freeze lacks its canonical manifest hash")
    shard_provenance = None
    if args.anchor_index:
        selection = load_source_anchor_shard(
            args.anchor_index,
            expected_manifest_sha256=manifest_sha,
            source_id=args.source_id,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        rows = selection.rows
        if args.num_shards > 1:
            shard_provenance = {
                "format_version": "predictor_endpoint_cache_shard_v1",
                "index": selection.shard_index,
                "count": selection.num_shards,
                "start": selection.start,
                "stop": selection.stop,
                "source_num_samples": selection.source_num_samples,
                "selection_order": selection.selection_order,
                "anchor_index_format_version": selection.index_format_version,
            }
    else:
        if args.num_shards != 1 or args.shard_index != 0:
            raise ValueError("sharded cache construction requires --anchor-index")
        rows_all = load_manifest_rows(args.anchor_manifest)
        actual_sha = canonical_sha256(rows_all)
        if manifest_sha != actual_sha:
            raise ValueError("anchor manifest does not match dataset freeze")
        rows = [row for row in rows_all if row.get("source_id") == args.source_id]
    if not rows or any(row.get("split") not in {"PT_train", "PT_val", "PT_test"} for row in rows):
        raise ValueError("source has no rows or contains invalid PT splits")
    rows.sort(key=lambda row: (
        str(row["video_path"]), float(row["target_video_timestamp"]), str(row["sample_id"])
    ))
    source = LeRobotRuntimeSource(
        source_id=args.source_id, state_schema=None, image_crop=args.image_crop,
        video_backend=args.video_backend,
    )
    teacher_sha = fingerprint_model_directory(args.teacher_model)
    from transformers import AutoProcessor
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.teacher_model, local_files_only=True)
    provenance = {
        "teacher_sha256": teacher_sha,
        "dataset_manifest_sha256": manifest_sha,
        "teacher_processor_fingerprint": build_action_aligned_vlm_processor_fingerprint(processor),
        "source_id": args.source_id,
        "image_crop": args.image_crop,
        **({"image_preprocessing": image_preprocessing_spec(args.image_preprocessing)}
           if args.image_preprocessing != "legacy_v1" else {}),
    }
    if shard_provenance is not None:
        provenance["cache_shard"] = shard_provenance
    writer = SparseEndpointCacheWriter(
        args.output_dir, sample_ids=[row["sample_id"] for row in rows],
        target_timestamps=[row["target_timestamp"] for row in rows],
        provenance=provenance,
        resume=args.resume,
    )
    # Older runs selected dedicated-video frames by rounded timestamps. A
    # small number of source episodes have clock drift, so invalidate only
    # already-written rows whose timestamp-derived frame differs from their
    # authoritative parquet row identity. All unaffected work remains cached.
    drifted_offsets = [
        offset
        for offset, row in enumerate(rows)
        if (
            args.video_backend == "opencv"
            and not bool(row.get("shared_data_file"))
            and row.get("target_row_index") is not None
            and int(round(float(row["target_video_timestamp"]) * float(row["nominal_fps"])))
            != int(row["target_row_index"])
        )
    ]
    if drifted_offsets and writer.completed_manifest is None:
        invalidated = writer.invalidate(drifted_offsets)
        if invalidated:
            writer.checkpoint()
            print(
                f"Invalidated {invalidated} timestamp-drifted rows for exact frame recomputation"
            )
    if args.reuse_cache_manifest and writer.completed_manifest is None:
        previous_cache = SparseEndpointCache(args.reuse_cache_manifest)
        previous_manifest_sha = str(
            previous_cache.manifest.get("provenance", {}).get("dataset_manifest_sha256", "")
        )
        if args.reuse_anchor_index:
            previous_rows = load_anchor_rows_by_sample_ids(
                args.reuse_anchor_index,
                expected_manifest_sha256=previous_manifest_sha,
                sample_ids=(row.get("legacy_sample_id", "") for row in rows),
                source_id=args.source_id,
            )
        else:
            previous_rows = load_manifest_rows(args.reuse_anchor_manifest)
            previous_manifest_sha = None
        reused = reuse_endpoint_fields(
            writer,
            rows,
            previous_rows,
            previous_cache,
            previous_manifest_sha256=previous_manifest_sha,
        )
        print(f"Reused {reused} exact target fields from the read-only previous cache")
    pending_offsets = writer.missing_offsets().tolist()
    pending_rows = [rows[offset] for offset in pending_offsets]
    print(
        f"Sparse cache progress for {args.source_id}: "
        f"{writer.num_written}/{len(rows)} committed, "
        f"{len(pending_rows)} remaining"
    )
    if not pending_rows:
        print(writer.publish())
        return

    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.teacher_model, local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spec = resolve_qwen3vl_visual_feature_spec(model)
    loader = DataLoader(_Targets(pending_rows, pending_offsets, source), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        collate_fn=_collate, persistent_workers=args.num_workers > 0,
                        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)
    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        for batch_index, (offsets, images) in enumerate(
            tqdm(loader, desc=f"cache {args.source_id}"), start=1
        ):
            fields = extract_endpoint_vlm_features(model, processor, images, device=device, spec=spec,
                                                  image_preprocessing=args.image_preprocessing)
            writer.write(offsets.tolist(), fields.numpy())
            if batch_index % args.checkpoint_every_batches == 0 or stop_requested:
                writer.checkpoint()
            if stop_requested:
                print(
                    f"Checkpointed {writer.num_written}/{len(rows)} rows after signal; "
                    "exit 75 requests Slurm requeue"
                )
                writer.close()
                raise SystemExit(75)
    except BaseException:
        if not stop_requested:
            writer.checkpoint()
            writer.close()
        raise
    print(writer.publish())


if __name__ == "__main__":
    main()
