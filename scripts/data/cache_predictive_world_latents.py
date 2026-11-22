#!/usr/bin/env python3
"""Materialize immutable raw E latents bound to a frozen reference bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from psi.config.predictive_world import load_predictive_world_config
from psi.data.predictive_world.assets import resolve_asset_path
from psi.data.predictive_world.cache import LatentCache, canonical_sha256, file_sha256
from psi.data.predictive_world.dataset import (
    PredictiveWorldRowReader,
    assert_predictive_world_training_data_contract,
)
from psi.data.predictive_world.index import temporal_abi_from_data_config
from psi.data.predictive_world.runtime import PredictiveWorldTrainingCollator
from psi.models.predictive_world.engine import move_to_device
from psi.models.predictive_world.factory import build_phase


class _Rows(Dataset):
    def __init__(self, rows, reader):
        self.rows, self.reader = rows, reader

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.reader.read(self.rows[index], occurrence_index=index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True,
        help="Resolved continuation config pointing at world-model-final",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config, _ = load_predictive_world_config(
        args.config, allow_existing_output_override=True
    )
    bundle = config.training.world_model
    if config.training.phase not in {"joint", "policy"} or not bundle:
        raise ValueError("latent caching requires a continuation config pointing at a world model")
    reference_sidecar = json.loads(
        (Path(bundle) / "predictive_world_checkpoint.json").read_text()
    )
    reference_id = reference_sidecar["metadata"]["reference_id"]
    artifacts = assert_predictive_world_training_data_contract(
        config.data, measurement_dim=config.model.measurement_dim
    )
    feature_manifest = artifacts.feature_manifest
    input_fingerprint = canonical_sha256({
        "index": file_sha256(Path(config.data.index_path)),
        "features": file_sha256(feature_manifest),
        "measurements": file_sha256(Path(config.data.measurement_registry)),
    })
    cache = LatentCache(
        args.output, reference_id=reference_id, latent_shape=config.model.latent_shape,
        input_fingerprint=input_fingerprint,
    )
    if args.verify_only:
        ids, values, valid = cache.load()
        print(json.dumps({
            "event": "predictive_world_latent_cache_verified",
            "samples": len(ids), "shape": list(values.shape),
            "valid_groups": int(valid.sum()), "reference_id": reference_id,
        }, sort_keys=True))
        return
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size/workers are invalid")
    from transformers import AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        str(resolve_asset_path(config.model.qwen_model, config.runtime.asset_path_map)), local_files_only=True)
    reader = PredictiveWorldRowReader(
        processor,
        artifacts.feature_manifest,
        index_path=config.data.index_path,
        observed_visual_mode=config.data.observed_visual_mode,
        current_visual_pooling=config.data.current_visual_pooling,
        capabilities=artifacts.capabilities,
        measurement_schemas=artifacts.measurement_schemas,
    )
    decoded_rows = [
        json.loads(line)
        for line in Path(config.data.index_path).read_text().splitlines()
        if line
    ]
    rows = [row for row in decoded_rows if row.get("split") == args.split]
    if not rows:
        raise ValueError(f"predictive-world index has no {args.split!r} rows")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("latent cache input sample IDs are not unique")
    collator = PredictiveWorldTrainingCollator(
        future_measurement_offsets_s=temporal_abi_from_data_config(config.data)[
            "future_measurement_offsets_s"
        ],
        future_group_ends_s=config.data.future_group_ends_s,
        state_count=config.data.state_history_count,
        action_horizon=config.data.action_horizon,
        body_points_per_group=config.model.body_points_per_group,
    )
    loader = DataLoader(
        _Rows(rows, reader), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collator,
    )
    built = build_phase(config, device=device)
    encoder = built.reference_modules["encoder"].eval().to(device)
    ids, latents, masks = [], [], []
    names = (
        "current_measurements", "current_measurement_valid", "future_patches",
        "future_patch_valid", "future_scene_times", "future_measurements",
        "future_measurement_valid", "future_measurement_times", "group_valid",
        "future_patch_positions", "future_patch_camera_indices",
    )
    with torch.inference_mode():
        for batch in loader:
            batch = move_to_device(batch, device)
            features = built.model.observation_encoder(batch["observed"])
            arguments = {name: batch[name] for name in names}
            arguments["anchor_patches"] = features.current_visual
            arguments["anchor_patch_valid"] = torch.ones(
                features.current_visual.shape[:2], dtype=torch.bool, device=device
            )
            latents.append(encoder(**arguments).cpu())
            masks.append(batch["group_valid"].cpu())
            ids.extend(batch["typed_batch"].metadata.sample_ids)
    online_latents, online_masks = torch.cat(latents), torch.cat(masks)
    cache.publish(ids, online_latents, online_masks)
    restored_ids, restored_latents, restored_masks = cache.load()
    if (
        restored_ids != tuple(ids)
        or not torch.equal(restored_latents, online_latents)
        or not torch.equal(restored_masks, online_masks)
    ):
        raise RuntimeError("published latent cache is not bit-exact with online frozen E")
    print(json.dumps({
        "event": "predictive_world_latent_cache_published",
        "samples": len(ids), "output": str(Path(args.output).resolve()),
        "reference_id": reference_id,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
