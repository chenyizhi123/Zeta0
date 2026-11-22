#!/usr/bin/env python3
"""Offline causal body-readout diagnostic for a packaged fine-tuned model.

No simulator, training, action sampling, or future encoder is run. Candidate
windows are chosen from measured motion BEFORE inference. Outputs retain all
candidates, including errors, and explicitly declare their demonstration split.
Run with the checkpoint's processor environment; rendering is a separate step.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True, help="Local task sonic_refpose_v3_1 directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--windows-per-episode", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.package, args.qwen, args.dataset = [v.resolve() for v in (args.package, args.qwen, args.dataset)]
    args.output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import transformers
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from psi.config.predictive_world import (
        load_predictive_world_config,
        predictive_world_architecture_contract,
        predictive_world_loss_contract,
    )
    from psi.data.lerobot.action_aligned_cache import preprocess_endpoint_vlm_image
    from psi.data.predictive_world.cache import file_sha256, canonical_sha256
    from psi.data.predictive_world.contracts import ObservedHistory
    from psi.data.predictive_world.provenance import verify_qwen_provenance
    from psi.data.predictive_world.video import decode_torchcodec_frames
    from psi.models.predictive_world.checkpoint import read_predictive_world_bundle
    from psi.models.predictive_world.factory import _forecast_modules, _state_registry
    from psi.models.predictive_world.latent_transition import LatentNormalizer, prediction_to_raw
    from psi.models.predictive_world.observation import FrozenQwenObservationEncoder
    from psi.models.predictive_world.phases import _history_condition
    from psi.models.predictive_world.representation import WorldReadout

    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device(args.device)
    config, payload = load_predictive_world_config(args.package / "config.json", allow_existing_output_override=True)
    asset_map = json.loads((args.package / "asset_path_map.local.json").read_text())
    def asset(path):
        result = Path(asset_map.get(path, path))
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    state_path = asset(config.data.state_schema_registry)
    protocol_path = asset(config.data.action_protocol_registry)
    measure_path = asset(config.data.measurement_registry)
    index_path = asset(config.data.index_path)
    manifest_path = asset(str(Path(config.data.feature_cache) / "feature_manifest.json"))
    provenance = json.loads(manifest_path.read_text())["provenance"]
    print("Verifying checkpoint and frozen observation identities", flush=True)
    sidecar = read_predictive_world_bundle(args.package / "checkpoint")
    meta = sidecar["metadata"]
    for key, expected in {
        "resolved_config_sha256": canonical_sha256(payload),
        "architecture": predictive_world_architecture_contract(config.model),
        "loss_contract": predictive_world_loss_contract(config.training),
        "protocol_registry_sha256": file_sha256(protocol_path),
        "measurement_registry_sha256": file_sha256(measure_path),
        "encoder_id": provenance["encoder_id"],
        "processor_id": provenance["processor_id"],
    }.items():
        if meta[key] != expected:
            raise ValueError(f"Checkpoint binding mismatch: {key}")
    processor = AutoProcessor.from_pretrained(args.qwen, local_files_only=True)
    qwen_identity = verify_qwen_provenance(args.qwen, provenance, processor=processor)
    schema = json.loads(measure_path.read_text())["sonic_measured_whole_body_q29_v1"]["canonical_body"]
    if schema["layout_id"] != "g1_measured_arm14_leg12_waist3_v1":
        raise ValueError("This renderer requires the audited G1 whole-body target")
    indices = np.array(schema["raw_indices"])
    mean, std = np.array(schema["mean"]), np.array(schema["std"])

    # Selection depends on measured motion only, never on model error.
    tables = {}
    rows_by_episode = defaultdict(list)
    selected_episodes = []
    for line in index_path.open():
        row = json.loads(line)
        ep = row["episode_id"]
        if ep not in selected_episodes:
            if len(selected_episodes) >= args.episodes:
                continue
            selected_episodes.append(ep)
        if all(row["future_measurement_valid"]) and all(row["future_visual_valid"]):
            rows_by_episode[ep].append(row)
    selected = []
    for ep in selected_episodes:
        candidates = []
        for row in rows_by_episode[ep]:
            old = Path(row["materialization"]["data_path"])
            local = args.dataset / "data" / old.parent.name / old.name
            if local not in tables:
                table = pq.read_table(local, columns=["observation.state"])
                tables[local] = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            states = tables[local]
            current = states[row["state_indices"][-1], indices]
            truth = states[np.array(row["future_measurement_indices"])[:, None], indices]
            motion = float(np.sqrt(np.mean((truth - current) ** 2)))
            candidates.append((motion, row, local))
        kept = []
        for motion, row, local in sorted(candidates, key=lambda c: (-c[0], c[1]["anchor_time"])):
            if all(abs(row["anchor_time"] - t) >= 0.65 for t in kept):
                selected.append((motion, row, local))
                kept.append(row["anchor_time"])
            if len(kept) == args.windows_per_episode:
                break
    if not selected:
        raise ValueError("No complete measured future windows")
    (args.output / "selection.json").write_text(json.dumps([
        {"sample_id": r["sample_id"], "episode": r["episode_id"], "anchor": r["anchor_time"], "motion_rmse_rad": m}
        for m, r, _ in selected
    ], indent=2))
    print(f"Selected {len(selected)} windows in {len(selected_episodes)} episodes, before inference", flush=True)

    protocols = json.loads(protocol_path.read_text())
    dims, stats = _state_registry(str(state_path))
    modules = _forecast_modules(config, protocols, dims, stats, qwen_dim=2048)
    model = config.model
    modules["normalizer"] = LatentNormalizer(model.latent_shape)
    modules["readout"] = WorldReadout(
        measurement_dim=model.measurement_dim, latent_shape=model.latent_shape,
        max_scene_patches=model.max_scene_patches, body_points_per_group=model.body_points_per_group,
        hidden_dim=model.representation_hidden_dim, num_heads=model.representation_heads,
        ff_dim=model.representation_ff_dim,
    )
    with safe_open(args.package / "checkpoint/predictive_world.safetensors", framework="pt", device="cpu") as weights:
        for name, module in modules.items():
            state = {k[len(name) + 1:]: weights.get_tensor(k) for k in weights.keys() if k.startswith(name + ".")}
            module.load_state_dict(state, strict=True)
            module.to(device).eval()
            module.requires_grad_(False)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    observation_encoder = FrozenQwenObservationEncoder(qwen).eval()

    @torch.inference_mode()
    def forecast(observed: ObservedHistory):
        # This function has no label, demonstration-action or future-encoder input.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features = observation_encoder(observed)
            condition = _history_condition(observed=observed, features=features,
                state_adapters=modules["history_state_adapters"], history_encoder=modules["history_encoder"])
            prediction = modules["latent_transition"](condition,
                context_valid=torch.ones(condition.shape[:2], dtype=torch.bool, device=device),
                mode="prior", action_condition=None)
            raw = prediction_to_raw(prediction, modules["normalizer"],
                                    space=config.training.latent_prediction_space)
            _, body = modules["readout"](
                anchor_patches=features.current_visual,
                anchor_patch_valid=torch.ones(features.current_visual.shape[:2], dtype=torch.bool, device=device),
                z_raw=raw, group_valid=torch.ones(raw.shape[:2], dtype=torch.bool, device=device),
                scene_patch_count=16,
            )
        return body.float().cpu().numpy()[0].reshape(30, 29)

    predictions, truths, currents, records, frames = [], [], [], [], []
    for ordinal, (motion, row, local) in enumerate(selected):
        mat = row["materialization"]
        old = Path(mat["video_paths"][mat["primary_camera_id"]])
        video = args.dataset / "videos" / mat["primary_camera_id"] / old.parent.name / old.name
        frame = decode_torchcodec_frames(str(video), [row["history_visual_indices"][-1]])[0]
        image = preprocess_endpoint_vlm_image(frame, image_preprocessing=provenance["image_preprocessing"])
        text = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "text", "text": f"Camera: {mat['primary_camera_id']}; observation time: +0.000 seconds."},
            {"type": "image", "image": image}, {"type": "text", "text": mat["instruction"]}
        ]}], tokenize=False, add_generation_prompt=True)
        encoded = dict(processor(text=[text], images=[image], padding=True, return_tensors="pt"))
        encoded["image_counts"] = torch.tensor([1])
        encoded["primary_current_image_indices"] = torch.tensor([0])
        if config.data.current_visual_pooling == "content_area_v1":
            from psi.models.action_aligned_image import content_boxes
            encoded["image_content_boxes"] = content_boxes([image])
        state = tables[local][row["state_indices"][-1]]
        observed = ObservedHistory(
            qwen_inputs={k: v.to(device) for k, v in encoded.items()},
            raw_states=torch.tensor(state[None, None], device=device),
            state_valid=torch.ones((1, 1, len(state)), dtype=torch.bool, device=device),
            state_times=torch.tensor([row["state_actual_times"]], dtype=torch.float32, device=device),
            state_schema_ids=(mat["state_schema_id"],), action_protocol_ids=(mat["action_protocol_id"],),
            action_native_dims=torch.tensor([40], device=device),
            action_normalization_spaces=(mat["action_normalization_space"],), committed_actions=None,
        )
        pred_std = forecast(observed)
        pred = pred_std * std + mean
        # Labels are read only AFTER the causal forecast has returned.
        truth = tables[local][np.array(row["future_measurement_indices"])[:, None], indices]
        current = state[indices]
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite prediction: {row['sample_id']}")
        predictions.append(pred); truths.append(truth); currents.append(current)
        frame_uint8 = (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        frames.append(frame_uint8)
        Image.fromarray(frame_uint8).save(args.output / f"context-{ordinal:02d}.jpg", quality=95)
        rmse = float(np.sqrt(np.mean((pred-truth)**2)))
        records.append({"sample_id": row["sample_id"], "episode": row["episode_id"],
            "anchor_s": row["anchor_time"], "split": row["split"], "motion_rmse_rad": motion,
            "rmse_rad": rmse, "data_path": str(local), "video_path": str(video),
            "future_source_indices": row["future_measurement_indices"],
            "actual_times_s": row["future_measurement_actual_times"],
            "context_frame_index": row["history_visual_indices"][-1]})
        print(f"{ordinal+1}/{len(selected)} {row['episode_id']} t={row['anchor_time']:.2f}  RMSE {rmse:.4f} rad  hold {motion:.4f}", flush=True)
    predictions, truths, currents = map(np.stack, (predictions, truths, currents))
    np.savez_compressed(args.output / "predictions.npz", predicted=predictions, truth=truths,
        current=currents, valid=np.ones_like(truths, dtype=bool), times=np.arange(1,31)/50,
        mean=mean, std=std, raw_indices=indices)
    err = predictions-truths
    hold_err = currents[:, None]-truths
    summary = {
        "format": "zeta0.causal_body_readout_visualization.v1",
        "hypothesis": "Causal predicted latents decode subsequent measured body motion better than holding the current pose.",
        "comparison": "Causal prior with empty committed prefix vs current-pose persistence; all 29 physical joint angles.",
        "budget": {"episodes": args.episodes, "windows_per_episode": args.windows_per_episode, "optimizer_updates": 0},
        "stopping_condition": "One forward pass per prespecified window; no outcome-based reselection.",
        "package": str(args.package), "checkpoint_sha256": sidecar["weights_sha256"],
        "completed_updates": sidecar["completed_updates"], "qwen": str(args.qwen), "qwen_identity": qwen_identity,
        "config_sha256": canonical_sha256(payload), "index_sha256": file_sha256(index_path),
        "measurement_registry_sha256": file_sha256(measure_path),
        "source_script_sha256": file_sha256(Path(__file__)),
        "source_module_sha256": {name: file_sha256(Path(__file__).resolve().parents[1] / "src/psi" / name)
            for name in ["models/predictive_world/observation.py", "models/predictive_world/history.py",
                "models/predictive_world/latent_transition.py", "models/predictive_world/representation.py",
                "models/predictive_world/phases.py", "models/predictive_world/factory.py"]},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "executable": sys.executable, "device": str(device), "gpu": torch.cuda.get_device_name(device)},
        "prediction_mode": "causal_prior_empty_committed_prefix", "future_encoder_used": False,
        "latent_prediction_space": config.training.latent_prediction_space,
        "root_convention": "Both robots share a fixed canonical pelvis transform. Global root motion is not predicted or scored.",
        "selection": "First episodes; largest measured-motion RMSE windows separated by >=0.65s; chosen before inference.",
        "display_indices": [i*args.windows_per_episode for i in range(min(3,len(selected_episodes)))],
        "split_note": "Demonstration windows from the fine-tuning index; this diagnostic is not held-out generalization or closed-loop evidence.",
        "joint_slots": schema["slots"], "records": records,
        "metrics": {"rmse_rad": float(np.sqrt(np.mean(err**2))),
            "persistence_rmse_rad": float(np.sqrt(np.mean(hold_err**2))),
            "rmse_deg": float(np.rad2deg(np.sqrt(np.mean(err**2)))),
            "persistence_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(hold_err**2)))),
            "rmse_deg_by_frame": np.rad2deg(np.sqrt(np.mean(err**2, axis=(0,2)))).tolist(),
            "persistence_rmse_deg_by_frame": np.rad2deg(np.sqrt(np.mean(hold_err**2, axis=(0,2)))).tolist()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
