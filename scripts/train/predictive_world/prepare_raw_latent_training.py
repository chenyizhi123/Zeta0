#!/usr/bin/env python3
"""Prepare a new raw-latent joint run; never starts training or edits old runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile

from psi.config.predictive_world import load_predictive_world_config
from psi.models.predictive_world.initialization import read_initialization_source
from psi.models.predictive_world.posttrain_transition import make_posttrain_transition, make_raw_latent_transition
from psi.models.predictive_world.scratch import scratch_training_from_reference


def raw_latent_config(reference, *, initialization, output_dir, max_updates=20_000,
                      forecast_warmup_updates=0, forecast_ramp_updates=0):
    """Keep the input/optimizer recipe and explicitly replace the latent objective.

    The reference is a target joint config, including any independently required
    VLM/body/state transitions. Pretrained initialization uses its exact source;
    scratch initialization never opens a task checkpoint.
    """
    if initialization not in {"scratch", "pretrained"}:
        raise ValueError("initialization must be scratch or pretrained")
    if reference.training.phase != "joint":
        raise ValueError("raw latent preparation requires a joint reference")
    model = reference.model
    if initialization == "scratch":
        training = scratch_training_from_reference(replace(reference.training, joint_from_scratch=False))
        model = replace(model, latent_transition=replace(model.latent_transition, output_affine=False))
    else:
        training = reference.training
        if training.joint_from_scratch or not training.init_joint or not training.init_source_config:
            raise ValueError("pretrained preparation requires init_joint and init_source_config")
        source, payload, _ = read_initialization_source(
            training.init_joint, training.init_source_config,
            asset_path_map=reference.runtime.asset_path_map,
        )
        source_space = payload["training"].get("latent_prediction_space", "whitened")
        migration = make_raw_latent_transition(source, payload) if source_space == "whitened" else None
        if source_space not in {"whitened", "raw"}:
            raise ValueError("unsupported source latent prediction space")
        source_pooling = payload["data"].get("current_visual_pooling", "canvas_v1")
        adaptation = (make_posttrain_transition(source, payload,
            current_visual_pooling=reference.data.current_visual_pooling, calibration_batches=0)
            if reference.data.current_visual_pooling != source_pooling else None)
        training = replace(training, init_latent_prediction_migration=migration,
                           init_posttrain_migration=adaptation)
        model = replace(model, latent_transition=replace(model.latent_transition,
            output_affine=bool(migration) or payload["model"]["latent_transition"].get("output_affine", False)))
    training = replace(training, latent_prediction_space="raw", normalizer_calibration_updates=0,
        max_updates=max_updates, forecast_warmup_updates=forecast_warmup_updates,
        forecast_ramp_updates=forecast_ramp_updates)
    if initialization == "pretrained" and training.init_objective_migration is not None:
        from psi.models.predictive_world.objective_transition import make_objective_transition
        training = replace(training, init_objective_migration=make_objective_transition(
            source, payload, training, training.init_objective_migration["variant_id"]))
    result = replace(reference, model=model, training=training,
        runtime=replace(reference.runtime, output_dir=str(output_dir), allow_existing_output=False))
    result.validate()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", required=True, type=Path)
    parser.add_argument("--initialization", required=True, choices=("scratch", "pretrained"))
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--forecast-warmup-updates", type=int, default=0)
    parser.add_argument("--forecast-ramp-updates", type=int, default=0)
    args = parser.parse_args()
    reference, _ = load_predictive_world_config(args.reference_config, allow_existing_output_override=True)
    config = raw_latent_config(reference, initialization=args.initialization,
        output_dir=args.output_dir, max_updates=args.updates,
        forecast_warmup_updates=args.forecast_warmup_updates,
        forecast_ramp_updates=args.forecast_ramp_updates)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output_config.parent,
                                         prefix=f".{args.output_config.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(asdict(config), indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic and exclusive: readers see complete JSON; prior configs are untouched.
        os.link(temporary, args.output_config)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"config": str(args.output_config.resolve()), "initialization": args.initialization,
        "latent_prediction_space": "raw", "normalizer_calibration_updates": 0,
        "max_updates": config.training.max_updates, "training_started": False}))


if __name__ == "__main__":
    main()
