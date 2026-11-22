#!/usr/bin/env python3
"""Two-stage DDP entry point for world pretraining and joint action adaptation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from itertools import islice
import json
import math
import os
import platform
import random
import signal
import subprocess
from time import perf_counter
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from psi.config.predictive_world import (
    load_predictive_world_config, predictive_world_architecture_contract,
    predictive_world_loss_contract,
)
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.assets import resolve_asset_path
from psi.data.predictive_world.dataset import (
    PredictiveWorldOccurrenceDataset,
    PredictiveWorldRowReader,
    assert_predictive_world_training_data_contract,
)
from psi.data.predictive_world.index import temporal_abi_from_data_config
from psi.data.predictive_world.runtime import PredictiveWorldTrainingCollator
from psi.models.predictive_world.checkpoint import (
    complete_predictive_world_bundle,
    prune_completed_checkpoints,
    load_predictive_world_bundle,
    predictive_world_code_fingerprint,
    save_predictive_world_bundle,
)
from psi.models.predictive_world.engine import (
    EngineConfig,
    engine_config_identity,
    load_engine_state,
    move_to_device,
    publish_run_contract,
    run_phase_training,
    save_engine_state,
)
from psi.models.predictive_world.factory import build_phase
from psi.models.predictive_world.monitoring import evaluate_joint, training_log_due
from psi.models.predictive_world.training import assert_optimizer_partition, module_state_sha256


class _AlternatingForecastIterator:
    """Select a prebuilt eligible occurrence stream for every whole optimizer update."""

    def __init__(
        self,
        prior: Iterator,
        world: Iterator,
        *,
        accumulation_steps: int,
        start_update: int,
    ) -> None:
        self.prior = prior
        self.world = world
        self.accumulation_steps = int(accumulation_steps)
        self.start_update = int(start_update)
        self.microbatches_yielded = 0
        if self.accumulation_steps <= 0 or self.start_update < 0:
            raise ValueError("invalid alternating forecast iterator state")

    def __iter__(self):
        return self

    def __next__(self):
        update = self.start_update + self.microbatches_yielded // self.accumulation_steps
        source = self.prior if update % 2 == 0 else self.world
        self.microbatches_yielded += 1
        return next(source)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--parquet-cache-capacity", type=int, default=2)
    parser.add_argument("--video-cache-capacity", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-updates", type=int, default=3)
    parser.add_argument("--save-diagnostic-checkpoint", action="store_true")
    parser.add_argument("--verify-preflight-checkpoint", action="store_true",
        help="Save a diagnostic-only checkpoint and verify model/optimizer/RNG/cursor restoration during bounded preflight")
    parser.add_argument("--minimum-memory-headroom-gib", type=float, default=8.0)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--effective-global-batch", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _init_dist(*, timeout_s: int) -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_s))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world, device


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        cwd=Path(__file__).resolve().parents[3],
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _metadata(config, payload, built, *, reference_id: str) -> dict:
    feature_manifest = Path(config.data.feature_cache)
    if feature_manifest.is_dir():
        feature_manifest = feature_manifest / "feature_manifest.json"
    feature_payload = json.loads(feature_manifest.read_text(encoding="utf-8"))
    index_path = Path(config.data.index_path)
    index_meta = json.loads(
        index_path.with_suffix(index_path.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = {
        "architecture": predictive_world_architecture_contract(config.model),
        "temporal_abi": {
            "history_visual_offsets_s": list(config.data.history_visual_offsets_s),
            **config.data.observation_contract,
            "state_history_count": config.data.state_history_count,
            "state_query_hz": config.data.state_query_hz,
            "future_group_ends_s": list(config.data.future_group_ends_s),
            "action_horizon": config.data.action_horizon,
            "action_hz": config.data.action_hz,
        },
        "encoder_id": str(feature_payload["provenance"]["encoder_id"]),
        "processor_id": str(feature_payload["provenance"]["processor_id"]),
        "protocol_registry_sha256": file_sha256(Path(config.data.action_protocol_registry)),
        "measurement_registry_sha256": file_sha256(Path(config.data.measurement_registry)),
        "normalization_id": module_state_sha256(built.reference_modules["normalizer"]),
        "reference_id": reference_id,
        "data_fingerprint": canonical_sha256({
            # The content digest was fully checked at startup; consuming the
            # immutable sidecar avoids rescanning a multi-GB index at every
            # checkpoint.
            "index": str(index_meta["index_sha256"]),
            "feature_cache": feature_payload["cache_identity"],
            "capabilities": file_sha256(Path(config.data.canonical_future_capabilities)),
        }),
        "resolved_config_sha256": canonical_sha256(payload),
        "code_revision": _git_revision(),
        "code_fingerprint": predictive_world_code_fingerprint(Path(__file__)),
        "environment_fingerprint": canonical_sha256({
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }),
        "optimization_contract": json.loads(json.dumps({
            key: value for key, value in asdict(config.training).items()
            if (key not in {"joint_from_scratch", "joint_from_gr00t"} or value)
            and (key not in {"init_state_history_migration", "init_observation_contract_migration", "init_body_target_migration", "init_vlm_action_migration", "init_posttrain_migration", "init_latent_prediction_migration", "action_condition_time_mode"} or value is not None)
            and (key != "latent_prediction_space" or value != "whitened")
            and (key != "future_body_input_enabled" or config.training.objective_contract_version >= 6 or not value)
        })),
        "loss_contract": predictive_world_loss_contract(config.training),
    }
    if built.initialization is not None:
        metadata["initialization"] = dict(built.initialization)
    if config.training.init_processor_migration is not None:
        metadata['processor_migration'] = config.training.init_processor_migration
        metadata['production_accepted'] = False
    if built.parent_world_model is not None:
        metadata["parent_world_model"] = dict(built.parent_world_model)
    return metadata


def _reference_id(modules) -> str:
    return canonical_sha256({name: module_state_sha256(module) for name, module in modules.items()})


def main() -> None:
    args = _parser().parse_args()
    # Validate provenance before allocating resources or starting distributed work.
    # Platform launchers may run this absolute script path from outside the repo.
    _git_revision()
    config, payload = load_predictive_world_config(
        args.config,
        allow_existing_output_override=bool(args.resume),
        training_overrides={
            "micro_batch": args.micro_batch,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch": args.effective_global_batch,
        },
    )
    if args.preflight_only and args.resume:
        raise ValueError("memory preflight uses a fresh isolated run and cannot resume")
    if args.verify_preflight_checkpoint and (not args.preflight_only or not 3 <= args.preflight_updates <= 20):
        raise ValueError("preflight checkpoint verification requires 3..20 isolated preflight updates")
    from psi.data.predictive_world.processor_diagnostic import require_diagnostic_execution
    from psi.data.predictive_world.processor_migration import require_migration_budget
    require_migration_budget(config)
    require_diagnostic_execution(config, preflight_only=args.preflight_only,
                                 updates=args.preflight_updates, resume=args.resume)
    if args.save_diagnostic_checkpoint and (
        not args.preflight_only or config.training.init_processor_diagnostic is None
    ):
        raise ValueError('diagnostic saving requires an explicitly bound processor preflight')
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError("worker count must be nonnegative and prefetch factor positive")
    if args.parquet_cache_capacity <= 0 or args.video_cache_capacity <= 0:
        raise ValueError("reader cache capacities must be positive")
    if args.dry_run:
        print(json.dumps({
            "event": "predictive_world_dry_run",
            "phase": config.training.phase,
            "world_size": config.runtime.world_size,
            "micro_batch": config.training.micro_batch,
            "accumulation": config.training.gradient_accumulation_steps,
            "effective_global_batch": config.training.effective_global_batch,
            "output": str(Path(config.runtime.output_dir).resolve()),
            "data_io": {
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "parquet_cache_capacity": args.parquet_cache_capacity,
                "video_cache_capacity": args.video_cache_capacity,
            },
        }, sort_keys=True))
        return
    rank, world, device = _init_dist(timeout_s=config.runtime.process_group_timeout_s)
    telemetry = None
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        print(json.dumps({'event': 'training_loader_resources', 'rank': rank,
                          'num_workers': args.num_workers,
                          'prefetch_factor': args.prefetch_factor if args.num_workers else None,
                          'prefetched_batches_per_loader': args.num_workers * args.prefetch_factor,
                          'parquet_cache_capacity': args.parquet_cache_capacity,
                          'video_cache_capacity': args.video_cache_capacity,
                          'shared_memory_available_bytes': os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize
                          if Path('/dev/shm').exists() else None}), flush=True)
        if world != config.runtime.world_size:
            raise ValueError(f"launched world_size={world}, configured={config.runtime.world_size}")
        seed = 101 if config.training.seed is None else config.training.seed
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed + rank)
        artifacts = assert_predictive_world_training_data_contract(
            config.data,
            measurement_dim=config.model.measurement_dim,
            verify_content_hashes=rank == 0,
        )
        # Rank zero performs the full immutable index/lookup hashes.  Other
        # ranks check the same metadata bindings and wait here before opening
        # training data, avoiding 40 concurrent full scans of shared storage.
        if world > 1:
            dist.barrier()
        capabilities = artifacts.capabilities
        if (
            config.training.phase in {"world_model", "joint"}
            and not any(value.supports_world_action for value in capabilities.values())
        ):
            raise ValueError("world-model training has no audited action-capable source")
        built = build_phase(config, device=device, allow_diagnostic_migration=args.preflight_only)
        trainable = [parameter for parameter in built.model.parameters() if parameter.requires_grad]
        if config.training.phase == "joint":
            student_parameters = [
                parameter
                for parameter in built.checkpoint_modules["student"].parameters()
                if parameter.requires_grad
            ]
            student_ids = {id(parameter) for parameter in student_parameters}
            world_parameters = [
                parameter for parameter in trainable if id(parameter) not in student_ids
            ]
            if not student_parameters or not world_parameters:
                raise ValueError("joint training requires distinct world and action parameters")
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": world_parameters,
                        "lr": config.training.joint_world_learning_rate,
                        "group_name": "world",
                    },
                    {
                        "params": student_parameters,
                        "lr": config.training.learning_rate,
                        "group_name": "action",
                    },
                ],
                weight_decay=config.training.weight_decay,
                betas=tuple(config.training.adam_betas),
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable, lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                betas=tuple(config.training.adam_betas),
            )
        assert_optimizer_partition(
            optimizer, trainable=built.checkpoint_modules.values()
        )
        warmup, total = config.training.warmup_updates, config.training.max_updates
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, (step + 1) / max(warmup, 1))
            * (0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, (step-warmup) / max(total-warmup, 1)))))
               if config.training.lr_schedule == "cosine"
               else max(0.1, 1.0 - step / max(total, 1))),
        )
        identity = engine_config_identity(payload)
        start_update = next_occurrence = 0
        resume_bundle = None
        if args.resume:
            resume_bundle = load_predictive_world_bundle(
                args.resume, phase=config.training.phase, modules=built.checkpoint_modules,
                expected_metadata={"loss_contract": predictive_world_loss_contract(config.training)},
            )
            start_update, next_occurrence = load_engine_state(
                args.resume, optimizer=optimizer, scheduler=scheduler,
                config_identity=identity, device=device,
            )
        from psi.data.predictive_world.provenance import load_qwen_processor

        processor = load_qwen_processor(
            resolve_asset_path(config.model.qwen_model, config.runtime.asset_path_map))
        reader = PredictiveWorldRowReader(
            processor,
            artifacts.feature_manifest,
            index_path=config.data.index_path,
            observed_visual_mode=config.data.observed_visual_mode,
            current_visual_pooling=config.data.current_visual_pooling,
            capabilities=capabilities,
            measurement_schemas=artifacts.measurement_schemas,
            index_already_verified=True,
            parquet_cache_capacity=args.parquet_cache_capacity,
            video_cache_capacity=args.video_cache_capacity,
        )
        collator = PredictiveWorldTrainingCollator(
            future_measurement_offsets_s=temporal_abi_from_data_config(config.data)[
                "future_measurement_offsets_s"
            ],
            future_group_ends_s=config.data.future_group_ends_s,
            state_count=config.data.state_history_count,
            action_horizon=config.data.action_horizon,
            body_points_per_group=config.model.body_points_per_group,
            action_condition_time_mode=config.training.action_condition_time_mode,
            action_hz=config.data.action_hz,
        )
        def make_loader(*, start: int, require_world: bool = False, require_complete: bool = False, workers=None):
            dataset = PredictiveWorldOccurrenceDataset(
                config.data.index_path, reader, split="train", seed=seed,
                rank=rank, world_size=world, start_occurrence=start,
                rank_batch_size=config.training.micro_batch,
                require_world_action=require_world,
                require_complete_future=require_complete,
                source_sampling_exponent=config.data.source_sampling_exponent,
                index_already_verified=True,
            )
            workers = args.num_workers if workers is None else workers
            return DataLoader(
                dataset, batch_size=config.training.micro_batch, collate_fn=collator,
                num_workers=workers, pin_memory=device.type == "cuda",
                persistent_workers=workers > 0,
                **({'prefetch_factor': args.prefetch_factor} if workers else {}),
                timeout=(config.runtime.dataloader_timeout_s if workers > 0 else 0),
            )

        adaptation = config.training.init_posttrain_migration
        if adaptation is not None and adaptation["calibration_batches"]:
            if args.resume:
                if not bool(built.forecast_modules["latent_transition"].output_calibrated):
                    raise ValueError("resume checkpoint lacks completed latent calibration")
                built.initialization = resume_bundle["metadata"].get("initialization", built.initialization)
            else:
                from psi.models.predictive_world.posttrain_transition import calibrate_posttrain
                # Separate iterator: no optimizer, training occurrence, or EMA clock advances.
                # Use audited world rows and their masks, just as joint training does.
                # A 30 Hz anchor leaves the first 50 Hz body query masked.
                calibration_loader = make_loader(start=0, require_world=True, workers=0)
                report = calibrate_posttrain(built, calibration_loader,
                    batch_count=adaptation["calibration_batches"], device=device,
                    seed=seed, distributed=world > 1)
                report["config_identity"] = identity
                output = Path(config.runtime.output_dir).resolve()
                output.mkdir(parents=True, exist_ok=True)
                publish_run_contract(output / "latent_calibration.json", report, device=device)
                built.initialization["latent_calibration"] = report
                if rank == 0:
                    print(json.dumps({"event": "latent_calibration_complete", **report}), flush=True)
        loaders = []
        if config.training.phase == "world_model":
            prior_updates = (start_update + 1) // 2
            world_updates = start_update // 2
            prior_loader = make_loader(
                start=prior_updates * config.training.effective_global_batch
            )
            world_loader = make_loader(
                start=world_updates * config.training.effective_global_batch,
                require_world=True,
            )
            loaders.extend((prior_loader, world_loader))
            iterator = _AlternatingForecastIterator(
                iter(prior_loader), iter(world_loader),
                accumulation_steps=config.training.gradient_accumulation_steps,
                start_update=start_update,
            )
        elif config.training.phase == "joint":
            # Joint adaptation must provide an action target on every update,
            # while still retaining partially observed future/action horizons.
            # The audited world view requires at least one valid paired group,
            # but does not require an artificially complete episode tail.
            loader = make_loader(start=next_occurrence, require_world=True)
            loaders.append(loader)
            iterator = iter(loader)
        else:
            loader = make_loader(
                start=next_occurrence,
                require_complete=config.training.phase == "policy",
            )
            loaders.append(loader)
            iterator = iter(loader)
        wrapped = DistributedDataParallel(
            built.model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=config.runtime.ddp_gradient_as_bucket_view,
        ) if world > 1 else built.model
        # Start the monitoring thread only after DataLoader workers have been
        # created, so a fork does not inherit an active telemetry subprocess.
        if rank == 0 and device.type == 'cuda':
            from psi.data.predictive_world.performance import GpuTelemetry
            properties = torch.cuda.get_device_properties(device)
            identifier = getattr(properties, 'uuid', None)
            if identifier is None:
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
                identifier = visible[device.index] if visible[0] else str(device.index)
            telemetry = GpuTelemetry(str(identifier))
        if args.preflight_only:
            if device.type != "cuda" or args.preflight_updates < 3:
                raise ValueError("memory preflight requires CUDA and at least three updates")
            torch.cuda.reset_peak_memory_stats(device)
        output = Path(config.runtime.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        run_contract = {
            "format": "zeta0.predictive_world.run.v1",
            "resolved_config": payload,
            "config_identity": identity,
            "world_size": world,
            "rank_batch_size": config.training.micro_batch,
            "accumulation": config.training.gradient_accumulation_steps,
            "loss_contract": predictive_world_loss_contract(config.training),
        }
        contract_path = output / "run_contract.json"
        publish_run_contract(contract_path, run_contract, device=device)

        def validity(batch):
            if config.training.phase == "policy":
                valid = batch["actions"].valid.any(dim=(1, 2)) & batch["group_valid"].any(dim=1)
            else:
                valid = batch["group_valid"].any(dim=1)
            return valid, None

        def prepare_update(update, micros):
            for batch in micros:
                batch["_optimizer_update"] = update
                if args.preflight_only:
                    batch["_force_forecast_active"] = True
            if config.training.phase in {"world_model", "joint"}:
                mode = "prior" if update % 2 == 0 else "world"
                for batch in micros:
                    batch["forecast_mode"] = mode

        def forward(module, batch):
            return module(batch)

        if config.training.joint_from_scratch or config.training.joint_from_gr00t:
            reference_id = (resume_bundle["metadata"]["reference_id"] if resume_bundle is not None
                            else _reference_id(built.reference_modules))
        elif config.training.phase == "world_model":
            reference_id = _reference_id(built.reference_modules)
        elif built.initialization is not None:
            reference_id = built.initialization["reference_id"]
        else:
            reference_id = json.loads(
                (Path(config.training.world_model) / "predictive_world_checkpoint.json").read_text()
            )["metadata"]["reference_id"]

        source_path = args.resume or config.training.init_joint or config.training.world_model
        source_record = None
        if source_path:
            source_root = Path(source_path).resolve()
            sidecar = json.loads((source_root / "predictive_world_checkpoint.json").read_text())
            source_record = {
                "path": str(source_root),
                "phase": sidecar["phase"],
                "completed_updates": sidecar["completed_updates"],
                "weights_sha256": sidecar["weights_sha256"],
                "reference_id": sidecar["metadata"]["reference_id"],
                "normalization_id": sidecar["metadata"]["normalization_id"],
                "parent_world_model": sidecar["metadata"].get("parent_world_model"),
            }
        if rank == 0:
            event = {
                "event": "training_session_started",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "start_update": start_update,
                "load_kind": ("exact_resume" if args.resume else "joint_weight_initialization"
                              if config.training.init_joint else "random_task_initialization"
                              if config.training.joint_from_scratch else "gr00t_base_initialization"
                              if config.training.joint_from_gr00t else "phase_initialization"),
                "initialization": built.initialization,
                "optimizer_restored": bool(args.resume),
                "loaded_checkpoint": source_record,
                "config_identity": identity,
                "code_revision": _git_revision(),
                "code_fingerprint": predictive_world_code_fingerprint(Path(__file__)),
                "monitoring": {
                    "log_both_modes": config.training.log_both_modes,
                    "validation_every": config.training.validation_every,
                    "validation_batches": config.training.validation_batches,
                    "validation_micro_batch": config.training.validation_micro_batch,
                    "validation_seed": config.training.validation_seed,
                },
                "data_io": {
                    "num_workers": args.num_workers,
                    "prefetch_factor": args.prefetch_factor,
                    "parquet_cache_capacity": args.parquet_cache_capacity,
                    "video_cache_capacity": args.video_cache_capacity,
                },
            }
            with (output / "training_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

        def validate_update(update):
            if args.preflight_only or not config.training.validation_every:
                return
            if (config.training.joint_from_scratch and config.training.latent_prediction_space == "whitened"
                    and update == 0):
                return  # No held-out sample may fit the fresh training normalizer.
            if (update != start_update and update != config.training.max_updates
                    and update % config.training.validation_every):
                return
            # Fresh occurrence stream, fixed seed, no training-worker/cursor reuse.
            dataset = PredictiveWorldOccurrenceDataset(
                config.data.index_path, reader, split="validation",
                seed=config.training.validation_seed, rank=rank, world_size=world,
                start_occurrence=0, rank_batch_size=config.training.validation_micro_batch,
                require_world_action=True,
                source_sampling_exponent=config.data.source_sampling_exponent,
                index_already_verified=True,
            )
            def validation_rows():
                stream = iter(dataset)
                for _ in range(config.training.validation_batches):
                    yield list(islice(stream, config.training.validation_micro_batch))
            report = evaluate_joint(
                built.model, rows=validation_rows(), collator=collator, device=device,
                seed=config.training.validation_seed, update=update,
            )
            report.update({
                "format": "zeta0.predictive_world.validation.v1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "session_baseline" if update == start_update else "periodic_validation",
                "config_identity": identity,
                "loaded_checkpoint": source_record,
                "eligible_sources": list(dataset.source_ids),
                "unsampled_sources": sorted(set(dataset.source_ids) - {
                    group["source_id"] for group in report["groups"]
                }),
            })
            if rank == 0:
                with (output / "validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
                print(json.dumps({"event": "validation_complete", "update": update,
                                  "sample_count": report["sample_count"]}), flush=True)

        def log_update(update, metrics):
            record = {"update": update, "phase": config.training.phase, **metrics}
            record['hardware'] = telemetry.snapshot() if telemetry is not None else {'gpu_telemetry_status': 'not_sampled'}
            record["timestamp"] = datetime.now(timezone.utc).isoformat()
            record["config_identity"] = identity
            if config.training.phase in {"world_model", "joint"}:
                record.update({
                    "mode": "prior" if (update - 1) % 2 == 0 else "world",
                    "optimizer_updates_by_mode": {
                        "prior": (update + 1) // 2,
                        "world": update // 2,
                    },
                })
            record["learning_rates"] = {
                str(group.get("group_name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            with (output / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        def checkpoint(update):
            started = perf_counter()
            directory = output / f"checkpoint-{update:08d}"
            if rank == 0:
                print(f"checkpoint status=started update={update} path={directory}", flush=True)
            if rank == 0:
                checkpoint_reference_id = (
                    _reference_id(built.reference_modules)
                    if config.training.phase in {"world_model", "joint"}
                    else reference_id
                )
                metadata = _metadata(
                    config, payload, built, reference_id=checkpoint_reference_id
                )
                if args.save_diagnostic_checkpoint or args.verify_preflight_checkpoint:
                    metadata['artifact_use'] = 'diagnostic_only'
                    metadata['production_accepted'] = False
                save_predictive_world_bundle(
                    directory, phase=config.training.phase, modules=built.checkpoint_modules,
                    completed_updates=update, metadata=metadata, publish_complete=False,
                )
            if world > 1:
                dist.barrier()
            save_engine_state(
                directory, optimizer=optimizer, scheduler=scheduler,
                completed_updates=update,
                next_occurrence=update * config.training.effective_global_batch,
                config_identity=identity,
            )
            if rank == 0:
                complete_predictive_world_bundle(directory)
                print(f"checkpoint status=complete update={update} path={directory}", flush=True)
                if config.training.checkpoint_retention is not None:
                    removed = prune_completed_checkpoints(output, keep=config.training.checkpoint_retention)
                    print(json.dumps({"event": "checkpoint_retention", "removed": removed}), flush=True)
            if world > 1:
                dist.barrier()

            if rank == 0:
                event = {'event': 'checkpoint_timing', 'update': update,
                         'seconds': perf_counter() - started, 'path': str(directory),
                         'timestamp': datetime.now(timezone.utc).isoformat()}
                print(json.dumps(event, sort_keys=True), flush=True)
                with (output / 'performance_events.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(event, sort_keys=True) + '\n')

        completed_updates = (
            min(config.training.max_updates, args.preflight_updates)
            if args.preflight_only else config.training.max_updates
        )
        validate_update(start_update)
        final_update = run_phase_training(
            model=wrapped, batches=iterator, optimizer=optimizer, scheduler=scheduler,
            config=EngineConfig(
                phase=config.training.phase,
                max_updates=completed_updates,
                accumulation_steps=config.training.gradient_accumulation_steps,
                max_grad_norm=config.training.max_grad_norm,
                log_every=config.training.log_every,
                checkpoint_every=config.training.checkpoint_every,
                first_checkpoint_update=config.training.first_checkpoint_update,
            ),
            validity_fn=validity, forward_fn=forward, device=device,
            start_update=start_update,
            on_update=log_update,
            on_checkpoint=None if args.preflight_only else checkpoint,
            should_stop=lambda: stop_requested,
            prepare_update=prepare_update,
            on_step_end=validate_update,
            log_due=lambda update: training_log_due(
                update, config.training.log_every,
                config.training.log_both_modes and config.training.phase in {"world_model", "joint"},
            ),
        )
        if args.preflight_only:
            if final_update != completed_updates:
                raise InterruptedError(f'preflight stopped early: {final_update}/{completed_updates} updates')
            if args.save_diagnostic_checkpoint or args.verify_preflight_checkpoint:
                # Save at the update boundary, before validation consumes RNG/data.
                checkpoint(final_update)
                from psi.models.predictive_world.checkpoint_diagnostic import verify_diagnostic_roundtrip
                roundtrip = verify_diagnostic_roundtrip(
                    output / f'checkpoint-{final_update:08d}', modules=built.checkpoint_modules,
                    optimizer=optimizer, scheduler=scheduler, config_identity=identity,
                    completed_updates=final_update,
                    next_occurrence=final_update * config.training.effective_global_batch, device=device)
                if rank == 0:
                    with (output/'checkpoint_roundtrip_report.json').open('x') as handle:
                        json.dump(roundtrip, handle, indent=2, allow_nan=False)
                    print(json.dumps({'event': 'diagnostic_checkpoint_roundtrip', **roundtrip}), flush=True)
            train_allocated = torch.cuda.max_memory_allocated(device) / 2**30
            train_reserved = torch.cuda.max_memory_reserved(device) / 2**30
            torch.cuda.reset_peak_memory_stats(device)
            built.model.eval()
            validation_batch = move_to_device(next(iterator), device)
            if config.training.phase in {"world_model", "joint"}:
                validation_batch["forecast_mode"] = "prior"
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                validation_result = built.model(validation_batch)
                validation_result.validate()
            torch.cuda.synchronize(device)
            validation_allocated = torch.cuda.max_memory_allocated(device) / 2**30
            validation_reserved = torch.cuda.max_memory_reserved(device) / 2**30
            phase_peaks = torch.tensor(
                [train_allocated, train_reserved, validation_allocated, validation_reserved],
                device=device,
            )
            allocated = torch.tensor(max(train_allocated, validation_allocated), device=device)
            reserved = torch.tensor(max(train_reserved, validation_reserved), device=device)
            total_memory = torch.tensor(
                torch.cuda.get_device_properties(device).total_memory / 2**30, device=device
            )
            if world > 1:
                dist.all_reduce(phase_peaks, op=dist.ReduceOp.MAX)
                dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
                dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
                dist.all_reduce(total_memory, op=dist.ReduceOp.MIN)
            headroom = float(total_memory - reserved)
            train_allocated, train_reserved, validation_allocated, validation_reserved = (
                float(value) for value in phase_peaks
            )
            passed = headroom >= args.minimum_memory_headroom_gib
            diagnostic_result = None
            if config.training.init_processor_diagnostic is not None:
                from psi.models.predictive_world.training import module_state_sha256
                from psi.data.predictive_world.processor_diagnostic import unique_diagnostic_rows
                heldout = PredictiveWorldOccurrenceDataset(
                    config.data.index_path, reader, split='validation',
                    seed=config.training.validation_seed, rank=rank, world_size=world,
                    start_occurrence=0, rank_batch_size=config.training.micro_batch,
                    require_world_action=True,
                    source_sampling_exponent=config.data.source_sampling_exponent,
                    index_already_verified=True,
                )
                heldout_report = evaluate_joint(
                    built.model, rows=[unique_diagnostic_rows(iter(heldout), config.training.micro_batch)],
                    collator=collator, device=device, seed=config.training.validation_seed,
                    update=final_update,
                )
                heldout_report['selection'] = 'first_unique_occurrences_max_256_draws'
                normalizer_id = module_state_sha256(built.checkpoint_modules['normalizer'])
                if normalizer_id != built.initialization['normalization_id']:
                    raise ValueError('diagnostic training changed the fixed source normalizer')
                diagnostic_result = {
                    'processor_transition': config.training.init_processor_diagnostic,
                    'initialization': built.initialization,
                    'normalization_id_after_updates': normalizer_id,
                    'actual_completed_updates': final_update,
                    'heldout_validation': heldout_report,
                    'environment': {'python': platform.python_version(),
                                    'torch': torch.__version__, 'cuda': torch.version.cuda},
                    'production_accepted': False,
                }
            if rank == 0:
                report = {
                    "phase": config.training.phase,
                    "updates": completed_updates,
                    "peak_allocated_gib": float(allocated),
                    "peak_reserved_gib": float(reserved),
                    "training_peak_allocated_gib": train_allocated,
                    "training_peak_reserved_gib": train_reserved,
                    "validation_peak_allocated_gib": validation_allocated,
                    "validation_peak_reserved_gib": validation_reserved,
                    "minimum_total_gib": float(total_memory),
                    "reserved_headroom_gib": headroom,
                    "minimum_memory_headroom_gib": args.minimum_memory_headroom_gib,
                    "status": "passed" if passed else "failed",
                }
                if diagnostic_result is not None:
                    report['diagnostic'] = diagnostic_result
                (output / "memory_preflight_report.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n"
                )
                print(json.dumps({"event": "predictive_world_memory_preflight", **report}, sort_keys=True), flush=True)
            if not passed:
                raise MemoryError(
                    f"preflight reserved headroom {headroom:.2f} GiB is below "
                    f"{args.minimum_memory_headroom_gib:.2f} GiB; "
                    f"peak allocated={float(allocated):.2f} GiB, reserved={float(reserved):.2f} GiB"
                )
            return
        if final_update < config.training.max_updates:
            if rank == 0:
                (output / ".requeue_ready").write_text(f"{final_update}\n")
                print(json.dumps({
                    "event": "predictive_world_graceful_stop",
                    "completed_updates": final_update,
                    "checkpoint": str(output / f"checkpoint-{final_update:08d}"),
                }, sort_keys=True), flush=True)
            if world > 1:
                dist.barrier()
            raise SystemExit(75)
        if config.training.phase == "world_model":
            if rank == 0:
                final_reference_id = _reference_id(built.reference_modules)
                save_predictive_world_bundle(
                    output / "world-model-final", phase="world_model",
                    modules=built.checkpoint_modules,
                    completed_updates=config.training.max_updates,
                    metadata=_metadata(
                        config, payload, built, reference_id=final_reference_id
                    ),
                )
            if world > 1:
                dist.barrier()
        elif config.training.phase == "joint":
            if rank == 0:
                final_reference_id = _reference_id(built.reference_modules)
                save_predictive_world_bundle(
                    output / "joint-final", phase="joint",
                    modules=built.checkpoint_modules,
                    completed_updates=config.training.max_updates,
                    metadata=_metadata(
                        config, payload, built, reference_id=final_reference_id
                    ),
                )
            if world > 1:
                dist.barrier()
        elif world > 1:
            dist.barrier()
        if rank == 0:
            (output / ".train_complete").write_text(f"{config.training.max_updates}\n")
    finally:
        if telemetry is not None:
            telemetry.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
