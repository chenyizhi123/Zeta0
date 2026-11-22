#!/usr/bin/env python3
"""Prepare an immutable native-protocol release and joint-initialized run config."""
from __future__ import annotations

import argparse
from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys
import uuid

from psi.config.predictive_world import PredictiveWorldDataConfig, predictive_world_config_from_payload
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.assets import load_asset_path_map, resolve_asset_path
from psi.data.predictive_world.index import publish_predictive_world_index, temporal_abi_from_data_config, verify_predictive_world_index
from psi.data.predictive_world.measurement_schema import load_measurement_registry, load_capability_registry
from psi.data.predictive_world.sonic_post_training import TASKS


def immutable_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"immutable artifact differs; use a new output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def main(*, dataset_format="sonic"):
    if dataset_format == "sonic":
        from psi.data.predictive_world.sonic_post_training import inspect_dataset, extend_registries, episode_windows
    elif dataset_format == "real_sonic":
        from psi.data.predictive_world.real_sonic_post_training import inspect_dataset, extend_registries, episode_windows
    elif dataset_format == "sonic_lerobot":
        from psi.data.predictive_world.sonic_lerobot_post_training import inspect_dataset, extend_registries, episode_windows
    else:
        raise ValueError(f"unsupported dataset format: {dataset_format}")
    parser = argparse.ArgumentParser(description=f"Prepare an immutable {dataset_format} native-protocol joint run.")
    parser.add_argument("--task", choices=TASKS if dataset_format == "sonic" else None)
    parser.add_argument("--dataset")
    parser.add_argument("--exclude-episode", action="append", type=int, default=[],
                        help="Original real-data episode index to exclude from samples and fitted statistics.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-joint", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--asset-path-map")
    parser.add_argument("--qwen-model", help="Explicit replacement frozen Qwen VLM; rebuild all task features.")
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--updates", type=int, default=100000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--checkpoint-retention", type=int, default=2)
    parser.add_argument("--first-checkpoint-update", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--rtc-training-max-delay", type=int, default=6)
    parser.add_argument("--training-variant", choices=["FULL", "FULL+ERT"], default="FULL",
                        help="Explicit response objective; FULL+ERT uses the final dual-reader contract v5.")
    parser.add_argument("--current-state-only", action="store_true",
                        help="Use one current state; explicitly migrate learned current-position rows.")
    parser.add_argument("--fixed-action-grid", action="store_true",
                        help="Model action conditions use fixed execution slots; retain logged times for auditing.")
    parser.add_argument("--independent-future-components", action="store_true",
                        help="Resample real future motion/left/right labels independently; body-only future measurements.")
    parser.add_argument("--collector-snapshots", action="store_true",
                        help="Keep the original paired observation row and collector timing limits; supervise held commands.")
    parser.add_argument("--recorded-action-projection", action="store_true",
                        help="Bind execution motion bounds to actual recorded training tokens, preserving every label.")
    parser.add_argument("--snapshot-max-age-ms", type=float,
                        help="Explicit training/deployment cutoff for unbounded collector v3; retain the source contract.")
    parser.add_argument("--real-sample-filter", choices=["deploy_input_v1", "collector_snapshot_v1"],
                        help="Exclude real samples violating live inputs/action bounds; save reasons.")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--diagnostic-processor-report")
    parser.add_argument("--processor-migration")
    parser.add_argument("--diagnostic-anchors-per-split", type=int, default=8)
    parser.add_argument("--diagnostic-episodes-per-split", type=int, default=1)
    parser.add_argument("--whole-body-target", action="store_true",
                        help="Supervise measured G1 arm14/leg12/waist3 using an explicit checkpoint migration.")
    parser.add_argument("--content-area-pooling", action="store_true",
                        help="Use the same content-area pooling as future features for current observations.")
    parser.add_argument("--latent-calibration-batches", type=int, default=0,
                        help="Train-only batches per rank to fit frozen initial E before optimizer updates.")
    args = parser.parse_args()
    if args.latent_calibration_batches < 0:
        parser.error("--latent-calibration-batches must be nonnegative")
    if (args.whole_body_target or args.content_area_pooling or args.latent_calibration_batches) and dataset_format != "sonic_lerobot":
        parser.error("this post-training adaptation requires the audited SONIC LeRobot state41 recordings")
    if args.exclude_episode and dataset_format != "real_sonic":
        parser.error("--exclude-episode requires real_sonic")
    if args.fixed_action_grid and dataset_format not in {"real_sonic", "sonic_lerobot"}:
        parser.error("--fixed-action-grid requires a supported real recording format")
    if (args.qwen_model or args.action_horizon is not None) and dataset_format != "sonic_lerobot":
        parser.error("VLM/action-grid replacement currently requires sonic_lerobot")
    if dataset_format == "sonic_lerobot" and (
        not args.qwen_model or not args.current_state_only or not args.fixed_action_grid
        or args.diagnostic_processor_report
    ):
        parser.error("sonic_lerobot requires replacement --qwen-model, current-state, fixed action grid")
    if args.independent_future_components and (dataset_format != "real_sonic" or not args.fixed_action_grid):
        parser.error("--independent-future-components requires real_sonic --fixed-action-grid")
    if args.real_sample_filter and (dataset_format != "real_sonic" or not args.current_state_only):
        parser.error("--real-sample-filter requires real_sonic --current-state-only")
    if args.collector_snapshots and (
        dataset_format != "real_sonic" or not args.current_state_only or not args.fixed_action_grid
        or not args.independent_future_components or args.real_sample_filter != "collector_snapshot_v1"
    ):
        parser.error("--collector-snapshots requires real_sonic/current-state/fixed-grid/independent-components and collector filter")
    if args.real_sample_filter == "collector_snapshot_v1" and not args.collector_snapshots:
        parser.error("collector filtering requires --collector-snapshots")
    if args.recorded_action_projection and not args.collector_snapshots:
        parser.error("--recorded-action-projection requires --collector-snapshots")
    if args.snapshot_max_age_ms is not None and not args.collector_snapshots:
        parser.error("--snapshot-max-age-ms requires --collector-snapshots")
    if not (args.dataset and args.task):
        parser.error("requires --dataset and --task")
    if min(args.gpus, args.accumulation, args.batch) <= 0:
        parser.error("GPU count, accumulation and micro batch must be positive")
    effective_batch = args.gpus * args.accumulation * args.batch
    if args.validation_fraction == 0 and args.validation_every:
        parser.error("validation requires --validation-fraction > 0")
    if args.diagnostic_processor_report and args.validation_fraction == 0:
        parser.error("processor diagnostic requires an explicit held-out --validation-fraction")
    if args.first_checkpoint_update <= 0:
        parser.error("--first-checkpoint-update must be positive")
    if args.checkpoint_retention <= 0:
        parser.error("--checkpoint-retention must be positive")
    if args.processor_migration and args.diagnostic_processor_report:
        parser.error('choose diagnostic preflight or approved migration, not both')
    asset_map = load_asset_path_map(args.asset_path_map)
    # Historical source configs identify weights; they need not be runnable under
    # today's training-startup rules. Validate their data ABI and exact checksum.
    source_payload = json.loads(Path(args.source_config).read_text())
    source_data_payload = dict(source_payload["data"])
    # Pretraining's node-owned video sampler is replaced by the task occurrence sampler.
    source_data_payload.pop("node_owned_video_sampling", None)
    for name in ("history_visual_offsets_s", "future_group_ends_s"):
        source_data_payload[name] = tuple(source_data_payload[name])
    source_data = PredictiveWorldDataConfig(**source_data_payload)
    source_data.validate()
    action_horizon = args.action_horizon if args.action_horizon is not None else source_data.action_horizon
    if type(action_horizon) is not int or action_horizon < 2:
        parser.error("action horizon must be an integer >= 2")
    if not 0 <= args.rtc_training_max_delay < action_horizon:
        raise ValueError("rtc-training-max-delay must be in [0, action_horizon-1]")
    source_sidecar = json.loads((Path(args.init_joint) / "predictive_world_checkpoint.json").read_text())
    if (source_sidecar["phase"] != "joint" or not (Path(args.init_joint) / ".complete").is_file()
            or canonical_sha256(source_payload) != source_sidecar["metadata"]["resolved_config_sha256"]):
        raise ValueError("initialization checkpoint and source config must identify the same complete joint run")
    migration = None
    if args.processor_migration:
        from psi.data.predictive_world.processor_migration import (
            load_processor_migration, verify_processor_migration, migration_target_provenance,
        )
        migration = load_processor_migration(args.processor_migration)
        verify_processor_migration(migration, source_sidecar,
                                   migration['diagnostic']['target_processor_id'], args.source_config)
        if args.updates > migration['max_updates'] or effective_batch > migration['max_effective_batch']:
            raise ValueError('requested budget exceeds approved migration')
    sampling = ({"future_component_alignment": "independent_components_v1"}
                if args.independent_future_components else {})
    if args.collector_snapshots:
        sampling.update(real_observation_mode="collector_snapshot_v1", real_action_label_mode="held_command_v1")
    if args.recorded_action_projection:
        sampling["recorded_action_projection"] = True
    if args.exclude_episode:
        sampling["excluded_episodes"] = args.exclude_episode
    info, instruction, episodes, audit, statistics = inspect_dataset(
        args.dataset, task=args.task, validation_fraction=args.validation_fraction, **sampling)
    if args.audit_only:
        print(json.dumps({"task": args.task, "episodes": len(audit["episodes"]) if dataset_format == "real_sonic" else len(episodes),
                          "contiguous_segments": len(episodes),
                          "train_episodes": len(audit["train_episodes"]),
                          "validation_episodes": len(audit["validation_episodes"]),
                          "frames": sum(len(ep["times"]) for ep in episodes),
                          "audit_identity": canonical_sha256(audit), "numeric_data": "finite"}))
        return
    root = Path(args.output).resolve()
    artifacts = root / "artifacts"
    source_registries = {name: json.loads(resolve_asset_path(source_payload["data"][name], asset_map).read_text()) for name in (
        "action_protocol_registry", "state_schema_registry", "measurement_registry")}
    registries = extend_registries(source_registries, info, statistics,
                                   task=args.task, audit_identity=canonical_sha256(audit))
    body_migration = None
    if args.whole_body_target:
        from psi.models.predictive_world.body_target_transition import (
            REAL_SCHEMA, real_whole_body_schema, make_body_target_transition,
        )
        capability = registries["capability_registry"][args.task]
        previous = registries["measurement_registry"][capability["measurement_schema_id"]]
        if REAL_SCHEMA in registries["measurement_registry"]:
            raise ValueError("whole-body schema already exists; use an exact resume")
        registries["measurement_registry"][REAL_SCHEMA] = real_whole_body_schema(
            previous, info["features"]["observation.state"]["names"],
            statistics["body_mean"], statistics["body_std"])
        capability["measurement_schema_id"] = REAL_SCHEMA
        body_migration = make_body_target_transition(source_sidecar, source_payload,
            registries["measurement_registry"], schema=REAL_SCHEMA)
    paths = {name: artifacts / f"{name}.json" for name in registries}
    for name, value in registries.items():
        immutable_json(paths[name], value)
    immutable_json(artifacts / "dataset_audit.json", audit)
    schemas = load_measurement_registry(paths["measurement_registry"])
    capabilities = load_capability_registry(paths["capability_registry"], measurement_schemas=schemas,
                                            action_protocol_ids=set(registries["action_protocol_registry"]))
    target_data = replace(source_data, state_history_count=1) if args.current_state_only else source_data
    if dataset_format == "sonic_lerobot":
        target_data = replace(target_data, action_horizon=action_horizon, action_hz=float(info["fps"]))
    if args.independent_future_components:
        target_data = replace(target_data, future_component_alignment="independent_components_v1")
    if args.collector_snapshots:
        from psi.real_observation_contract import snapshot_age_limit_s
        explicit_age = None if args.snapshot_max_age_ms is None else args.snapshot_max_age_ms / 1000
        age = snapshot_age_limit_s(audit["real_observation_contract"], explicit_age)
        target_data = replace(target_data, real_observation_mode="collector_snapshot_v1",
            real_observation_contract=audit["real_observation_contract"], real_action_label_mode="held_command_v1",
            real_snapshot_max_age_s=explicit_age, state_max_age_s=age, visual_max_age_s=age,
            future_measurement_max_age_s=.04, future_visual_max_age_s=.08)
    if args.recorded_action_projection:
        target_data = replace(target_data, real_action_projection=audit["real_action_projection"])
    temporal = temporal_abi_from_data_config(target_data)
    state_migration = None
    if args.current_state_only and source_data.state_history_count != 1:
        from psi.models.predictive_world.state_history_transition import current_state_transition
        state_migration = current_state_transition(source_sidecar, source_payload)
    binding = {"measurement_registry_sha256": file_sha256(paths["measurement_registry"]),
               "capability_registry_sha256": file_sha256(paths["capability_registry"]),
               "action_protocol_registry_sha256": file_sha256(paths["action_protocol_registry"]),
               "state_schema_registry_sha256": file_sha256(paths["state_schema_registry"])}
    provenance = {"registry_binding": binding, "dataset_audit_identity": canonical_sha256(audit),
                  "anchor_selection": ("every_recorded_decision_in_contiguous_segment_v1" if dataset_format == "real_sonic"
                                       else "every_native_recorded_row_v1" if dataset_format == "sonic_lerobot"
                                       else "every_native_frame_50hz_v2"), "statistics_split": "train",
                  "time_alignment": audit["time_alignment"]}
    if target_data.real_snapshot_max_age_s is not None:
        provenance["time_alignment"] = {**audit["time_alignment"],
            "real_snapshot_max_age_s": target_data.real_snapshot_max_age_s}
    selector = None
    if args.real_sample_filter:
        from psi.data.predictive_world.real_sonic_selection import RealSonicSampleSelector, selection_contract
        selector = RealSonicSampleSelector(args.dataset, selection_contract(target_data, audit, policy=args.real_sample_filter))
        provenance["sample_selection"] = selector.contract
    diagnostic = None
    selection = None
    if migration is not None and dataset_format != "sonic_lerobot":
        provenance['processor_migration'] = migration
    if args.diagnostic_processor_report:
        from psi.data.predictive_world.processor_diagnostic import (
            processor_diagnostic_contract, diagnostic_anchor_selection, verify_processor_diagnostic,
        )
        diagnostic = processor_diagnostic_contract(args.diagnostic_processor_report)
        verify_processor_diagnostic(diagnostic, source_sidecar,
                                    diagnostic['target_processor_id'], args.source_config)
        if args.batch > 2:
            raise ValueError('diagnostic processor trial requires batch <= 2')
        selection = diagnostic_anchor_selection(episodes, args.diagnostic_anchors_per_split,
                                               episodes_per_split=args.diagnostic_episodes_per_split)
        provenance['anchor_selection'] = selection
        provenance['processor_diagnostic'] = diagnostic
    index = artifacts / "predictive_world_index.jsonl"
    if index.exists():
        meta = verify_predictive_world_index(index)
        if meta["provenance"] != provenance or meta["temporal_abi"] != temporal:
            raise ValueError("existing task index has a different identity")
        if selector is not None:
            selector.verify_existing(artifacts, meta)
    else:
        def rows():
            for ep in episodes:
                selected = None if selection is None else selection['episode_frames'].get(str(ep['episode']))
                if selection is not None and selected is None:
                    continue
                task = args.task
                capability = capabilities[task]
                windows = episode_windows(ep, task=task, instruction=instruction, temporal=temporal,
                                          capability=capability, measurement_schema=schemas[capability.measurement_schema_id])
                yield from (row for row in windows if selected is None or row['anchor_ordinal'] in selected)
        published_rows = rows() if selector is None else selector.select(rows(), artifacts)
        # Real episodes/segments and their anchors are already in canonical order.
        # Stream them so large three-camera releases do not retain every window.
        publish_predictive_world_index(index, rows=published_rows, provenance=provenance, temporal_abi=temporal,
                                       presorted=dataset_format in {"real_sonic", "sonic_lerobot"})
    feature_path = resolve_asset_path(source_data.feature_cache, asset_map)
    if feature_path.is_dir():
        feature_path /= "feature_manifest.json"
    feature_provenance = json.loads(feature_path.read_text())["provenance"]
    # Feature extraction identity is inherited; source-release reuse hints are not.
    feature_provenance = {key: value for key, value in feature_provenance.items()
                          if key in {"encoder_id", "processor_id", "image_preprocessing", "feature_layer", "dtype", "feature_shape"}}
    feature_provenance["video_decoder"] = "torchcodec"
    if migration is not None:
        feature_provenance = migration_target_provenance(
            feature_provenance, migration, source_sidecar, args.source_config)
    if diagnostic is not None:
        feature_provenance['processor_id'] = diagnostic['target_processor_id']
        feature_provenance['processor_diagnostic'] = diagnostic
    vlm_action_migration = None
    if args.qwen_model:
        from transformers import AutoConfig, AutoProcessor
        from psi.data.lerobot.action_aligned_cache import build_action_aligned_vlm_processor_fingerprint
        from psi.data.predictor_pretrain.runtime import fingerprint_model_directory
        from psi.models.action_aligned_vlm_features import resolve_qwen3vl_visual_feature_spec
        from psi.models.predictive_world.vlm_action_transition import make_vlm_action_transition
        qwen_path = str(Path(args.qwen_model).resolve())
        qwen_config = AutoConfig.from_pretrained(qwen_path, local_files_only=True)
        resolve_qwen3vl_visual_feature_spec(qwen_config)
        processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
        feature_provenance = dict(encoder_id="qwen3vl:" + fingerprint_model_directory(qwen_path),
            processor_id="qwen3vl-processor:" + canonical_sha256(build_action_aligned_vlm_processor_fingerprint(processor)),
            image_preprocessing="aspect_v1", feature_layer="post_merger_content_area_overlap_4x4_v1",
            dtype="float16", feature_shape=[16, 2048], video_decoder="torchcodec")
        compact = dict(history_visual_offsets_s=list(target_data.history_visual_offsets_s),
            **target_data.observation_contract, state_history_count=target_data.state_history_count,
            state_query_hz=target_data.state_query_hz, future_group_ends_s=list(target_data.future_group_ends_s),
            action_horizon=target_data.action_horizon, action_hz=target_data.action_hz)
        vlm_action_migration = make_vlm_action_transition(source_sidecar, source_payload,
            encoder_id=feature_provenance["encoder_id"], processor_id=feature_provenance["processor_id"],
            temporal=compact, qwen_model=qwen_path,
            processor_migration=migration, source_config=args.source_config)
        feature_provenance["vlm_action_migration"] = vlm_action_migration
        if migration is not None:
            feature_provenance["processor_migration"] = migration
    posttrain_migration = None
    if args.content_area_pooling or args.latent_calibration_batches:
        from psi.models.predictive_world.posttrain_transition import make_posttrain_transition
        posttrain_migration = make_posttrain_transition(source_sidecar, source_payload,
            current_visual_pooling="content_area_v1" if args.content_area_pooling else "canvas_v1",
            calibration_batches=args.latent_calibration_batches)
    immutable_json(artifacts / "feature_provenance.json", feature_provenance)
    config = copy.deepcopy(source_payload)
    config["data"].pop("node_owned_video_sampling", None)
    config["training"].pop("from_scratch", None)
    if args.qwen_model:
        config["model"]["qwen_model"] = qwen_path
        config["data"].update(action_horizon=action_horizon, action_hz=target_data.action_hz)
    config["data"].update({name: str(path) for name, path in paths.items() if name != "capability_registry"})
    config["data"].update(canonical_future_capabilities=str(paths["capability_registry"]),
                           index_path=str(index), feature_cache=str(artifacts / "feature_cache"))
    if args.current_state_only:
        config["data"]["state_history_count"] = 1
    if args.independent_future_components:
        config["data"]["future_component_alignment"] = "independent_components_v1"
    if args.collector_snapshots:
        for field in ("real_observation_mode", "real_observation_contract", "real_action_label_mode",
                      "state_max_age_s", "visual_max_age_s", "future_measurement_max_age_s", "future_visual_max_age_s"):
            config["data"][field] = getattr(target_data, field)
        if target_data.real_snapshot_max_age_s is not None:
            config["data"]["real_snapshot_max_age_s"] = target_data.real_snapshot_max_age_s
    if args.recorded_action_projection:
        config["data"]["real_action_projection"] = target_data.real_action_projection
    if posttrain_migration is not None:
        config["data"]["current_visual_pooling"] = posttrain_migration["current_visual_pooling"]
        config["model"]["latent_transition"]["output_affine"] = bool(args.latent_calibration_batches)
    config["training"].update(
        phase="joint", joint_from_scratch=False, normalizer_calibration_updates=0,
        forecast_warmup_updates=0, forecast_ramp_updates=0,
        world_model=None, init_joint=str(Path(args.init_joint).resolve()),
        init_source_config=str(Path(args.source_config).resolve()), seed=292285,
        rtc_training_max_delay=args.rtc_training_max_delay,
        micro_batch=args.batch, gradient_accumulation_steps=args.accumulation, effective_global_batch=effective_batch,
        max_updates=args.updates, learning_rate=1e-4, joint_world_learning_rate=3e-5,
        weight_decay=1e-6, adam_betas=[0.95, 0.999], lr_schedule="cosine", warmup_updates=1000,
        checkpoint_every=args.checkpoint_every, checkpoint_retention=args.checkpoint_retention,
        first_checkpoint_update=args.first_checkpoint_update,
        log_every=10, log_both_modes=True, validation_every=args.validation_every,
        validation_micro_batch=16, validation_batches=20, validation_seed=2026,
        # A source config records how its *own* checkpoint was initialized.
        # It is not an authorization to apply that historical processor
        # transition again when this checkpoint already has the target processor.
        init_processor_diagnostic=None, init_processor_migration=None,
        init_posttrain_migration=posttrain_migration, init_body_target_migration=body_migration,
        init_state_history_migration=None, init_observation_contract_migration=None,
        init_vlm_action_migration=None, init_objective_migration=None,
    )
    if state_migration is not None:
        config["training"]["init_state_history_migration"] = state_migration
    if vlm_action_migration is not None:
        config["training"]["init_vlm_action_migration"] = vlm_action_migration
    if args.collector_snapshots:
        from psi.models.predictive_world.observation_contract_transition import observation_contract_transition
        compact_temporal = dict(history_visual_offsets_s=list(target_data.history_visual_offsets_s),
            **target_data.observation_contract, state_history_count=target_data.state_history_count,
            state_query_hz=target_data.state_query_hz, future_group_ends_s=list(target_data.future_group_ends_s),
            action_horizon=target_data.action_horizon, action_hz=target_data.action_hz)
        config["training"]["init_observation_contract_migration"] = observation_contract_transition(
            source_sidecar, source_payload, compact_temporal)
    if args.fixed_action_grid:
        config["training"]["action_condition_time_mode"] = "fixed_action_grid_v1"
    config["runtime"].update(nodes=1, gpus_per_node=args.gpus, output_dir=str(root / "joint"))
    if diagnostic is not None:
        config['training'].update(init_processor_diagnostic=diagnostic, log_every=1,
                                  validation_micro_batch=2, validation_batches=1)
    if migration is not None:
        config['training']['init_processor_migration'] = migration
    if asset_map:
        config["runtime"]["asset_path_map"] = asset_map
    if args.training_variant == "FULL+ERT":
        from dataclasses import asdict
        from psi.config.predictive_world import PredictiveWorldTrainingConfig
        from psi.models.predictive_world.objective_transition import full_ert_training, make_objective_transition
        training = PredictiveWorldTrainingConfig(**config["training"])
        if training.objective_contract_version == 5 and training.forecast_control_reader == "both":
            # The joint-29000 source already uses FULL+ERT; preserve its exact objective.
            from psi.config.predictive_world import predictive_world_loss_contract
            if predictive_world_loss_contract(training) != source_sidecar["metadata"]["loss_contract"]:
                raise ValueError("source FULL+ERT loss contract disagrees with the requested continuation")
        else:
            training = full_ert_training(training)
            training = replace(training, init_objective_migration=make_objective_transition(
                source_sidecar, source_payload, training, "FULL+ERT"))
        config["training"] = asdict(training)
    predictive_world_config_from_payload(config, allow_existing_output_override=True)
    immutable_json(root / "config.json", config)
    preflight = copy.deepcopy(config)
    preflight["runtime"]["output_dir"] = str(root / "preflight")
    immutable_json(root / "preflight.json", preflight)
    repo = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, str(repo / "scripts/data/audit_predictive_world_sources.py"),
                    "--measurement-registry", str(paths["measurement_registry"]),
                    "--capability-registry", str(paths["capability_registry"]),
                    "--action-protocol-registry", str(paths["action_protocol_registry"]),
                    "--state-schema-registry", str(paths["state_schema_registry"]),
                    "--index", str(index), "--output", str(artifacts / "source_audit.json")], check=True)
    print(json.dumps({"event": f"{dataset_format}_task_prepared", "task": args.task, "config": str(root / "config.json")}))


if __name__ == "__main__":
    main(dataset_format="sonic_lerobot")
