"""Phase-specific production model construction from one resolved config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from psi.config.predictive_world import (
    PredictiveWorldConfig, predictive_world_architecture_contract,
    predictive_world_loss_contract,
)
from psi.data.predictive_world.cache import file_sha256
from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.models.predictive_world.action_expert import (
    TemporalNativeActionExpert,
    clone_independent_expert,
)
from psi.models.predictive_world.checkpoint import load_predictive_world_bundle
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.latent_transition import LatentNormalizer, TemporalLatentTransition
from psi.models.predictive_world.observation import FrozenQwenObservationEncoder
from psi.models.predictive_world.phases import (
    ForecastObjective,
    JointObjective,
    PhaseModelWithObservation,
    PolicyObjective,
    RepresentationObjective,
    WorldModelObjective,
)
from psi.models.predictive_world.representation import (
    EMAFutureTrajectoryEncoder,
    FutureTrajectoryEncoder,
    WorldReadout,
)
from psi.models.predictive_world.training import freeze_module
from psi.models.predictive_world.initialization import initialize_joint


@dataclass
class BuiltPhase:
    model: nn.Module
    checkpoint_modules: dict[str, nn.Module]
    reference_modules: dict[str, nn.Module]
    forecast_modules: dict[str, nn.Module]
    parent_world_model: dict[str, Any] | None = None
    initialization: dict[str, Any] | None = None


def _registry(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"registry must be a non-empty object: {path}")
    return value


def _state_registry(path: str) -> tuple[dict[str, int], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    payload = _registry(path)
    dimensions, statistics = {}, {}
    for name, value in payload.items():
        dimension = int(value["dim"])
        mean = torch.as_tensor(value["mean"], dtype=torch.float32)
        std = torch.as_tensor(value["std"], dtype=torch.float32)
        if mean.shape != (dimension,) or std.shape != (dimension,) or (std <= 0).any():
            raise ValueError(f"invalid state registry entry {name!r}")
        dimensions[str(name)] = dimension
        statistics[str(name)] = (mean, std)
    return dimensions, statistics


def _reference_modules(
    config: PredictiveWorldConfig,
    protocols: Mapping[str, Mapping[str, Any]],
    state_dims: Mapping[str, int],
    state_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    qwen_dim: int,
) -> dict[str, nn.Module]:
    model = config.model
    online_encoder = FutureTrajectoryEncoder(
        measurement_dim=model.measurement_dim,
        latent_shape=model.latent_shape,
        scene_slots=model.latent_scene_slots,
        hidden_dim=model.representation_hidden_dim,
        num_blocks=model.representation_blocks,
        num_heads=model.representation_heads,
        ff_dim=model.representation_ff_dim,
    )
    encoder = EMAFutureTrajectoryEncoder(
        online_encoder, decay=config.training.target_encoder_ema_decay
    )
    readout = WorldReadout(
        measurement_dim=model.measurement_dim,
        latent_shape=model.latent_shape,
        max_scene_patches=model.max_scene_patches,
        body_points_per_group=model.body_points_per_group,
        hidden_dim=model.representation_hidden_dim,
        num_heads=model.representation_heads,
        ff_dim=model.representation_ff_dim,
    )
    teacher = _action_expert(config, protocols, state_dims, state_stats, qwen_dim=qwen_dim)
    return {
        "encoder": encoder,
        "readout": readout,
        "teacher": teacher,
        "normalizer": LatentNormalizer(model.latent_shape),
    }


def _action_expert(config, protocols, state_dims, state_stats, *, qwen_dim):
    model = config.model
    if model.host_architecture == "gr00t_n1d7":
        from psi.data.predictive_world.assets import resolve_asset_path
        from psi.models.predictive_world.gr00t import load_action_expert
        path = resolve_asset_path(model.gr00t_model, config.runtime.asset_path_map)
        if file_sha256(path / "config.json") != model.gr00t_config_sha256:
            raise ValueError("GR00T base configuration identity mismatch")
        expert = load_action_expert(path, protocols, state_dims, state_stats, latent_shape=model.latent_shape)
        expert.activation_checkpointing = config.runtime.gr00t_activation_checkpointing
        if expert.action_steps != config.data.action_horizon or expert.head.config.backbone_embedding_dim != qwen_dim:
            raise ValueError("GR00T checkpoint dimensions disagree with the resolved training configuration")
        return expert
    return TemporalNativeActionExpert(
        protocols, state_dims, state_statistics=state_stats, qwen_dim=qwen_dim,
        context_tokens=model.expert_context_tokens, latent_dim=model.latent_shape[2],
        latent_groups=model.latent_shape[0], latent_slots=model.latent_shape[1],
        state_history_count=config.data.state_history_count,
        action_steps=config.data.action_horizon, hidden_dim=model.expert_hidden_dim,
        num_blocks=model.expert_blocks, num_heads=model.expert_heads,
        ff_dim=model.expert_ff_dim,
    )


def _forecast_modules(
    config: PredictiveWorldConfig,
    protocols: Mapping[str, Mapping[str, Any]],
    state_dims: Mapping[str, int],
    state_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    qwen_dim: int,
) -> dict[str, nn.Module]:
    model = config.model
    adapters = StateAdapterBank(dict(state_dims), statistics=dict(state_stats))
    history = PredictiveHistoryEncoder(
        qwen_dim=qwen_dim,
        state_history_count=config.data.state_history_count,
        output_tokens=16,
        context_tokens=model.history_context_tokens,
        hidden_dim=model.history_hidden_dim,
        num_blocks=model.history_blocks,
        num_heads=model.history_heads,
        ff_dim=model.history_ff_dim,
    )
    transition = TemporalLatentTransition(
        protocols,
        context_dim=history.hidden_dim,
        latent_shape=model.latent_shape,
        group_ends_s=config.data.future_group_ends_s,
        hidden_dim=model.latent_transition.hidden_dim,
        num_blocks=model.latent_transition.num_blocks,
        num_heads=model.latent_transition.num_heads,
        ff_dim=model.latent_transition.ff_dim,
        output_affine=model.latent_transition.output_affine,
    )
    return {
        "history_state_adapters": adapters,
        "history_encoder": history,
        "latent_transition": transition,
    }


def build_phase(config: PredictiveWorldConfig, *, device: torch.device,
                allow_diagnostic_migration: bool = False) -> BuiltPhase:
    """Construct only the modules required by one phase and load strict dependencies."""
    if config.training.init_processor_diagnostic is not None and not allow_diagnostic_migration:
        raise ValueError('processor transition config is restricted to diagnostic preflight')
    model = config.model
    from psi.data.predictive_world.processor_migration import require_migration_budget
    require_migration_budget(config)
    from transformers import Qwen3VLForConditionalGeneration
    from psi.data.predictive_world.assets import resolve_asset_path
    from psi.data.predictive_world.provenance import verify_qwen_provenance

    qwen_path = resolve_asset_path(config.model.qwen_model, config.runtime.asset_path_map)
    gr00t_initialization = None
    if model.host_architecture == "gr00t_n1d7":
        from psi.models.predictive_world.gr00t_assets import verify_backbone_export
        gr00t_initialization = verify_backbone_export(
            qwen_path, resolve_asset_path(model.gr00t_model, config.runtime.asset_path_map),
            expected_config_sha256=model.gr00t_config_sha256)
    feature_manifest = Path(config.data.feature_cache)
    if feature_manifest.is_dir():
        feature_manifest /= "feature_manifest.json"
    feature_provenance = json.loads(feature_manifest.read_text(encoding="utf-8"))["provenance"]
    if config.training.init_processor_migration is not None:
        if feature_provenance.get('processor_migration') != config.training.init_processor_migration:
            raise ValueError('feature cache is not bound to the approved processor migration')
    verify_qwen_provenance(qwen_path, feature_provenance)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if model.host_architecture == "gr00t_n1d7":
        from psi.models.predictive_world.gr00t_assets import load_backbone
        qwen = load_backbone(qwen_path, dtype=dtype, device=device)
    else:
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            str(qwen_path), local_files_only=True, dtype=dtype, attn_implementation="sdpa"
        ).to(device)
    qwen_dim = int(getattr(qwen.config, "text_config", qwen.config).hidden_size)
    observation = FrozenQwenObservationEncoder(
        qwen, **({"include_image_mask": True} if model.host_architecture == "gr00t_n1d7" else {}))
    protocols = _registry(config.data.action_protocol_registry)
    state_dims, state_stats = _state_registry(config.data.state_schema_registry)
    reference = _reference_modules(
        config, protocols, state_dims, state_stats, qwen_dim=qwen_dim
    )
    phase = config.training.phase
    forecast: dict[str, nn.Module] = {}
    parent_world_model: dict[str, Any] | None = None
    initialization = None
    temporal_abi = {
        "history_visual_offsets_s": list(config.data.history_visual_offsets_s),
        **config.data.observation_contract,
        "state_history_count": config.data.state_history_count,
        "state_query_hz": config.data.state_query_hz,
        "future_group_ends_s": list(config.data.future_group_ends_s),
        "action_horizon": config.data.action_horizon,
        "action_hz": config.data.action_hz,
    }
    expected = {
        "architecture": predictive_world_architecture_contract(config.model),
        "temporal_abi": temporal_abi,
        "encoder_id": str(feature_provenance["encoder_id"]),
        "processor_id": str(feature_provenance["processor_id"]),
        "protocol_registry_sha256": file_sha256(Path(config.data.action_protocol_registry)),
        "measurement_registry_sha256": file_sha256(Path(config.data.measurement_registry)),
    }
    if phase == "world_model":
        representation = RepresentationObjective(
            reference["encoder"], reference["readout"], reference["teacher"],
            scene_weight=config.training.scene_loss_weight,
            body_weight=config.training.body_loss_weight,
            action_weight=config.training.teacher_action_loss_weight,
            time_weight=config.training.temporal_loss_weight,
            regularizer_weight=config.training.latent_regularizer_weight,
        )
        forecast = _forecast_modules(
            config, protocols, state_dims, state_stats, qwen_dim=qwen_dim
        )
        forecast_objective = ForecastObjective(
            reference_encoder=reference["encoder"],
            normalizer=reference["normalizer"],
            latent_prediction_space=config.training.latent_prediction_space,
            history_state_adapters=forecast["history_state_adapters"],
            history_encoder=forecast["history_encoder"],
            latent_transition=forecast["latent_transition"],
            prior_weight=config.training.prior_prediction_loss_weight,
            world_weight=config.training.world_prediction_loss_weight,
            readout=reference["readout"],
            readout_weight=config.training.forecast_readout_loss_weight,
            control_teacher=reference["teacher"],
            control_weight=config.training.forecast_control_loss_weight,
            control_scope=config.training.forecast_control_scope,
            temporal_weight=config.training.temporal_loss_weight,
            freeze_reference=False,
        )
        objective = WorldModelObjective(
            representation,
            forecast_objective,
            reference["normalizer"],
            normalizer_calibration_updates=config.training.normalizer_calibration_updates,
            forecast_warmup_updates=config.training.forecast_warmup_updates,
            forecast_ramp_updates=config.training.forecast_ramp_updates,
        )
        checkpoint_modules = {**reference, **forecast}
    else:
        forecast = _forecast_modules(
            config, protocols, state_dims, state_stats, qwen_dim=qwen_dim
        )
        all_world_modules = {**reference, **forecast}
        if phase == "joint":
            expected["loss_contract"] = predictive_world_loss_contract(config.training)
        if config.training.joint_from_gr00t:
            from psi.models.predictive_world.gr00t_assets import initialization_contract
            initialization = initialization_contract(config.training, gr00t_initialization)
            student = clone_independent_expert(reference["teacher"])
            optimizer_update_offset = 0
        elif config.training.joint_from_scratch:
            from psi.models.predictive_world.scratch import random_joint_initialization

            initialization = random_joint_initialization(config.training)
            student = clone_independent_expert(reference["teacher"])
            optimizer_update_offset = 0
        elif config.training.init_joint:
            student = clone_independent_expert(reference["teacher"])
            initialization, parent_world_model = initialize_joint(
                config, {**all_world_modules, "student": student}, expected,
                allow_diagnostic_migration=allow_diagnostic_migration)
            optimizer_update_offset = initialization["optimizer_update_offset"]
        else:
            assert config.training.world_model is not None
            world_sidecar = load_predictive_world_bundle(
                config.training.world_model,
                phase="world_model",
                modules=all_world_modules,
                expected_metadata=expected,
            )
            world_metadata = world_sidecar["metadata"]
            if (world_metadata.get("loss_contract", {}).get("latent_prediction_space", "whitened")
                    != config.training.latent_prediction_space):
                raise ValueError("world-model latent prediction space mismatch")
            parent_world_model = {
                "phase": "world_model",
                "completed_updates": int(world_sidecar["completed_updates"]),
                "weights_sha256": str(world_sidecar["weights_sha256"]),
                "reference_id": str(world_metadata["reference_id"]),
                "normalization_id": str(world_metadata["normalization_id"]),
            }
            if phase == "joint":
                source_optimization = world_metadata.get("optimization_contract")
                if not isinstance(source_optimization, Mapping):
                    raise ValueError(
                        "joint continuation requires the world-model optimization contract"
                    )
                source_ema_decay = source_optimization.get("target_encoder_ema_decay")
                if source_ema_decay is None or float(source_ema_decay) != float(
                    config.training.target_encoder_ema_decay
                ):
                    raise ValueError(
                        "joint target_encoder_ema_decay must match the source world-model bundle"
                    )
            optimizer_update_offset = int(world_sidecar["completed_updates"])
            student = clone_independent_expert(reference["teacher"])
        if phase == "policy":
            for module in reference.values():
                freeze_module(module)
        policy_objective = PolicyObjective(
            student=student,
            normalizer=reference["normalizer"],
            latent_prediction_space=config.training.latent_prediction_space,
            history_state_adapters=forecast["history_state_adapters"],
            history_encoder=forecast["history_encoder"],
            latent_transition=forecast["latent_transition"],
            freeze_world=phase == "policy",
        )
        if phase == "joint":
            representation = RepresentationObjective(
                reference["encoder"], reference["readout"], reference["teacher"],
                scene_weight=config.training.scene_loss_weight,
                body_weight=config.training.body_loss_weight,
                action_weight=config.training.teacher_action_loss_weight,
                time_weight=config.training.temporal_loss_weight,
                regularizer_weight=config.training.latent_regularizer_weight,
            )
            forecast_objective = ForecastObjective(
                reference_encoder=reference["encoder"],
                normalizer=reference["normalizer"],
                latent_prediction_space=config.training.latent_prediction_space,
                history_state_adapters=forecast["history_state_adapters"],
                history_encoder=forecast["history_encoder"],
                latent_transition=forecast["latent_transition"],
                prior_weight=config.training.prior_prediction_loss_weight,
                world_weight=config.training.world_prediction_loss_weight,
                readout=reference["readout"],
                readout_weight=config.training.forecast_readout_loss_weight,
                control_teacher=reference["teacher"],
                control_weight=config.training.forecast_control_loss_weight,
                control_scope=config.training.forecast_control_scope,
                control_reader=config.training.forecast_control_reader,
                temporal_weight=config.training.temporal_loss_weight,
                freeze_reference=False,
            )
            world_objective = WorldModelObjective(
                representation,
                forecast_objective,
                reference["normalizer"],
                normalizer_calibration_updates=config.training.normalizer_calibration_updates,
                forecast_warmup_updates=config.training.forecast_warmup_updates,
                forecast_ramp_updates=config.training.forecast_ramp_updates,
                optimizer_update_offset=optimizer_update_offset,
                require_complete_initial_calibration=config.training.joint_from_scratch,
            )
            objective = JointObjective(
                world_objective,
                policy_objective,
                world_weight=config.training.joint_world_loss_weight,
                action_weight=config.training.joint_action_loss_weight,
            )
            checkpoint_modules = {**all_world_modules, "student": student}
        else:
            objective = policy_objective
            checkpoint_modules = {"student": student}
    wrapped = PhaseModelWithObservation(
        observation, objective, rtc_training_max_delay=config.training.rtc_training_max_delay,
    ).to(device)
    return BuiltPhase(
        wrapped,
        checkpoint_modules,
        reference,
        forecast,
        parent_world_model=parent_world_model,
        initialization=initialization,
    )


def build_deploy_policy(
    config: PredictiveWorldConfig,
    deploy_bundle: str | Path,
    *,
    device: torch.device,
):
    """Build only the causal serving graph and strictly load a deploy bundle."""

    if config.training.init_processor_diagnostic is not None:
        raise ValueError('processor diagnostic configs cannot be deployed')
    if config.training.init_processor_migration is not None:
        raise ValueError('research processor migration requires separate deployment acceptance')
    from transformers import Qwen3VLForConditionalGeneration

    from psi.models.predictive_world.policy import PredictiveWorldPolicy
    from psi.data.predictive_world.assets import resolve_asset_path
    from psi.data.predictive_world.provenance import verify_qwen_provenance

    qwen_path = resolve_asset_path(config.model.qwen_model, config.runtime.asset_path_map)
    feature_manifest = Path(config.data.feature_cache)
    if feature_manifest.is_dir():
        feature_manifest /= "feature_manifest.json"
    provenance = json.loads(feature_manifest.read_text(encoding="utf-8"))["provenance"]
    if config.model.host_architecture == "gr00t_n1d7":
        from psi.models.predictive_world.gr00t_assets import verify_backbone_export
        verify_backbone_export(
            qwen_path, resolve_asset_path(config.model.gr00t_model, config.runtime.asset_path_map),
            expected_config_sha256=config.model.gr00t_config_sha256)
    verify_qwen_provenance(qwen_path, provenance)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if config.model.host_architecture == "gr00t_n1d7":
        from psi.models.predictive_world.gr00t_assets import load_backbone
        qwen = load_backbone(qwen_path, dtype=dtype, device=device)
    else:
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            str(qwen_path), local_files_only=True, dtype=dtype, attn_implementation="sdpa"
        ).to(device)
    qwen_dim = int(getattr(qwen.config, "text_config", qwen.config).hidden_size)
    protocols = _registry(config.data.action_protocol_registry)
    state_dims, state_stats = _state_registry(config.data.state_schema_registry)
    forecast = _forecast_modules(
        config, protocols, state_dims, state_stats, qwen_dim=qwen_dim
    )
    normalizer = LatentNormalizer(config.model.latent_shape)
    student = _action_expert(config, protocols, state_dims, state_stats, qwen_dim=qwen_dim)
    modules = {
        **forecast,
        "latent_normalizer": normalizer,
        "student": student,
    }
    load_predictive_world_bundle(
        deploy_bundle,
        phase="deploy",
        modules=modules,
        expected_metadata={
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
            "encoder_id": str(provenance["encoder_id"]),
            "processor_id": str(provenance["processor_id"]),
            "protocol_registry_sha256": file_sha256(
                Path(config.data.action_protocol_registry)
            ),
            "measurement_registry_sha256": file_sha256(
                Path(config.data.measurement_registry)
            ),
            "qwen_reference": str(config.model.qwen_model),
            "loss_contract": predictive_world_loss_contract(config.training),
        },
    )
    policy = PredictiveWorldPolicy(
        observation_encoder=FrozenQwenObservationEncoder(qwen,
            **({"include_image_mask": True} if config.model.host_architecture == "gr00t_n1d7" else {})),
        history_state_adapters=forecast["history_state_adapters"],
        history_encoder=forecast["history_encoder"],
        latent_transition=forecast["latent_transition"], latent_normalizer=normalizer,
        student=student,
        action_solver_steps=config.model.action_solver_steps,
        latent_prediction_space=config.training.latent_prediction_space,
    ).to(device).eval()
    return policy


__all__ = ["BuiltPhase", "build_deploy_policy", "build_phase"]
