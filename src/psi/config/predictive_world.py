"""Validated configuration for the temporal latent predictive-world route.

This module is deliberately dependency-free: importing configuration must not
initialize a VLM, CUDA, distributed state, or touch a dataset.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping

Phase = Literal["world_model", "joint", "policy"]


@dataclass(frozen=True)
class PredictiveWorldDataConfig:
    history_visual_offsets_s: tuple[float, ...] = (-0.4, -0.2, 0.0)
    observed_visual_mode: Literal["history", "current_multiview"] = "history"
    current_visual_pooling: str = "canvas_v1"
    future_component_alignment: str | None = None
    real_observation_mode: str | None = None
    real_observation_contract: dict | None = None
    real_snapshot_max_age_s: float | None = None
    real_action_label_mode: str | None = None
    real_action_projection: dict | None = None
    future_measurement_max_age_s: float | None = None
    future_visual_max_age_s: float | None = None
    state_history_count: int = 21
    state_query_hz: float = 50.0
    future_group_ends_s: tuple[float, ...] = (0.2, 0.4, 0.6)
    action_horizon: int = 30
    action_hz: float = 50.0
    visual_max_age_s: float = 0.08
    state_max_age_s: float = 0.04
    action_max_age_s: float = 0.04
    source_sampling_exponent: float = 0.5
    measurement_registry: str = ""
    canonical_future_capabilities: str = ""
    action_protocol_registry: str = ""
    state_schema_registry: str = ""
    index_path: str = ""
    feature_cache: str = ""
    goal_supervision_path: str = ""

    @property
    def observation_contract(self) -> dict:
        # Preserve legacy bundle metadata; new input semantics must reject old bundles.
        contract = {} if self.observed_visual_mode == "history" else {
            "observed_visual_mode": self.observed_visual_mode
        }
        if self.current_visual_pooling != "canvas_v1":
            contract["current_visual_pooling"] = self.current_visual_pooling
        if self.real_observation_mode is not None:
            contract.update(real_observation_mode=self.real_observation_mode,
                            real_observation_contract=self.real_observation_contract,
                            real_action_label_mode=self.real_action_label_mode)
        if self.real_action_projection is not None:
            contract["real_action_projection"] = self.real_action_projection
        if self.real_snapshot_max_age_s is not None:
            contract["real_snapshot_max_age_s"] = self.real_snapshot_max_age_s
        return contract

    @property
    def observed_visual_offsets_s(self) -> tuple[float, ...]:
        return (0.0,) if self.observed_visual_mode == "current_multiview" else self.history_visual_offsets_s

    def validate(self) -> None:
        if self.current_visual_pooling not in {"canvas_v1", "content_area_v1"}:
            raise ValueError("unsupported current visual pooling")
        if self.goal_supervision_path:
            raise ValueError("goal-intent supervision is retired; use the original source snapshot")
        if self.real_observation_mode not in (None, "collector_snapshot_v1"):
            raise ValueError("unsupported real_observation_mode")
        if self.real_action_label_mode not in (None, "held_command_v1"):
            raise ValueError("unsupported real_action_label_mode")
        if self.real_observation_mode is None and (self.real_observation_contract is not None or self.real_action_label_mode is not None):
            raise ValueError("real observation/action contracts require collector_snapshot_v1")
        if self.real_snapshot_max_age_s is not None and self.real_observation_mode is None:
            raise ValueError("snapshot age limit requires collector snapshots")
        if self.real_action_projection is not None and self.real_observation_mode != "collector_snapshot_v1":
            raise ValueError("recorded action projection requires collector snapshots")
        if self.real_action_projection is not None:
            from psi.real_observation_contract import validate_action_projection
            validate_action_projection(self.real_action_projection)
        if self.real_observation_mode is not None:
            if (self.state_history_count != 1 or self.observed_visual_mode != "current_multiview"
                    or self.future_component_alignment != "independent_components_v1"
                    or self.real_action_label_mode != "held_command_v1"):
                raise ValueError("collector snapshots require one current state, current cameras and held independent actions")
            if not isinstance(self.real_observation_contract, dict):
                raise ValueError("collector snapshots require a bound real observation contract")
            from psi.real_observation_contract import (
                make_deployment_observation_contract,
            )
            make_deployment_observation_contract(dict(real_observation_mode=self.real_observation_mode,
                real_observation_contract=self.real_observation_contract, state_history_count=self.state_history_count,
                state_max_age_s=self.state_max_age_s, visual_max_age_s=self.visual_max_age_s,
                real_snapshot_max_age_s=self.real_snapshot_max_age_s))
            if self.future_measurement_max_age_s != .04 or self.future_visual_max_age_s != .08:
                raise ValueError("collector snapshots retain future body40ms/visual80ms supervision")
        for value in (self.future_measurement_max_age_s, self.future_visual_max_age_s):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("future supervision ages must be finite and nonnegative")
        if self.future_component_alignment not in (None, "independent_components_v1"):
            raise ValueError("unsupported future_component_alignment")
        if self.observed_visual_mode not in {"history", "current_multiview"}:
            raise ValueError("unsupported observed_visual_mode")
        if not self.history_visual_offsets_s or self.history_visual_offsets_s[-1] != 0.0:
            raise ValueError("visual history must end at the decision time (0 seconds)")
        if tuple(sorted(self.history_visual_offsets_s)) != self.history_visual_offsets_s:
            raise ValueError("visual history offsets must be sorted")
        if any(value > 0 for value in self.history_visual_offsets_s):
            raise ValueError("observed visual offsets must be causal")
        if self.state_history_count <= 0 or self.state_query_hz <= 0:
            raise ValueError("state history count and query frequency must be positive")
        if not self.future_group_ends_s or any(value <= 0 for value in self.future_group_ends_s):
            raise ValueError("future group ends must be positive")
        if tuple(sorted(set(self.future_group_ends_s))) != self.future_group_ends_s:
            raise ValueError("future group ends must be strictly increasing")
        if self.action_horizon <= 0 or self.action_hz <= 0:
            raise ValueError("action horizon and frequency must be positive")
        age_tolerances = (
            self.visual_max_age_s,
            self.state_max_age_s,
            self.action_max_age_s,
        )
        if any(not math.isfinite(value) or value < 0 for value in age_tolerances):
            raise ValueError("alignment age tolerances must be finite and non-negative")
        if (
            not math.isfinite(self.source_sampling_exponent)
            or not 0.0 <= self.source_sampling_exponent <= 1.0
        ):
            raise ValueError("source sampling exponent must be in [0,1]")
        horizon_s = self.action_horizon / self.action_hz
        if horizon_s + 1e-9 < self.future_group_ends_s[-1]:
            raise ValueError("action horizon does not cover all future groups")
        required_paths = (
            self.measurement_registry,
            self.canonical_future_capabilities,
            self.action_protocol_registry,
            self.state_schema_registry,
            self.index_path,
            self.feature_cache,
        )
        if any(not value or "REQUIRED_" in value for value in required_paths):
            raise ValueError("audited registries, index and feature cache paths are required")


@dataclass(frozen=True)
class LatentTransitionConfig:
    hidden_dim: int = 768
    num_blocks: int = 8
    num_heads: int = 12
    ff_dim: int = 3072
    semantics: str = "parallel_process"
    output_affine: bool = False

    def validate(self) -> None:
        if type(self.output_affine) is not bool:
            raise ValueError("output_affine must be boolean")
        if min(self.hidden_dim, self.num_blocks, self.num_heads, self.ff_dim) <= 0:
            raise ValueError("transition dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("transition hidden width must be divisible by its heads")
        if self.semantics != "parallel_process":
            raise ValueError("unsupported transition semantics")


@dataclass(frozen=True)
class PredictiveWorldModelConfig:
    host_architecture: str = "zeta_native"
    gr00t_model: str | None = None
    gr00t_config_sha256: str | None = None
    latent_shape: tuple[int, int, int] = (3, 8, 256)
    latent_scene_slots: int = 4
    latent_body_slots: int = 4
    history_context_tokens: int = 64
    history_hidden_dim: int = 1024
    history_blocks: int = 16
    history_heads: int = 16
    history_ff_dim: int = 4096
    expert_context_tokens: int = 16
    action_solver_steps: int = 10
    qwen_model: str = ""
    measurement_dim: int = 0
    max_scene_patches: int = 160
    body_points_per_group: int = 10
    representation_hidden_dim: int = 512
    representation_blocks: int = 4
    representation_heads: int = 8
    representation_ff_dim: int = 2048
    expert_hidden_dim: int = 768
    expert_blocks: int = 8
    expert_heads: int = 12
    expert_ff_dim: int = 3072
    latent_transition: LatentTransitionConfig = field(default_factory=LatentTransitionConfig)
    # Neutral legacy field retained to parse existing resolved-config identities.
    goal_intent_enabled: bool = False

    def validate(self, data: PredictiveWorldDataConfig) -> None:
        if self.host_architecture not in {"zeta_native", "gr00t_n1d7"}:
            raise ValueError("unsupported host architecture")
        if self.host_architecture == "gr00t_n1d7":
            if not self.gr00t_model or "REQUIRED_" in self.gr00t_model:
                raise ValueError("GR00T N1.7 requires a local base checkpoint")
            digest = self.gr00t_config_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("GR00T N1.7 requires the base configuration SHA256")
            if (data.state_history_count != 1 or data.observed_visual_mode != "current_multiview"
                    or data.action_horizon != 40 or self.action_solver_steps != 4):
                raise ValueError("GR00T N1.7 requires current observations, horizon40 and solver4")
        elif self.gr00t_model is not None or self.gr00t_config_sha256 is not None:
            raise ValueError("GR00T assets require host_architecture=gr00t_n1d7")
        if type(self.goal_intent_enabled) is not bool:
            raise ValueError("goal_intent_enabled must be boolean")
        if self.goal_intent_enabled:
            raise ValueError("goal-intent experiments are retired; use the original source snapshot")
        groups, slots, width = self.latent_shape
        if groups != len(data.future_group_ends_s):
            raise ValueError("latent groups must match future group ends")
        if slots != self.latent_scene_slots + self.latent_body_slots:
            raise ValueError("scene and body slots must exactly partition each latent group")
        dimensions = (
            groups, slots, width, self.history_context_tokens, self.expert_context_tokens,
            self.history_hidden_dim, self.history_blocks,
            self.history_heads, self.history_ff_dim,
            self.measurement_dim, self.max_scene_patches, self.body_points_per_group,
            self.representation_hidden_dim, self.representation_blocks,
            self.representation_heads, self.representation_ff_dim,
            self.expert_hidden_dim, self.expert_blocks, self.expert_heads, self.expert_ff_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.action_solver_steps <= 0:
            raise ValueError("action solver steps must be positive")
        if not self.qwen_model or "REQUIRED_" in self.qwen_model:
            raise ValueError("a local frozen Qwen model path is required")
        if self.representation_hidden_dim % self.representation_heads:
            raise ValueError("representation hidden width must be divisible by heads")
        if self.history_hidden_dim % self.history_heads:
            raise ValueError("history hidden width must be divisible by heads")
        if self.expert_hidden_dim % self.expert_heads:
            raise ValueError("expert hidden width must be divisible by heads")
        self.latent_transition.validate()


@dataclass(frozen=True)
class PredictiveWorldTrainingConfig:
    phase: Phase = "world_model"
    seed: int | None = 101
    max_updates: int = 0
    world_model: str | None = None
    init_joint: str | None = None
    init_source_config: str | None = None
    # Explicit trial only: default factory/serving reject this; train requires
    # bounded preflight and never writes deployable checkpoints for this mode.
    init_processor_diagnostic: dict | None = None
    # Separately approved research transition; never implicit diagnostic promotion.
    init_processor_migration: dict | None = None
    # A weight-only, source-bound objective transition. Never relaxes exact resume.
    init_objective_migration: dict | None = None
    # Explicit additive migration from a source without goal-intent modules.
    init_goal_intent: dict | None = None
    # Source-bound history -> current-state weight transfer; exact resumes stay strict.
    init_state_history_migration: dict | None = None
    init_observation_contract_migration: dict | None = None
    # Explicit body-coordinate semantics change; never inferred from tensor width.
    init_body_target_migration: dict | None = None
    init_vlm_action_migration: dict | None = None
    # One-time, train-only calibration and/or explicit current-image pooling transition.
    init_posttrain_migration: dict | None = None
    # Weight-only migration from legacy whitened F outputs to raw future latents.
    init_latent_prediction_migration: dict | None = None
    # Omission means the historical objective; new training explicitly selects raw.
    latent_prediction_space: str = "whitened"
    # Random task modules, retaining only the frozen VLM. Never implicit fallback.
    joint_from_scratch: bool = False
    # Pretrained native DiT/VL processing; new Zeta modules and task projectors.
    joint_from_gr00t: bool = False
    objective_contract_version: int = 1
    micro_batch: int = 256
    gradient_accumulation_steps: int = 1
    effective_global_batch: int = 10_240
    learning_rate: float = 1e-4
    joint_world_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    lr_schedule: str = "linear_floor"
    warmup_updates: int = 1000
    max_grad_norm: float = 1.0
    log_every: int = 10
    log_both_modes: bool = False
    validation_every: int = 0
    validation_batches: int = 4
    validation_micro_batch: int = 8
    validation_seed: int = 2026
    checkpoint_every: int = 3000
    checkpoint_retention: int | None = None
    first_checkpoint_update: int | None = None
    target_encoder_ema_decay: float = 0.99
    normalizer_calibration_updates: int = 500
    forecast_warmup_updates: int = 500
    forecast_ramp_updates: int = 500
    # Retain the key when loading and hashing historical resolved configs.
    latent_diagnostic_every: int = 100
    policy_latent_origin: str = "predicted"
    # None preserves historical full-horizon prefix sampling. Zero disables it.
    rtc_training_max_delay: int | None = None
    # None preserves logged source times. The explicit real-robot variant uses
    # execution-slot coordinates for model action conditions only.
    action_condition_time_mode: str | None = None
    scene_loss_weight: float = 1.0
    body_loss_weight: float = 1.0
    temporal_loss_weight: float = 0.25
    teacher_action_loss_weight: float = 1.0
    latent_regularizer_weight: float = 0.001
    prior_prediction_loss_weight: float = 1.0
    world_prediction_loss_weight: float = 1.0
    forecast_readout_loss_weight: float = 0.25
    forecast_control_loss_weight: float = 1.0
    forecast_control_scope: str = "prior"
    forecast_control_target: str = "teacher_response"
    forecast_control_reader: str = "teacher"
    preserve_control_sampling: bool = False
    body_auxiliary_enabled: bool = True
    future_body_input_enabled: bool = True
    student_latent_gradient: str = "joint"
    student_latent_mode: str = "predicted"
    joint_world_loss_weight: float = 1.0
    joint_action_loss_weight: float = 1.0
    goal_reconstruction_loss_weight: float = 1.0
    goal_prediction_loss_weight: float = 1.0

    def validate(self, *, world_size: int) -> None:
        if self.latent_prediction_space not in {"whitened", "raw"}:
            raise ValueError("latent_prediction_space must be whitened or raw")
        if self.latent_prediction_space == "raw" and self.normalizer_calibration_updates != 0:
            raise ValueError("raw latent prediction cannot fit latent normalization statistics")
        if (self.latent_prediction_space == "raw" and self.init_posttrain_migration is not None
                and (not isinstance(self.init_posttrain_migration, dict)
                     or self.init_posttrain_migration.get("calibration_batches") != 0)):
            raise ValueError("raw latent prediction cannot perform post-training calibration")
        if self.init_latent_prediction_migration is not None and (
            not isinstance(self.init_latent_prediction_migration, dict)
            or self.latent_prediction_space != "raw" or self.phase != "joint"
            or not self.init_joint or not self.init_source_config or self.joint_from_scratch
            or self.init_processor_diagnostic is not None
            or (self.init_posttrain_migration is not None
                and (not isinstance(self.init_posttrain_migration, dict)
                     or self.init_posttrain_migration.get("calibration_batches") != 0))
        ):
            raise ValueError("raw latent migration requires explicit joint initialization without calibration migration")
        if self.init_posttrain_migration is not None and (
            not isinstance(self.init_posttrain_migration, dict)
            or self.phase != "joint" or not self.init_joint or not self.init_source_config
            or self.joint_from_scratch or self.init_processor_diagnostic is not None
            or self.init_observation_contract_migration is not None
        ):
            raise ValueError("post-training migration requires explicit joint initialization")
        if type(self.joint_from_scratch) is not bool:
            raise ValueError("joint_from_scratch must be boolean")
        if type(self.joint_from_gr00t) is not bool or (self.joint_from_gr00t and self.joint_from_scratch):
            raise ValueError("GR00T initialization is explicit and distinct from random task initialization")
        if self.joint_from_scratch or self.joint_from_gr00t:
            if self.phase != "joint" or self.world_model or self.init_joint or self.init_source_config:
                raise ValueError("scratch joint training cannot load a world or joint checkpoint")
            if any(getattr(self, field) is not None for field in (
                "init_processor_diagnostic", "init_processor_migration", "init_objective_migration",
                "init_goal_intent", "init_state_history_migration", "init_observation_contract_migration",
                "init_body_target_migration", "init_vlm_action_migration",
                "init_latent_prediction_migration",
            )):
                raise ValueError("scratch joint training cannot inherit initialization migrations")
            if (self.objective_contract_version not in {5, 6}
                    or (self.latent_prediction_space == "whitened" and (
                        self.normalizer_calibration_updates <= 0
                        or self.forecast_warmup_updates < self.normalizer_calibration_updates
                        or self.forecast_ramp_updates <= 0))
                    or self.forecast_warmup_updates + self.forecast_ramp_updates >= self.max_updates):
                raise ValueError("scratch FULL+ERT requires a valid latent prediction startup within its budget")
            if type(self.seed) is not int or self.seed < 0:
                raise ValueError("scratch joint training requires an explicit nonnegative seed")
        if self.init_vlm_action_migration is not None and (
            not isinstance(self.init_vlm_action_migration, dict)
            or self.phase != "joint" or not self.init_joint or not self.init_source_config
            or self.init_processor_diagnostic is not None
            or self.init_observation_contract_migration is not None
        ):
            raise ValueError("VLM/action migration requires explicit joint initialization without diagnostic/snapshot migration")
        if self.init_vlm_action_migration is not None and self.init_processor_migration is not None:
            transition = self.init_vlm_action_migration
            if (transition.get("format") != "zeta0.predictive_world.action_grid_initialization.v1"
                    or not transition.get("source_encoder_id")
                    or transition.get("source_encoder_id") != transition.get("target_encoder_id")
                    or transition.get("processor_migration") != self.init_processor_migration):
                raise ValueError("action-grid migration requires the same encoder and bound processor evidence")
        if self.init_body_target_migration is not None and (
            not isinstance(self.init_body_target_migration, dict)
            or self.phase != "joint" or not self.init_joint or not self.init_source_config
            or self.body_auxiliary_enabled is not True
        ):
            raise ValueError("body-target migration requires joint initialization and body supervision")
        if self.objective_contract_version == 3 or self.init_goal_intent is not None:
            raise ValueError("goal-intent experiments are retired; use the original source snapshot")
        if self.init_observation_contract_migration is not None and (
            not isinstance(self.init_observation_contract_migration, dict)
            or self.phase != "joint" or not self.init_joint or not self.init_source_config
        ):
            raise ValueError("observation migration requires explicit joint initialization")
        if self.init_state_history_migration is not None and (
            not isinstance(self.init_state_history_migration, dict)
            or self.phase != "joint" or not self.init_joint or not self.init_source_config
            or self.init_goal_intent is not None
        ):
            raise ValueError("current-state migration requires explicit joint initialization without goal migration")
        if type(self.objective_contract_version) is not int or self.objective_contract_version not in {1, 2, 3, 4, 5, 6}:
            raise ValueError("objective_contract_version must be 1, 2, 3, 4, 5 or 6")
        if self.student_latent_gradient != "joint":
            raise ValueError("student_latent_gradient must be joint")
        if self.student_latent_mode != "predicted":
            raise ValueError("student_latent_mode must be predicted")
        if self.forecast_control_target != "teacher_response":
            raise ValueError("forecast_control_target must be teacher_response")
        if self.body_auxiliary_enabled is not True or self.future_body_input_enabled is not True:
            raise ValueError("future body inputs and body supervision are required")
        if type(self.preserve_control_sampling) is not bool:
            raise ValueError("preserve_control_sampling must be boolean")
        if self.preserve_control_sampling != (self.objective_contract_version == 6):
            raise ValueError("preserve_control_sampling must match the version-6 checkpoint contract")
        if self.forecast_control_reader not in {"teacher", "student", "both"}:
            raise ValueError("forecast_control_reader must be teacher, student or both")
        if self.forecast_control_reader == "student" and self.objective_contract_version != 4:
            raise ValueError("student forecast_control_reader requires objective_contract_version=4")
        if self.objective_contract_version == 4 and (
            self.phase != "joint" or self.forecast_control_reader != "student"
            or self.forecast_control_loss_weight <= 0 or self.forecast_control_scope != "prior"
        ):
            raise ValueError("objective_contract_version=4 requires joint prior student response supervision")
        if self.forecast_control_reader == "both" and self.objective_contract_version not in {5, 6}:
            raise ValueError("both forecast_control_reader requires objective_contract_version=5 or 6")
        if self.objective_contract_version == 5 and (
            self.phase != "joint" or self.forecast_control_reader != "both"
            or self.forecast_control_loss_weight != 1.0 or self.forecast_control_scope != "prior"
            or self.body_loss_weight <= 0 or self.temporal_loss_weight <= 0
        ):
            raise ValueError("objective_contract_version=5 requires joint FULL+ERT with fixed half teacher/student responses and full body/temporal supervision")
        if self.objective_contract_version == 6 and (
            self.phase != "joint" or self.latent_prediction_space != "raw"
            or self.normalizer_calibration_updates != 0
            or self.forecast_control_scope != "prior" or self.forecast_control_reader != "both"
            or self.forecast_control_loss_weight != 1.0 or self.body_loss_weight <= 0
            or self.temporal_loss_weight <= 0 or self.world_prediction_loss_weight <= 0
        ):
            raise ValueError("objective_contract_version=6 requires raw joint FULL+ERT training")
        if self.init_objective_migration is not None and (
            not isinstance(self.init_objective_migration, dict)
            or self.objective_contract_version not in {5, 6} or self.phase != "joint"
            or not self.init_joint or not self.init_source_config
        ):
            raise ValueError("objective migration requires explicit version-5 or version-6 joint initialization")
        if self.init_goal_intent is not None and (
            not isinstance(self.init_goal_intent, dict)
            or self.objective_contract_version != 3 or self.phase != "joint"
            or not self.init_joint or not self.init_source_config
            or self.init_objective_migration is not None
        ):
            raise ValueError("goal-intent migration requires explicit version-3 joint initialization")
        if self.init_processor_migration is not None and (
            not isinstance(self.init_processor_migration, dict)
            or self.init_processor_diagnostic is not None
            or self.phase != 'joint' or not self.init_joint or not self.init_source_config
        ):
            raise ValueError('processor migration requires joint initialization, without diagnostic mode')
        if self.init_processor_diagnostic is not None and (
            not isinstance(self.init_processor_diagnostic, dict)
            or self.phase != 'joint' or not self.init_joint or not self.init_source_config
        ):
            raise ValueError('processor diagnostic requires explicit joint initialization')
        if self.rtc_training_max_delay is not None and (
            type(self.rtc_training_max_delay) is not int or self.rtc_training_max_delay < 0
        ):
            raise ValueError("rtc_training_max_delay must be a non-negative integer or null")
        if self.action_condition_time_mode not in (None, "fixed_action_grid_v1"):
            raise ValueError("unsupported action_condition_time_mode")
        if self.phase not in ("world_model", "joint", "policy"):
            raise ValueError(f"unsupported training phase: {self.phase!r}")
        if self.max_updates <= 0:
            raise ValueError("max_updates is an explicit required budget")
        if min(world_size, self.micro_batch, self.gradient_accumulation_steps) <= 0:
            raise ValueError("world size, micro batch and accumulation must be positive")
        expected = world_size * self.micro_batch * self.gradient_accumulation_steps
        if self.effective_global_batch != expected:
            raise ValueError(
                f"effective_global_batch={self.effective_global_batch} does not equal "
                f"world_size*micro_batch*accumulation={expected}"
            )
        if self.init_joint or self.init_source_config:
            if self.phase != "joint" or not (self.init_joint and self.init_source_config) or self.world_model:
                raise ValueError("joint initialization requires init_joint and init_source_config, without world_model")
        if self.lr_schedule not in {"linear_floor", "cosine"}:
            raise ValueError("unsupported learning-rate schedule")
        if len(self.adam_betas) != 2 or any(not 0 <= b < 1 for b in self.adam_betas):
            raise ValueError("Adam betas must be in [0,1)")
        if self.phase in {"joint", "policy"} and not (
            self.world_model or self.init_joint or self.joint_from_scratch or self.joint_from_gr00t
        ):
            raise ValueError(f"phase={self.phase} requires a world-model bundle")
        if (
            self.learning_rate <= 0
            or self.joint_world_learning_rate <= 0
            or self.weight_decay < 0
            or self.max_grad_norm <= 0
        ):
            raise ValueError("optimizer values are invalid")
        if self.checkpoint_retention is not None and self.checkpoint_retention < 1:
            raise ValueError("checkpoint_retention must be positive")
        if self.validation_every < 0 or min(self.validation_batches, self.validation_micro_batch) <= 0:
            raise ValueError("validation cadence/batch values are invalid")
        if self.validation_every and self.phase != "joint":
            raise ValueError("periodic validation currently requires phase=joint")
        if self.first_checkpoint_update is not None and self.first_checkpoint_update <= 0:
            raise ValueError("first_checkpoint_update must be positive")
        if min(
            self.warmup_updates,
            self.log_every,
            self.checkpoint_every,
            self.normalizer_calibration_updates,
            self.forecast_warmup_updates,
            self.forecast_ramp_updates,
            self.latent_diagnostic_every,
        ) < 0:
            raise ValueError("training cadence values cannot be negative")
        if self.log_every == 0 or self.checkpoint_every == 0 or self.latent_diagnostic_every == 0:
            raise ValueError("log/checkpoint cadence must be positive")
        if not 0 <= self.target_encoder_ema_decay < 1:
            raise ValueError("target encoder EMA decay must be in [0,1)")
        if self.normalizer_calibration_updates > self.forecast_warmup_updates:
            raise ValueError("normalizer calibration must finish before forecast warmup")
        if self.policy_latent_origin != "predicted":
            raise ValueError("policy uses only deployable predicted latents; old latent mixtures are unsupported")
        if self.forecast_control_scope not in {"prior", "both"}:
            raise ValueError("forecast_control_scope must be prior or both")
        weights = (
            self.scene_loss_weight, self.body_loss_weight, self.temporal_loss_weight,
            self.teacher_action_loss_weight, self.latent_regularizer_weight,
            self.prior_prediction_loss_weight, self.world_prediction_loss_weight,
            self.forecast_readout_loss_weight, self.forecast_control_loss_weight,
            self.joint_world_loss_weight, self.joint_action_loss_weight,
            self.goal_reconstruction_loss_weight, self.goal_prediction_loss_weight,
        )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in weights
        ):
            raise ValueError("loss weights must be finite and non-negative")
        if self.phase in {"world_model", "joint"} and not any(
            value > 0 for value in (
                self.scene_loss_weight, self.body_loss_weight,
                self.temporal_loss_weight, self.teacher_action_loss_weight,
            )
        ):
            raise ValueError("world-model training requires at least one supervised loss")
        if self.phase in {"world_model", "joint"} and (
            self.prior_prediction_loss_weight <= 0
            or (self.world_prediction_loss_weight <= 0 and self.objective_contract_version != 6)
        ):
            raise ValueError("world-model training requires positive prior and world prediction losses")
        if self.phase == "joint":
            if not self.joint_from_scratch and self.normalizer_calibration_updates != 0:
                raise ValueError("joint continuation must reuse the fitted latent normalizer")
            if (self.latent_prediction_space == "whitened" and not self.joint_from_scratch
                    and (self.forecast_warmup_updates != 0 or self.forecast_ramp_updates != 0)):
                raise ValueError("joint continuation keeps forecast losses active from update zero")
            if self.joint_world_loss_weight <= 0 or self.joint_action_loss_weight <= 0:
                raise ValueError("joint continuation requires positive world and action losses")


@dataclass(frozen=True)
class PredictiveWorldRuntimeConfig:
    nodes: int = 5
    gpus_per_node: int = 8
    output_dir: str = ""
    allow_existing_output: bool = False
    process_group_timeout_s: int = 600
    dataloader_timeout_s: int = 600
    asset_path_map: dict[str, str] = field(default_factory=dict)
    gr00t_activation_checkpointing: bool = False
    ddp_gradient_as_bucket_view: bool = False

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    def validate(self) -> None:
        if self.nodes <= 0 or self.gpus_per_node <= 0:
            raise ValueError("nodes and GPUs per node must be positive")
        if self.process_group_timeout_s <= 0 or self.dataloader_timeout_s <= 0:
            raise ValueError("process-group and DataLoader timeouts must be positive")
        if self.process_group_timeout_s < self.dataloader_timeout_s:
            raise ValueError("process-group timeout must be >= DataLoader timeout")
        if not self.output_dir:
            raise ValueError("a new output directory is required")
        if "REQUIRED_" in self.output_dir:
            raise ValueError("placeholder output paths are not runnable")
        output = Path(self.output_dir)
        if output.exists() and not output.is_dir():
            raise ValueError("predictive-world output path exists and is not a directory")
        if output.exists() and any(output.iterdir()) and not self.allow_existing_output:
            raise ValueError("predictive-world output directory must be new or empty")


@dataclass(frozen=True)
class PredictiveWorldConfig:
    route: str = "predictive_world"
    data: PredictiveWorldDataConfig = field(default_factory=PredictiveWorldDataConfig)
    model: PredictiveWorldModelConfig = field(default_factory=PredictiveWorldModelConfig)
    training: PredictiveWorldTrainingConfig = field(default_factory=PredictiveWorldTrainingConfig)
    runtime: PredictiveWorldRuntimeConfig = field(default_factory=PredictiveWorldRuntimeConfig)

    def validate(self, *, allow_existing_output_override: bool = False) -> None:
        from psi.data.predictive_world.assets import validate_asset_path_map
        validate_asset_path_map(self.runtime.asset_path_map)
        if self.route != "predictive_world":
            raise ValueError("PredictiveWorldConfig only supports route=predictive_world")
        self.data.validate()
        self.model.validate(self.data)
        if allow_existing_output_override:
            if not self.runtime.output_dir or "REQUIRED_" in self.runtime.output_dir:
                raise ValueError("a non-placeholder resume output directory is required")
        else:
            self.runtime.validate()
        self.training.validate(world_size=self.runtime.world_size)
        if type(self.runtime.gr00t_activation_checkpointing) is not bool:
            raise ValueError("gr00t_activation_checkpointing must be boolean")
        if type(self.runtime.ddp_gradient_as_bucket_view) is not bool:
            raise ValueError("ddp_gradient_as_bucket_view must be boolean")
        if self.runtime.gr00t_activation_checkpointing and self.model.host_architecture != "gr00t_n1d7":
            raise ValueError("GR00T activation checkpointing requires the N1.7 host architecture")
        if self.model.host_architecture == "gr00t_n1d7":
            if (not self.training.joint_from_gr00t or self.training.latent_prediction_space != "raw"
                    or self.training.normalizer_calibration_updates != 0):
                raise ValueError("N1.7 v1 requires joint_from_gr00t with raw latents and no calibration")
        elif self.training.joint_from_gr00t:
            raise ValueError("joint_from_gr00t requires the N1.7 host architecture")
        adaptation = self.training.init_posttrain_migration
        raw_migration = self.training.init_latent_prediction_migration
        if raw_migration is not None:
            if not self.model.latent_transition.output_affine:
                raise ValueError("raw latent migration requires an explicit output coordinate transform")
        if adaptation is not None:
            batches = adaptation.get("calibration_batches")
            if type(batches) is not int or batches < 0:
                raise ValueError("post-training calibration batches must be a nonnegative integer")
            if ((raw_migration is None and self.model.latent_transition.output_affine != (batches > 0))
                    or self.data.current_visual_pooling != adaptation.get("current_visual_pooling")):
                raise ValueError("post-training coordinate/pooling configuration disagrees with migration")
        elif (raw_migration is None and self.model.latent_transition.output_affine
              and not (self.training.latent_prediction_space == "raw"
                       and (self.training.init_joint or self.training.world_model))):
            raise ValueError("output affine requires explicit output coordinate migration")
        if self.data.real_observation_mode is not None and self.training.action_condition_time_mode != "fixed_action_grid_v1":
            raise ValueError("held-command supervision requires fixed action-slot model conditions")
        if self.training.init_observation_contract_migration is not None and self.data.real_observation_mode is None:
            raise ValueError("observation migration requires the explicit collector snapshot contract")
        if self.training.init_state_history_migration is not None and self.data.state_history_count != 1:
            raise ValueError("current-state migration requires exactly one observed state")
        if (self.training.rtc_training_max_delay is not None
                and self.training.rtc_training_max_delay >= self.data.action_horizon):
            raise ValueError("rtc_training_max_delay must be smaller than action_horizon")


def predictive_world_architecture_contract(model: PredictiveWorldModelConfig) -> dict:
    """Keep old checkpoint architecture bytes when the optional method is off."""
    contract = json.loads(json.dumps(asdict(model)))
    if model.host_architecture == "zeta_native":
        for name in ("host_architecture", "gr00t_model", "gr00t_config_sha256"):
            contract.pop(name)
    else:
        contract["gr00t_adapter"] = "zeta0.gr00t_n1d7.native_head.v1"
    if not model.goal_intent_enabled:
        contract.pop("goal_intent_enabled")
    if not model.latent_transition.output_affine:
        contract["latent_transition"].pop("output_affine")
    return contract


DUAL_RESPONSE_TEACHER_WEIGHT = .5
DUAL_RESPONSE_STUDENT_WEIGHT = .5


def predictive_world_loss_contract(training: PredictiveWorldTrainingConfig) -> dict:
    """Persist objective semantics that cannot be inferred from tensor shapes."""
    contract = {
        "measurement_slots": "fixed_nominal_queries",
        "measurement_delta_pairs": "matching_query_offset_with_valid_endpoints",
        "forecast_readout_parameters": "detached",
        "forecast_control_scope": training.forecast_control_scope,
        "forecast_readout_loss_weight": training.forecast_readout_loss_weight,
        "forecast_control_loss_weight": training.forecast_control_loss_weight,
    }
    if training.objective_contract_version >= 2:
        contract.update(effective_objective_fields(training))
        contract["version"] = training.objective_contract_version
    # Read-only identity support for old source configs; v3 execution is retired.
    if training.objective_contract_version == 3:
        contract.update({
            "goal_intent": "dynamic_endpoint_visual_change_v1",
            "goal_reconstruction_loss_weight": training.goal_reconstruction_loss_weight,
            "goal_prediction_loss_weight": training.goal_prediction_loss_weight,
            "goal_teacher_condition": "online_endpoint_change",
            "goal_student_condition": "causal_predicted_change",
        })
    if training.objective_contract_version in {5, 6} and training.forecast_control_reader == "both":
        contract.update({
            "forecast_control_teacher_response_weight": DUAL_RESPONSE_TEACHER_WEIGHT,
            "forecast_control_student_response_weight": DUAL_RESPONSE_STUDENT_WEIGHT,
        })
    if training.latent_prediction_space != "whitened":
        contract["latent_prediction_space"] = training.latent_prediction_space
    return contract


def effective_objective_fields(training: PredictiveWorldTrainingConfig) -> dict:
    """Full objective identity, also recoverable from legacy resolved configs."""
    fields = {name: getattr(training, name) for name in (
        "scene_loss_weight", "body_loss_weight", "temporal_loss_weight",
        "teacher_action_loss_weight", "latent_regularizer_weight",
        "prior_prediction_loss_weight", "world_prediction_loss_weight",
        "forecast_readout_loss_weight", "forecast_control_loss_weight", "forecast_control_scope",
        "joint_world_loss_weight", "joint_action_loss_weight", "forecast_control_target",
        "preserve_control_sampling", "body_auxiliary_enabled",
        "student_latent_gradient", "student_latent_mode",
    )}
    # Preserve field presence in historical checkpoint contracts.
    if training.objective_contract_version >= 4:
        fields["forecast_control_reader"] = training.forecast_control_reader
    if training.objective_contract_version >= 6:
        fields["future_body_input_enabled"] = training.future_body_input_enabled
    return fields


def load_predictive_world_config(
    path: str | Path,
    *,
    allow_existing_output_override: bool = False,
    training_overrides: Mapping[str, int | None] | None = None,
) -> tuple[PredictiveWorldConfig, dict]:
    """Load JSON into the typed config without importing model/runtime dependencies."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return predictive_world_config_from_payload(
        payload, allow_existing_output_override=allow_existing_output_override,
        training_overrides=training_overrides,
    )


def predictive_world_config_from_payload(
    payload: dict, *, allow_existing_output_override: bool = False,
    training_overrides: Mapping[str, int | None] | None = None,
) -> tuple[PredictiveWorldConfig, dict]:
    """Validate an in-memory resolved config through the same path as JSON input."""
    if not isinstance(payload, dict):
        raise ValueError("predictive-world config must be a JSON object")
    overrides = {
        key: int(value) for key, value in (training_overrides or {}).items()
        if value is not None
    }
    allowed = {
        "micro_batch", "gradient_accumulation_steps", "effective_global_batch"
    }
    if overrides.keys() - allowed:
        raise ValueError(f"unsupported training overrides: {sorted(overrides.keys() - allowed)}")
    if overrides:
        training = dict(payload.get("training", {}))
        training.update(overrides)
        if (
            {"micro_batch", "gradient_accumulation_steps"} & overrides.keys()
            and "effective_global_batch" not in overrides
        ):
            runtime = PredictiveWorldRuntimeConfig(**payload.get("runtime", {}))
            training["effective_global_batch"] = (
                runtime.world_size
                * int(training.get("micro_batch", PredictiveWorldTrainingConfig.micro_batch))
                * int(training.get(
                    "gradient_accumulation_steps",
                    PredictiveWorldTrainingConfig.gradient_accumulation_steps,
                ))
            )
        payload = {**payload, "training": training}
    data = dict(payload.get("data", {}))
    for name in ("history_visual_offsets_s", "future_group_ends_s"):
        if name in data:
            data[name] = tuple(data[name])
    model = dict(payload.get("model", {}))
    if "latent_flow" in model or "latent_transition" not in model:
        raise ValueError("legacy latent-flow config is incompatible; resolve a new direct-transition run")

    if "latent_shape" in model:
        model["latent_shape"] = tuple(model["latent_shape"])
    latent_transition = LatentTransitionConfig(**model.pop("latent_transition", {}))
    config = PredictiveWorldConfig(
        route=str(payload.get("route", "")),
        data=PredictiveWorldDataConfig(**data),
        model=PredictiveWorldModelConfig(**model, latent_transition=latent_transition),
        training=PredictiveWorldTrainingConfig(**payload.get("training", {})),
        runtime=PredictiveWorldRuntimeConfig(**payload.get("runtime", {})),
    )
    config.validate(allow_existing_output_override=allow_existing_output_override)
    return config, payload


__all__ = [
    "LatentTransitionConfig",
    "Phase",
    "PredictiveWorldConfig",
    "PredictiveWorldDataConfig",
    "PredictiveWorldModelConfig",
    "PredictiveWorldRuntimeConfig",
    "PredictiveWorldTrainingConfig",
    "load_predictive_world_config",
    "predictive_world_config_from_payload",
    "predictive_world_architecture_contract",
    "predictive_world_loss_contract",
    "effective_objective_fields",
]
