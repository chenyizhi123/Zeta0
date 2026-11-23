"""Deploy-only predictive-world policy composition with no future-label input."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.data.predictive_world.contracts import ObservedHistory, assert_observed_only
from psi.models.predictive_world.action_expert import TemporalNativeActionExpert
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.latent_transition import (
    LatentNormalizer, TemporalLatentTransition, prediction_to_raw,
)


@dataclass(frozen=True)
class ObservationFeatures:
    qwen_context: Tensor
    context_valid: Tensor
    current_visual: Tensor
    # Native GR00T alternates cross-attention to image and non-image tokens.
    image_mask: Tensor | None = None
    # Reserved for layered KV-cache hosts such as pi0.5 (PaliGemma + Gemma expert).
    kv_cache: Any | None = None
    kv_attention_mask: Tensor | None = None

    @property
    def context(self) -> Tensor:
        """Backbone-agnostic alias for System 2 multimodal context tokens."""
        return self.qwen_context


class ObservationEncoder(Protocol):
    def __call__(self, observed: ObservedHistory) -> ObservationFeatures: ...


class PredictiveWorldPolicy(nn.Module):
    """The exportable Qwen/history/prior/student chain.

    Future encoder, readout, teacher, and training cache readers are absent by
    construction. Random tensors are explicit inputs so replay can reproduce
    action sampling exactly. Future process prediction is deterministic.
    """

    def __init__(
        self,
        *,
        observation_encoder: nn.Module,
        history_state_adapters: StateAdapterBank,
        history_encoder: PredictiveHistoryEncoder,
        latent_transition: TemporalLatentTransition,
        latent_normalizer: LatentNormalizer,
        student: TemporalNativeActionExpert,
        action_solver_steps: int = 10,
        latent_prediction_space: str = "whitened",
    ) -> None:
        super().__init__()
        if action_solver_steps <= 0:
            raise ValueError("deploy solver steps must be positive")
        if latent_prediction_space not in {"whitened", "raw"}:
            raise ValueError("latent_prediction_space must be whitened or raw")
        self.observation_encoder = observation_encoder
        self.history_state_adapters = history_state_adapters
        self.history_encoder = history_encoder
        self.latent_transition = latent_transition
        self.latent_normalizer = latent_normalizer
        self.student = student
        self.action_solver_steps = int(action_solver_steps)
        self.latent_prediction_space = latent_prediction_space

    @torch.inference_mode()
    def sample(
        self,
        observed: ObservedHistory,
        *,
        action_noise: Tensor,
        stage_timings: dict[str, float] | None = None,
    ) -> Tensor:
        observed = assert_observed_only(observed)
        device_type = observed.raw_states.device.type
        profile_enabled = stage_timings is not None

        def mark(name: str, started: float) -> float:
            if not profile_enabled:
                return started
            if device_type == "cuda":
                # CUDA work is asynchronous.  This path is diagnostics-only,
                # so synchronize at stage boundaries for meaningful timings.
                torch.cuda.synchronize(observed.raw_states.device)
            now = time.perf_counter()
            assert stage_timings is not None
            stage_timings[name] = now - started
            return now

        stage_started = time.perf_counter() if profile_enabled else 0.0
        total_started = stage_started
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device_type == "cuda" else contextlib.nullcontext()
        )
        with autocast:
            batch = observed.raw_states.shape[0]
            observed.validate(
                batch_size=batch,
                state_count=self.history_encoder.state_history_count,
                action_horizon=self.student.action_steps,
            )
            stage_started = mark("validate_s", stage_started)
            features = self.observation_encoder(observed)
            stage_started = mark("observation_encoder_s", stage_started)
            time_valid = observed.state_valid.any(dim=-1)
            adapted_states = self.history_state_adapters(
                observed.raw_states,
                observed.state_valid,
                list(observed.state_schema_ids),
                state_present=time_valid.any(dim=1),
            )
            adapted_states = adapted_states * time_valid[:, :, None].to(adapted_states.dtype)
            condition = self.history_encoder(
                qwen_context=features.qwen_context,
                context_valid=features.context_valid,
                current_visual=features.current_visual,
                adapted_states=adapted_states,
                state_valid=time_valid,
                state_times=observed.state_times,
            )
            stage_started = mark("history_encoder_s", stage_started)
            committed_condition = observed.committed_actions
            if committed_condition is not None and not bool(committed_condition.valid.any()):
                committed_condition = None
            transition_output = self.latent_transition(
                condition,
                context_valid=torch.ones(
                    condition.shape[:2], device=condition.device, dtype=torch.bool
                ),
                mode="prior",
                action_condition=committed_condition,
            )
            latent_raw = prediction_to_raw(
                transition_output, self.latent_normalizer, space=self.latent_prediction_space,
            )
            stage_started = mark("latent_transition_s", stage_started)
            latent_valid = torch.ones(
                batch, latent_raw.shape[1], device=latent_raw.device, dtype=torch.bool
            )
            host_extras = {}
            if features.image_mask is not None:
                host_extras["image_mask"] = features.image_mask
            if features.kv_cache is not None:
                host_extras["kv_cache"] = features.kv_cache
            if features.kv_attention_mask is not None:
                host_extras["kv_attention_mask"] = features.kv_attention_mask
            actions = self.student.sample(
                **host_extras,
                qwen_context=features.qwen_context,
                context_valid=features.context_valid,
                raw_states=observed.raw_states,
                state_dimension_valid=observed.state_valid,
                state_schema_ids=observed.state_schema_ids,
                state_time_valid=time_valid,
                state_times=observed.state_times,
                z_raw=latent_raw,
                latent_group_valid=latent_valid,
                protocol_ids=observed.action_protocol_ids,
                native_dims=observed.action_native_dims,
                noise=action_noise,
                committed_prefix=observed.committed_actions,
                steps=self.action_solver_steps,
            )
            mark("action_expert_s", stage_started)
            if profile_enabled:
                assert stage_timings is not None
                stage_timings["total_s"] = time.perf_counter() - total_started
            return actions


__all__ = ["ObservationEncoder", "ObservationFeatures", "PredictiveWorldPolicy"]
