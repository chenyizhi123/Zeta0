"""Concrete representation, forecast and policy phase objectives."""

from __future__ import annotations

import contextlib
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn

from psi.config.predictive_world import (
    DUAL_RESPONSE_STUDENT_WEIGHT,
    DUAL_RESPONSE_TEACHER_WEIGHT,
)
from psi.data.predictive_world.contracts import NativeActionBatch, ObservedHistory
from psi.data.predictive_world.objective import masked_per_sample_mse
from psi.models.predictive_world.action_expert import (
    ActionFlowTrainingSample,
    TemporalNativeActionExpert,
    committed_action_prefix_lengths,
    make_committed_action_prefix,
    prepare_action_flow_training,
    sample_action_prefix_lengths,
)
from psi.models.predictive_world.engine import PhaseLoss
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.latent_transition import (
    LatentNormalizer,
    TemporalLatentTransition,
    prediction_to_raw,
)
from psi.models.predictive_world.policy import (
    ObservationFeatures,
)
from psi.models.predictive_world.representation import (
    EMAFutureTrajectoryEncoder,
    FutureTrajectoryEncoder,
    WorldReadout,
    representation_regularizer_per_sample,
)
from psi.models.predictive_world.training import (
    freeze_module,
)


def _combine(parts: list[tuple[Tensor, Tensor, float]]) -> tuple[Tensor, Tensor]:
    if not parts:
        raise ValueError("phase objective has no enabled loss terms")
    batch = parts[0][0].shape[0]
    numerator = parts[0][0].new_zeros(batch)
    enabled = parts[0][0].new_zeros(batch, dtype=torch.bool)
    for values, valid, weight in parts:
        if values.shape != (batch,) or valid.shape != (batch,) or weight < 0:
            raise ValueError("phase loss parts must be aligned non-negative [B] terms")
        numerator = numerator + torch.where(valid, values * weight, torch.zeros_like(values))
        enabled = enabled | (valid & (weight > 0))
    return numerator, enabled


def _encoder_arguments(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    names = (
        "anchor_patches", "anchor_patch_valid", "current_measurements",
        "current_measurement_valid", "future_patches", "future_patch_valid",
        "future_scene_times", "future_measurements", "future_measurement_valid",
        "future_measurement_times", "group_valid",
    )
    missing = [name for name in names if name not in batch]
    if missing:
        raise ValueError(f"prepared phase batch is missing encoder inputs: {missing}")
    result = {name: batch[name] for name in names}
    for name in ("future_patch_positions", "future_patch_camera_indices"):
        if name in batch:
            result[name] = batch[name]
    return result


def _history_condition(
    *,
    observed: ObservedHistory,
    features: ObservationFeatures,
    state_adapters: nn.Module,
    history_encoder: PredictiveHistoryEncoder,
) -> Tensor:
    time_valid = observed.state_valid.any(dim=-1)
    adapted = state_adapters(
        observed.raw_states,
        observed.state_valid,
        list(observed.state_schema_ids),
        state_present=time_valid.any(dim=1),
    )
    return history_encoder(
        qwen_context=features.qwen_context,
        context_valid=features.context_valid,
        current_visual=features.current_visual,
        adapted_states=adapted,
        state_valid=time_valid,
        state_times=observed.state_times,
    )


def _expert_arguments(
    observed: ObservedHistory,
    features: ObservationFeatures,
    z_raw: Tensor,
    group_valid: Tensor,
) -> dict[str, Any]:
    result = {
        "protocol_ids": observed.action_protocol_ids,
        "qwen_context": features.qwen_context,
        "context_valid": features.context_valid,
        "raw_states": observed.raw_states,
        "state_dimension_valid": observed.state_valid,
        "state_schema_ids": observed.state_schema_ids,
        "state_time_valid": observed.state_valid.any(dim=-1),
        "state_times": observed.state_times,
        "z_raw": z_raw,
        "latent_group_valid": group_valid,
    }
    if features.image_mask is not None:
        result["image_mask"] = features.image_mask
    return result


def _response_velocity(reader: nn.Module, state: dict, arguments: dict) -> Tensor:
    """Deterministic frozen reader, retaining derivatives with respect to z.

    Detaching parameters alone does not disable N1.7's attention/state dropout.
    Restore each module's original mode, including independently frozen children.
    """
    def evaluate(state, arguments):
        modes = [(module, module.training) for module in reader.modules()]
        try:
            reader.eval()
            return torch.func.functional_call(reader, state, (), arguments, strict=True)
        finally:
            for module, mode in modes:
                module.training = mode

    if getattr(reader, "activation_checkpointing", False) and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(evaluate, state, arguments, use_reentrant=False)
    return evaluate(state, arguments)


def _optimizer_update(batch: Mapping[str, Any], *, fallback: int = 0) -> int:
    value = batch.get("_optimizer_update", fallback)
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("optimizer update marker must be scalar")
        value = value.item()
    update = int(value)
    if update < 0:
        raise ValueError("optimizer update marker cannot be negative")
    return update


def _sample_training_prefix(batch: Mapping[str, Any], actions: NativeActionBatch) -> Tensor:
    return sample_action_prefix_lengths(
        actions, max_delay=batch.get("_rtc_training_max_delay"),
    )


def _committed_prefix(
    batch: Mapping[str, Any], observed: ObservedHistory, actions: NativeActionBatch
) -> NativeActionBatch:
    if observed.committed_actions is not None:
        prefix = observed.committed_actions
        lengths = committed_action_prefix_lengths(prefix)
    else:
        lengths = batch.get("_action_prefix_lengths", batch.get("action_prefix_lengths"))
        if lengths is None:
            lengths = _sample_training_prefix(batch, actions)
        prefix = make_committed_action_prefix(actions, lengths.to(actions.values.device))
    maximum = batch.get("_rtc_training_max_delay")
    if maximum is not None and (lengths > maximum).any():
        raise ValueError("training action prefix exceeds rtc_training_max_delay")
    return prefix


def _action_training_sample(
    batch: Mapping[str, Any],
    observed: ObservedHistory,
    actions: NativeActionBatch,
    expert: TemporalNativeActionExpert,
) -> tuple[Tensor, ActionFlowTrainingSample]:
    normalized, action_valid = expert.normalize_actions(actions)
    noise = batch.get("action_noise")
    if noise is None:
        noise = torch.randn_like(normalized)
    sigma = batch.get("action_sigma")
    if sigma is None:
        # Deployment advances every uncommitted step at one common ODE time;
        # training represents the same state explicitly as [B,T].
        sigma = (expert.sample_sigma(normalized.shape[0], normalized.device)
                 if hasattr(expert, "sample_sigma")
                 else torch.rand(normalized.shape[0], device=normalized.device))
    prefix = _committed_prefix(batch, observed, actions)
    prefix_clean, _ = expert.normalize_actions(prefix)
    return normalized, prepare_action_flow_training(
        normalized,
        action_valid,
        noise,
        sigma,
        committed_prefix=prefix,
        prefix_clean=prefix_clean,
    )


def _readout_losses(
    readout: WorldReadout,
    batch: Mapping[str, Any],
    z_raw: Tensor,
    group_valid: Tensor,
    *,
    detach_parameters: bool = False,
) -> list[tuple[Tensor, Tensor]]:
    """Same physical reconstruction/delta definitions for targets and forecasts."""
    arguments = dict(
        anchor_patches=batch["anchor_patches"],
        anchor_patch_valid=batch["anchor_patch_valid"],
        z_raw=z_raw,
        group_valid=group_valid,
        scene_patch_count=batch["future_patches"].shape[2],
    )
    if detach_parameters:
        # D remains trainable on real future representations. Forecast losses
        # differentiate through D's input, but must not move the measuring stick.
        state = {
            name: value.detach()
            for name, value in list(readout.named_parameters()) + list(readout.named_buffers())
        }
        scene_prediction, body_prediction = torch.func.functional_call(
            readout, state, (), arguments, strict=True
        )
    else:
        scene_prediction, body_prediction = readout(**arguments)
    scene_valid = (
        batch["future_patch_valid"][:, :, :, None].expand_as(scene_prediction)
        & group_valid[:, :, None, None]
    )
    scene_loss, scene_sample_valid = masked_per_sample_mse(
        scene_prediction, batch["future_patches"], scene_valid
    )
    body_valid = (
        batch["future_measurement_valid"]
        & group_valid[:, :, None, None]
    )
    body_loss, body_sample_valid = masked_per_sample_mse(
        body_prediction,
        batch["future_measurements"],
        body_valid,
        element_weight=batch.get("future_measurement_weights"),
    )
    scene_delta_valid = scene_valid[:, 1:] & scene_valid[:, :-1]
    scene_delta_loss, scene_delta_sample_valid = masked_per_sample_mse(
        scene_prediction[:, 1:] - scene_prediction[:, :-1],
        batch["future_patches"][:, 1:] - batch["future_patches"][:, :-1],
        scene_delta_valid,
    )
    body_delta_valid = body_valid[:, 1:] & body_valid[:, :-1]
    if "future_measurement_pair_valid" in batch:
        body_delta_valid = body_delta_valid & batch["future_measurement_pair_valid"][None, :, :, None]
    body_delta_loss, body_delta_sample_valid = masked_per_sample_mse(
        body_prediction[:, 1:] - body_prediction[:, :-1],
        batch["future_measurements"][:, 1:] - batch["future_measurements"][:, :-1],
        body_delta_valid,
        element_weight=(
            torch.minimum(
                batch["future_measurement_weights"][:, 1:],
                batch["future_measurement_weights"][:, :-1],
            )
            if "future_measurement_weights" in batch else None
        ),
    )
    time_loss, time_valid = _combine([
        (scene_delta_loss, scene_delta_sample_valid, 1.0),
        (body_delta_loss, body_delta_sample_valid, 1.0),
    ])

    return [
        (scene_loss, scene_sample_valid),
        (body_loss, body_sample_valid),
        (time_loss, time_valid),
    ]


class RepresentationObjective(nn.Module):
    """Learn E/D/T jointly while keeping Qwen features outside the trainable graph."""

    def __init__(
        self,
        encoder: FutureTrajectoryEncoder | EMAFutureTrajectoryEncoder,
        readout: WorldReadout,
        teacher: TemporalNativeActionExpert,
        *,
        scene_weight: float = 1.0,
        body_weight: float = 1.0,
        action_weight: float = 1.0,
        time_weight: float = 0.25,
        regularizer_weight: float = 0.001,
    ) -> None:
        super().__init__()
        self.encoder, self.readout, self.teacher = encoder, readout, teacher
        self.weights = (
            float(scene_weight), float(body_weight), float(action_weight), float(time_weight)
        )
        self.regularizer_weight = float(regularizer_weight)
        if any(value < 0 for value in (*self.weights, self.regularizer_weight)):
            raise ValueError("representation loss weights must be non-negative")

    def forward(self, batch: Mapping[str, Any]) -> PhaseLoss:
        observed: ObservedHistory = batch["observed"]
        features: ObservationFeatures = batch["features"]
        actions: NativeActionBatch = batch["actions"]
        z_raw = batch.get("_z_raw")
        if z_raw is None:
            if isinstance(self.encoder, EMAFutureTrajectoryEncoder):
                z_raw = self.encoder.encode_online(**_encoder_arguments(batch))
            else:
                z_raw = self.encoder(**_encoder_arguments(batch))
        group_valid = batch["group_valid"].bool()
        (scene_loss, scene_sample_valid), (body_loss, body_sample_valid), (time_loss, time_valid) = (
            _readout_losses(
                self.readout, batch, z_raw, group_valid,
            )
        )
        _, action_flow = _action_training_sample(batch, observed, actions, self.teacher)
        predicted = self.teacher.velocity(
            normalized_action_samples=action_flow.mixed,
            action_valid=action_flow.model_valid,
            sigma_a=action_flow.sigma,
            **_expert_arguments(observed, features, z_raw, batch["group_valid"]),
        )
        action_loss, action_sample_valid = masked_per_sample_mse(
            predicted, action_flow.target, action_flow.loss_valid
        )
        total, valid = _combine([
            (scene_loss, scene_sample_valid, self.weights[0]),
            (body_loss, body_sample_valid, self.weights[1]),
            (action_loss, action_sample_valid, self.weights[2]),
            (time_loss, time_valid, self.weights[3]),
        ])
        regularizer, regularizer_valid = representation_regularizer_per_sample(
            z_raw, group_valid
        )
        total = total + self.regularizer_weight * torch.where(
            regularizer_valid, regularizer, torch.zeros_like(regularizer)
        )
        rms = z_raw.float().square().mean(dim=(1, 2, 3)).clamp_min(1e-12).sqrt()
        token_std = z_raw.float().std(dim=-2, unbiased=False).mean(dim=-1)
        action_mse = (
            action_loss[action_sample_valid].mean()
            if action_sample_valid.any() else action_loss.sum() * 0
        )
        metrics = {
            "scene_mse": scene_loss[scene_sample_valid].mean() if scene_sample_valid.any() else scene_loss.sum() * 0,
            "body_mse": body_loss[body_sample_valid].mean() if body_sample_valid.any() else body_loss.sum() * 0,
            "teacher_action_mse": action_mse,
            "action_clean_prefix_fraction": action_flow.model_valid.any(dim=-1).logical_and(
                ~action_flow.loss_valid.any(dim=-1)
            ).float().mean(),
            "temporal_delta_mse": time_loss[time_valid].mean() if time_valid.any() else time_loss.sum() * 0,
            "latent_rms": rms.mean(),
            "latent_token_std": token_std[group_valid].mean() if group_valid.any() else token_std.sum() * 0,
            "latent_regularizer": (
                regularizer[regularizer_valid].mean()
                if regularizer_valid.any() else regularizer.sum() * 0
            ),
        }
        metric_weights = {name: valid.detach().double().sum() for name in metrics}
        metric_weights.update({
            "scene_mse": scene_sample_valid.detach().double().sum(),
            "body_mse": body_sample_valid.detach().double().sum(),
            "temporal_delta_mse": time_valid.detach().double().sum(),
            "latent_regularizer": regularizer_valid.detach().double().sum(),
            "latent_token_std": group_valid.detach().double().sum(),
            "latent_rms": total.new_tensor(total.numel(), dtype=torch.float64),
        })
        metric_weights["teacher_action_mse"] = action_sample_valid.detach().double().sum()
        return PhaseLoss(total, valid, metrics, metric_weights=metric_weights)


class ForecastObjective(nn.Module):
    """Regress a full open-loop rollout to stop-gradient future process targets."""

    def __init__(
        self,
        *,
        reference_encoder: FutureTrajectoryEncoder | EMAFutureTrajectoryEncoder,
        normalizer: LatentNormalizer,
        history_state_adapters: nn.Module,
        history_encoder: PredictiveHistoryEncoder,
        latent_transition: TemporalLatentTransition,
        prior_weight: float = 1.0,
        world_weight: float = 1.0,
        readout: WorldReadout | None = None,
        readout_weight: float = 0.0,
        control_teacher: TemporalNativeActionExpert | None = None,
        control_weight: float = 0.0,
        control_scope: str = "prior",
        temporal_weight: float = 0.25,
        freeze_reference: bool = True,
        control_reader: str = "teacher",
        latent_prediction_space: str = "whitened",
    ) -> None:
        super().__init__()
        self.reference_encoder = reference_encoder
        self.normalizer = normalizer
        if latent_prediction_space not in {"raw", "whitened"}:
            raise ValueError("latent prediction space must be raw or whitened")
        self.latent_prediction_space = latent_prediction_space
        self.history_state_adapters = history_state_adapters
        self.history_encoder = history_encoder
        self.latent_transition = latent_transition
        self.readout = readout
        self.weights = (float(prior_weight), float(world_weight))
        self.readout_weight = float(readout_weight)
        self.control_teacher = control_teacher
        self.control_weight = float(control_weight)
        self.control_scope = str(control_scope)
        self.control_reader = str(control_reader)
        self._validated_control_student_id: int | None = None
        if self.control_reader not in {"teacher", "student", "both"}:
            raise ValueError("control reader must be teacher, student or both")
        if self.control_scope not in {"prior", "both"}:
            raise ValueError("control scope must be prior or both")
        if not 0 <= self.control_weight < float("inf"):
            raise ValueError("control weight must be finite and non-negative")
        if self.control_weight and control_teacher is None:
            raise ValueError("control consistency requires an action teacher")
        self.temporal_weight = float(temporal_weight)
        self.freeze_reference = bool(freeze_reference)
        if min(*self.weights, self.readout_weight, self.temporal_weight) < 0:
            raise ValueError("forecast weights must be non-negative")
        if self.readout_weight and readout is None:
            raise ValueError("forecast readout supervision requires a readout")
        if self.freeze_reference:
            for module in (reference_encoder, normalizer, readout):
                if module is not None:
                    freeze_module(module)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_reference:
            for module in (self.reference_encoder, self.normalizer, self.readout):
                if module is not None:
                    module.eval()
        return self

    def _both_reader_response(
        self, *, control_student, teacher_state, arguments, target_velocity,
        sample, response_mask, mode,
    ):
        """Average T/T and G/T responses at one identical action-flow sample.

        The observed-future teacher target is evaluated once by the caller.
        Both readers use detached parameters/buffers and the same predicted
        latent object; only the causal prediction path receives this gradient.
        """
        parts, metrics, weights, diagnostic_parts = [], {}, {}, {}
        for name, reader, coefficient in (
            ("teacher", self.control_teacher, DUAL_RESPONSE_TEACHER_WEIGHT),
            ("student", control_student, DUAL_RESPONSE_STUDENT_WEIGHT),
        ):
            state = teacher_state if name == "teacher" else {
                key: value.detach()
                for key, value in list(reader.named_parameters()) + list(reader.named_buffers())
            }
            predicted = _response_velocity(reader, state, arguments)
            loss, valid = masked_per_sample_mse(predicted, target_velocity.detach(), response_mask)
            parts.append((loss, valid, coefficient))
            metrics[f"{mode}_control_{name}_response_weight"] = loss.new_tensor(coefficient * self.control_weight)
            weights[f"{mode}_control_{name}_response_weight"] = valid.detach().double().sum()
            for diagnostic, target in (
                ("objective_mse", target_velocity),
                ("response_mse", target_velocity),
                ("predicted_flow_mse", sample.target),
            ):
                values, eligible = masked_per_sample_mse(predicted.detach(), target.detach(), response_mask)
                key = f"{mode}_control_{name}_{diagnostic}"
                metrics[key] = values[eligible].mean() if eligible.any() else values.sum() * 0
                weights[key] = eligible.detach().double().sum()
                diagnostic_parts.setdefault(diagnostic, []).append((values, eligible, coefficient))
        total, valid = _combine(parts)
        for diagnostic, terms in diagnostic_parts.items():
            values, eligible = _combine(terms)
            key = f"{mode}_control_{diagnostic}"
            metrics[key] = values[eligible].mean() if eligible.any() else values.sum() * 0
            weights[key] = eligible.detach().double().sum()
        values, eligible = masked_per_sample_mse(target_velocity.detach(), sample.target.detach(), response_mask)
        key = f"{mode}_control_oracle_flow_mse"
        metrics[key] = values[eligible].mean() if eligible.any() else values.sum() * 0
        weights[key] = eligible.detach().double().sum()
        return total, valid, metrics, weights

    def forward(
        self, batch: Mapping[str, Any], *,
        control_student: TemporalNativeActionExpert | None = None,
    ) -> PhaseLoss:
        if self.control_reader in {"student", "both"}:
            teacher = self.control_teacher
            if control_student is None or teacher is None or control_student is teacher:
                raise ValueError("student control reader requires independent teacher and student experts")
            if self._validated_control_student_id != id(control_student):
                if (control_student.protocol_ids != teacher.protocol_ids
                        or control_student.latent_shape != teacher.latent_shape
                        or control_student.action_steps != teacher.action_steps
                       ):
                    raise ValueError("student and teacher action expert contracts disagree")
                for protocol_id in teacher.protocol_ids:
                    teacher_adapter = teacher.adapters[protocol_id]
                    student_adapter = control_student.adapters[protocol_id]
                    if (teacher_adapter.normalization_space != student_adapter.normalization_space
                            or not torch.equal(teacher_adapter.mean, student_adapter.mean)
                            or not torch.equal(teacher_adapter.std, student_adapter.std)):
                        raise ValueError("student and teacher action normalization disagrees")
                # Statistics are fixed buffers. Avoid a GPU synchronization on
                # every update after verifying this particular G instance.
                self._validated_control_student_id = id(control_student)
        observed: ObservedHistory = batch["observed"]
        features: ObservationFeatures = batch["features"]
        actions: NativeActionBatch = batch["actions"]
        with torch.no_grad():
            z_raw = batch.get("_z_target_raw")
            if z_raw is None:
                z_raw = self.reference_encoder(**_encoder_arguments(batch))
            target_latent = (
                z_raw.detach() if self.latent_prediction_space == "raw"
                else self.normalizer.whiten(z_raw.detach())
            )
        condition = _history_condition(
            observed=observed, features=features,
            state_adapters=self.history_state_adapters, history_encoder=self.history_encoder,
        )
        mode = str(batch.get("forecast_mode", "prior"))
        if mode not in {"prior", "world"}:
            raise ValueError("forecast_mode must be prior or world")
        mode_group_valid = batch["group_valid"].bool()
        if mode == "world":
            coverage = batch["world_action_group_valid"]
            if coverage.shape != mode_group_valid.shape:
                raise ValueError("world-action group eligibility must have shape [B,G]")
            # Missing commands also make later action-conditioned rollouts
            # uncertain. This masks only this loss, never drops policy rows.
            covered_prefix = coverage.bool().long().cumprod(dim=1).bool()
            mode_group_valid = mode_group_valid & covered_prefix
        prefix = _committed_prefix(batch, observed, actions) if mode == "prior" else None
        transition_output = self.latent_transition(
            condition, mode=mode,
            action_condition=(
                actions if mode == "world"
                else prefix if prefix is not None and bool(prefix.valid.any()) else None
            ),
        )
        prediction = transition_output
        predicted_raw = prediction_to_raw(
            prediction, self.normalizer, space=self.latent_prediction_space,
        )
        mask = mode_group_valid[:, :, None, None].expand_as(target_latent)
        prediction_loss, prediction_valid = masked_per_sample_mse(prediction, target_latent, mask)
        parts = [(prediction_loss, prediction_valid, self.weights[0 if mode == "prior" else 1])]
        metric_weights = {f"{mode}_prediction_mse": prediction_valid.detach().double().sum()}
        metrics = {f"{mode}_prediction_mse": (
            prediction_loss[prediction_valid].mean()
            if prediction_valid.any() else prediction_loss.sum() * 0
        )}
        for group in range(mode_group_valid.shape[1]):
            group_mask = torch.zeros_like(mode_group_valid)
            group_mask[:, group] = mode_group_valid[:, group]
            values, valid = masked_per_sample_mse(
                prediction, target_latent, group_mask[:, :, None, None].expand_as(target_latent)
            )
            metrics[f"{mode}_prediction_mse_group_{group}"] = (
                values[valid].mean() if valid.any() else values.sum() * 0
            )
            metric_weights[f"{mode}_prediction_mse_group_{group}"] = valid.detach().double().sum()
        if self.readout is not None and self.readout_weight:
            decoded = _readout_losses(
                self.readout, batch, predicted_raw, mode_group_valid,
                detach_parameters=True,
            )
            for (values, valid), name, weight in zip(
                decoded, ("scene", "body", "temporal_delta"), (1.0, 1.0, self.temporal_weight),
                strict=True,
            ):
                parts.append((values, valid, self.readout_weight * weight))
                metrics[f"{mode}_{name}_mse"] = values[valid].mean() if valid.any() else values.sum() * 0
                metric_weights[f"{mode}_{name}_mse"] = valid.detach().double().sum()
        control_active = (
            self.control_teacher is not None and self.control_weight > 0
            and (mode == "prior" or self.control_scope == "both")
        )
        metrics[f"{mode}_control_response_active"] = prediction.new_tensor(float(control_active))
        metrics[f"{mode}_control_response_weight"] = prediction.new_tensor(
            self.control_weight if control_active else 0.0
        )
        if control_active:
            prepared = dict(batch)
            if prefix is not None:
                prepared["_action_prefix_lengths"] = committed_action_prefix_lengths(prefix)
            _, sample = _action_training_sample(prepared, observed, actions, self.control_teacher)
            # The target is always the true-future teacher response. Detached
            # reader weights keep the gradient on the predicted latent only.
            teacher_state = {
                name: value.detach()
                for name, value in (
                    list(self.control_teacher.named_parameters())
                    + list(self.control_teacher.named_buffers())
                )
            }
            arguments = {
                **_expert_arguments(
                    observed, features, z_raw.detach(), mode_group_valid,
                ),
                "normalized_action_samples": sample.mixed,
                "action_valid": sample.model_valid,
                "sigma_a": sample.sigma,
            }
            arguments = {k: v.detach() if isinstance(v, Tensor) else v for k, v in arguments.items()}
            with torch.no_grad():
                target_velocity = _response_velocity(self.control_teacher, teacher_state, arguments)
            arguments["z_raw"] = predicted_raw
            if self.control_reader == "both":
                response_mask = sample.loss_valid & mode_group_valid.any(dim=1)[:, None, None]
                control_loss, control_valid, diagnostics, diagnostic_weights = self._both_reader_response(
                    control_student=control_student, teacher_state=teacher_state, arguments=arguments,
                    target_velocity=target_velocity, sample=sample, response_mask=response_mask, mode=mode,
                )
                parts.append((control_loss, control_valid, self.control_weight))
                metrics.update(diagnostics)
                metric_weights.update(diagnostic_weights)
            else:
                # Preserve the legacy single-reader forward, metric and RNG order.
                reader = self.control_teacher
                reader_state = teacher_state
                if self.control_reader == "student":
                    reader = control_student
                    reader_state = {
                        name: value.detach()
                        for name, value in (
                            list(reader.named_parameters()) + list(reader.named_buffers())
                        )
                    }
                predicted_velocity = _response_velocity(reader, reader_state, arguments)
                response_mask = sample.loss_valid & mode_group_valid.any(dim=1)[:, None, None]
                control_loss, control_valid = masked_per_sample_mse(
                    predicted_velocity, target_velocity.detach(), response_mask
                )
                parts.append((control_loss, control_valid, self.control_weight))
                for name, prediction, target in (
                    ("control_objective_mse", predicted_velocity, target_velocity),
                    ("control_response_mse", predicted_velocity, target_velocity),
                    ("control_predicted_flow_mse", predicted_velocity, sample.target),
                    ("control_oracle_flow_mse", target_velocity, sample.target),
                ):
                    values, valid = masked_per_sample_mse(prediction.detach(), target.detach(), response_mask)
                    metrics[f"{mode}_{name}"] = values[valid].mean() if valid.any() else values.sum() * 0
                    metric_weights[f"{mode}_{name}"] = valid.detach().double().sum()
        total, valid = _combine(parts)
        all_metric_weights = {name: valid.detach().double().sum() for name in metrics}
        all_metric_weights.update(metric_weights)
        return PhaseLoss(total, valid, metrics, metric_weights=all_metric_weights)


class WorldModelObjective(nn.Module):
    """Jointly learn action-useful future coordinates and their causal dynamics."""

    def __init__(
        self,
        representation: RepresentationObjective,
        forecast: ForecastObjective,
        normalizer: LatentNormalizer,
        *,
        normalizer_decay: float = 0.99,
        normalizer_calibration_updates: int = 0,
        forecast_warmup_updates: int = 0,
        forecast_ramp_updates: int = 0,
        optimizer_update_offset: int = 0,
        require_complete_initial_calibration: bool = False,
    ) -> None:
        super().__init__()
        if not 0 <= normalizer_decay < 1:
            raise ValueError("normalizer decay must be in [0,1)")
        if min(
            normalizer_calibration_updates,
            forecast_warmup_updates,
            forecast_ramp_updates,
            optimizer_update_offset,
        ) < 0:
            raise ValueError("world-model stage update counts cannot be negative")
        if normalizer_calibration_updates > forecast_warmup_updates:
            raise ValueError("normalizer calibration must finish before forecast warmup")
        if forecast.latent_prediction_space == "raw" and normalizer_calibration_updates:
            raise ValueError("raw latent prediction does not use normalizer calibration")
        self.representation = representation
        self.forecast = forecast
        self.normalizer = normalizer
        self.normalizer_decay = float(normalizer_decay)
        self.normalizer_calibration_updates = int(normalizer_calibration_updates)
        self.forecast_warmup_updates = int(forecast_warmup_updates)
        self.forecast_ramp_updates = int(forecast_ramp_updates)
        self.optimizer_update_offset = int(optimizer_update_offset)
        self.require_complete_initial_calibration = bool(
            require_complete_initial_calibration and forecast.latent_prediction_space == "whitened"
        )

    def _forecast_weight(self, update: int) -> float:
        if update < self.forecast_warmup_updates:
            return 0.0
        if self.forecast_ramp_updates == 0:
            return 1.0
        return min(
            1.0,
            (update - self.forecast_warmup_updates + 1) / self.forecast_ramp_updates,
        )

    def forward(
        self, batch: Mapping[str, Any], *,
        control_student: TemporalNativeActionExpert | None = None,
    ) -> PhaseLoss:
        if self.forecast.control_reader == "student" and control_student is None:
            raise ValueError("executor response transfer requires the joint policy student")
        prepared = dict(batch)
        actions: NativeActionBatch = prepared["actions"]
        observed: ObservedHistory = prepared["observed"]
        if observed.committed_actions is None and not any(
            name in prepared for name in ("_action_prefix_lengths", "action_prefix_lengths")
        ):
            prepared["_action_prefix_lengths"] = _sample_training_prefix(prepared, actions)
        encoder = self.representation.encoder
        local_update = _optimizer_update(
            prepared,
            fallback=(
                max(
                    0,
                    int(encoder.last_target_update.item())
                    + 1
                    - self.optimizer_update_offset,
                )
                if isinstance(encoder, EMAFutureTrajectoryEncoder) else 0
            ),
        )
        absolute_update = self.optimizer_update_offset + local_update
        if (self.require_complete_initial_calibration and not bool(self.normalizer.fitted)
                and not bool(prepared["group_valid"].any(dim=0).all())):
            raise ValueError("first scratch calibration batch must cover every future group")
        target_advanced = False
        if isinstance(encoder, EMAFutureTrajectoryEncoder):
            if self.training:
                target_advanced = encoder.update_target(absolute_update)
            z_raw = encoder.encode_online(**_encoder_arguments(prepared))
            with torch.no_grad():
                z_target_raw = encoder(**_encoder_arguments(prepared))
        else:
            z_raw = encoder(**_encoder_arguments(prepared))
            z_target_raw = z_raw.detach()
        prepared["_z_raw"] = z_raw
        prepared["_z_target_raw"] = z_target_raw
        with torch.no_grad():
            latent_valid = prepared["group_valid"][:, :, None, None].expand_as(z_raw)
            latent_drift, latent_drift_valid = masked_per_sample_mse(
                z_raw.detach(), z_target_raw, latent_valid
            )
            token_valid = prepared["group_valid"][:, :, None].expand(z_raw.shape[:3])
            token_cosine = torch.nn.functional.cosine_similarity(
                z_raw.detach().float(), z_target_raw.float(), dim=-1, eps=1e-8
            )
            cosine_denominator = token_valid.float().sum().clamp_min(1)
            latent_cosine = (
                torch.where(token_valid, token_cosine, torch.zeros_like(token_cosine)).sum()
                / cosine_denominator
            )
            target_group_rms = z_target_raw.float().square().mean(dim=(-1, -2)).sqrt()
            target_group_token_std = z_target_raw.float().std(dim=-2, unbiased=False).mean(dim=-1)
            target_groups = prepared["group_valid"].bool()
            target_group_count = target_groups.float().sum().clamp_min(1)
            target_rms = torch.where(target_groups, target_group_rms, 0).sum() / target_group_count
            target_token_std = torch.where(target_groups, target_group_token_std, 0).sum() / target_group_count
        representation = self.representation(prepared)
        ema = {}
        normalizer_update = (
            self.forecast.latent_prediction_space == "whitened"
            and self.training and local_update < self.normalizer_calibration_updates
        )
        if (self.forecast.latent_prediction_space == "whitened"
                and isinstance(encoder, EMAFutureTrajectoryEncoder)):
            if normalizer_update:
                normalizer_update = target_advanced and encoder.mark_normalizer_update(
                    absolute_update
                )
            elif self.training and local_update >= self.normalizer_calibration_updates:
                encoder.freeze_normalizer_stage()
        if normalizer_update:
            ema = self.normalizer.update_ema(
                z_target_raw, prepared["group_valid"], decay=self.normalizer_decay,
                distributed=dist.is_available() and dist.is_initialized(),
            )
            if (
                isinstance(encoder, EMAFutureTrajectoryEncoder)
                and local_update + 1 >= self.normalizer_calibration_updates
            ):
                encoder.freeze_normalizer_stage()
        forecast_weight = self._forecast_weight(local_update)
        force_forecast = bool(prepared.get("_force_forecast_active", False))
        if forecast_weight > 0 or force_forecast:
            forecast = self.forecast(prepared, control_student=control_student)
            if force_forecast and forecast_weight == 0:
                forecast_weight = 1.0
            forecast_inactive = z_raw.new_zeros(())
        else:
            forecast = PhaseLoss(
                per_sample=representation.per_sample * 0.0,
                valid=torch.zeros_like(representation.valid),
                metrics={},
            )
            forecast_inactive = z_raw.new_ones(())
        valid = representation.valid | forecast.valid
        total = torch.where(
            representation.valid, representation.per_sample,
            torch.zeros_like(representation.per_sample),
        ) + torch.where(
            forecast.valid,
            forecast.per_sample * forecast_weight,
            torch.zeros_like(forecast.per_sample),
        )
        metrics = {**representation.metrics, **forecast.metrics}
        metrics.update({
            name: total.new_tensor(value) for name, value in ema.items()
        })
        metrics["forecast_objective_weight"] = total.new_tensor(forecast_weight)
        metrics["forecast_inactive"] = forecast_inactive
        metrics["target_encoder_ema_update"] = total.new_tensor(float(target_advanced))
        metrics["absolute_optimizer_update"] = total.new_tensor(float(absolute_update))
        metrics["normalizer_calibration_active"] = total.new_tensor(float(normalizer_update))
        metrics["online_target_latent_mse"] = (
            latent_drift[latent_drift_valid].mean()
            if latent_drift_valid.any() else latent_drift.sum() * 0
        )
        metrics["online_target_latent_cosine"] = latent_cosine
        metrics["target_latent_rms"] = target_rms
        metrics["target_latent_token_std"] = target_token_std
        metrics["world_model_objective"] = total[valid].mean() if valid.any() else total.sum() * 0
        metric_weights = {name: valid.detach().double().sum() for name in metrics}
        metric_weights.update(representation.metric_weights or {})
        metric_weights.update(forecast.metric_weights or {})
        for name in ("target_latent_rms", "target_latent_token_std"):
            metric_weights[name] = target_groups.detach().double().sum()
        return PhaseLoss(total, valid, metrics, metric_weights=metric_weights)


class PolicyObjective(nn.Module):
    """Train G on deterministic prior rollouts from deployable inputs only."""

    def __init__(
        self,
        *,
        student: TemporalNativeActionExpert,
        normalizer: LatentNormalizer,
        history_state_adapters: nn.Module,
        history_encoder: PredictiveHistoryEncoder,
        latent_transition: TemporalLatentTransition,
        freeze_world: bool = True,
        latent_prediction_space: str = "whitened",
    ) -> None:
        super().__init__()
        self.student = student
        self.normalizer = normalizer
        if latent_prediction_space not in {"raw", "whitened"}:
            raise ValueError("latent prediction space must be raw or whitened")
        self.latent_prediction_space = latent_prediction_space
        self.history_state_adapters = history_state_adapters
        self.history_encoder = history_encoder
        self.latent_transition = latent_transition
        self.freeze_world = bool(freeze_world)
        if self.freeze_world:
            for module in self._world_modules():
                freeze_module(module)

    def _world_modules(self):
        return (
            self.normalizer, self.history_state_adapters,
            self.history_encoder, self.latent_transition,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_world:
            for module in self._world_modules():
                module.eval()
        return self

    def forward(self, batch: Mapping[str, Any]) -> PhaseLoss:
        observed: ObservedHistory = batch["observed"]
        features: ObservationFeatures = batch["features"]
        actions: NativeActionBatch = batch["actions"]
        group_valid = batch["group_valid"]
        prefix = _committed_prefix(batch, observed, actions)
        with torch.no_grad() if self.freeze_world else contextlib.nullcontext():
            condition = _history_condition(
                observed=observed, features=features,
                state_adapters=self.history_state_adapters, history_encoder=self.history_encoder,
            )
            transition_output = self.latent_transition(
                condition, mode="prior",
                action_condition=prefix if bool(prefix.valid.any()) else None,
            )
            latent = prediction_to_raw(
                transition_output, self.normalizer, space=self.latent_prediction_space,
            )
        # Future-label availability belongs in loss eligibility, not policy inputs.
        predicted_valid = torch.ones_like(group_valid, dtype=torch.bool)
        prepared = dict(batch)
        # Reuse this exact prefix for both predictor and action-flow training.
        prepared["_action_prefix_lengths"] = committed_action_prefix_lengths(prefix)
        _, action_flow = _action_training_sample(prepared, observed, actions, self.student)
        velocity = self.student.velocity(
            normalized_action_samples=action_flow.mixed,
            action_valid=action_flow.model_valid,
            sigma_a=action_flow.sigma,
            **_expert_arguments(observed, features, latent, predicted_valid),
        )
        per_sample, valid = masked_per_sample_mse(velocity, action_flow.target, action_flow.loss_valid)
        valid = valid & group_valid.any(dim=1)
        metrics = {
            "student_action_mse": per_sample[valid].mean() if valid.any() else per_sample.sum() * 0,
            "policy_origin_predicted": per_sample.new_tensor(1.0),
            "action_clean_prefix_fraction": action_flow.model_valid.any(dim=-1).logical_and(
                ~action_flow.loss_valid.any(dim=-1)
            ).float().mean(),
        }
        return PhaseLoss(per_sample, valid, metrics)


class JointObjective(nn.Module):
    """Continue world learning while adapting G to predicted future latents.

    Every optimizer update carries both objectives. The action loss trains H/F
    through a deployable prior rollout; future targets remain in the world loss.
    """

    def __init__(
        self,
        world: WorldModelObjective,
        policy: PolicyObjective,
        *,
        world_weight: float = 1.0,
        action_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.world = world
        self.policy = policy
        self.world_weight = float(world_weight)
        self.action_weight = float(action_weight)
        if self.world_weight <= 0 or self.action_weight <= 0:
            raise ValueError("joint world and action weights must be positive")
        if self.policy.freeze_world:
            raise ValueError("joint policy must train predicted world representations")
        if self.world.forecast.latent_prediction_space != self.policy.latent_prediction_space:
            raise ValueError("joint forecast and policy latent prediction spaces must agree")

    def forward(self, batch: Mapping[str, Any]) -> PhaseLoss:
        world = self.world(batch, control_student=self.policy.student)
        policy = self.policy(batch)
        valid = world.valid | policy.valid
        total = (
            torch.where(
                world.valid,
                world.per_sample * self.world_weight,
                torch.zeros_like(world.per_sample),
            )
            + torch.where(
                policy.valid,
                policy.per_sample * self.action_weight,
                torch.zeros_like(policy.per_sample),
            )
        )
        metrics = dict(world.metrics)
        metric_weights = {
            name: world.valid.detach().double().sum() for name in metrics
        }
        metric_weights.update(world.metric_weights or {})
        for name, value in policy.metrics.items():
            output_name = name if name not in metrics else f"student_{name}"
            metrics[output_name] = value
            metric_weights[output_name] = policy.valid.detach().double().sum()
        world_component = (
            world.per_sample[world.valid].mean()
            if world.valid.any() else world.per_sample.sum() * 0
        )
        action_component = (
            policy.per_sample[policy.valid].mean()
            if policy.valid.any() else policy.per_sample.sum() * 0
        )
        metrics.update({
            "joint_world_component": world_component,
            "joint_action_component": action_component,
            "joint_objective": total[valid].mean() if valid.any() else total.sum() * 0,
        })
        joint_weight = valid.detach().double().sum()
        metric_weights.update({
            "joint_world_component": world.valid.detach().double().sum(),
            "joint_action_component": policy.valid.detach().double().sum(),
            "joint_objective": joint_weight,
        })
        return PhaseLoss(total, valid, metrics, metric_weights=metric_weights)


class PhaseModelWithObservation(nn.Module):
    """Run frozen causal observation extraction once, then a concrete phase objective."""

    def __init__(
        self, observation_encoder: nn.Module, objective: nn.Module, *,
        rtc_training_max_delay: int | None = None,
    ) -> None:
        super().__init__()
        self.observation_encoder = observation_encoder
        self.objective = objective
        self.rtc_training_max_delay = rtc_training_max_delay
        freeze_module(self.observation_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        self.observation_encoder.eval()
        self.objective.train(mode)
        return self

    def forward(self, batch: Mapping[str, Any]) -> PhaseLoss:
        prepared = dict(batch)
        # One config-owned bound for teacher, forecast/control and student paths.
        # Do not pre-sample here: preserve their existing RNG order and sharing.
        prepared["_rtc_training_max_delay"] = self.rtc_training_max_delay
        with torch.no_grad():
            features = self.observation_encoder(prepared["observed"])
        prepared["features"] = features
        prepared["anchor_patches"] = features.current_visual
        prepared["anchor_patch_valid"] = torch.ones(
            features.current_visual.shape[:2], device=features.current_visual.device,
            dtype=torch.bool,
        )
        return self.objective(prepared)


__all__ = [
    "ForecastObjective", "JointObjective", "PhaseModelWithObservation", "PolicyObjective",
    "RepresentationObjective", "WorldModelObjective",
]
