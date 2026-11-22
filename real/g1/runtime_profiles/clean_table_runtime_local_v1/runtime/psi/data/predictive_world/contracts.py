"""Typed, validated separation between deployable observations and supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


def sanitize_masked_tensor(values: Tensor, valid: Tensor, *, name: str) -> Tensor:
    """Return finite values with invalid locations zeroed; reject valid NaN/Inf."""

    if values.shape != valid.shape:
        raise ValueError(f"{name} values and valid mask must have identical shapes")
    mask = valid.to(device=values.device, dtype=torch.bool)
    if not torch.isfinite(values[mask]).all():
        raise ValueError(f"{name} contains non-finite values at valid locations")
    return torch.where(mask, values, torch.zeros_like(values))


@dataclass(frozen=True)
class RaggedPatchBatch:
    values: Tensor
    offsets: Tensor
    grid_hw: Tensor
    content_valid: Tensor
    sample_indices: Tensor
    group_indices: Tensor
    camera_ids: tuple[str, ...]
    actual_times: Tensor

    def validate(self, *, batch_size: int, groups: int) -> None:
        if self.values.ndim != 2:
            raise ValueError("ragged patch values must have shape [N,D]")
        images = len(self.camera_ids)
        if self.offsets.shape != (images + 1,) or self.offsets.dtype != torch.long:
            raise ValueError("ragged offsets must be int64 with one terminal offset")
        if self.grid_hw.shape != (images, 2):
            raise ValueError("grid_hw must have shape [num_images,2]")
        if self.content_valid.shape != (self.values.shape[0],):
            raise ValueError("content_valid must have one entry per patch")
        if self.sample_indices.shape != (images,) or self.group_indices.shape != (images,):
            raise ValueError("ragged image mappings must have one entry per image")
        if self.actual_times.shape != (images,):
            raise ValueError("actual_times must have one entry per image")
        if images and (self.offsets[0].item() != 0 or self.offsets[-1].item() != len(self.values)):
            raise ValueError("ragged offsets must cover the patch tensor exactly")
        if (self.offsets[1:] < self.offsets[:-1]).any():
            raise ValueError("ragged offsets must be monotonic")
        if images and ((self.sample_indices < 0).any() or (self.sample_indices >= batch_size).any()):
            raise ValueError("ragged sample index is outside the batch")
        if images and ((self.group_indices < 0).any() or (self.group_indices >= groups).any()):
            raise ValueError("ragged group index is outside the configured future")
        expected = (self.grid_hw[:, 0] * self.grid_hw[:, 1]).to(self.offsets.device)
        if images and not torch.equal(self.offsets[1:] - self.offsets[:-1], expected):
            raise ValueError("grid sizes do not match ragged patch lengths")
        if not torch.isfinite(self.values[self.content_valid.bool()]).all():
            raise ValueError("future patches contain non-finite valid values")
        if not torch.isfinite(self.actual_times).all() or (self.actual_times <= 0).any():
            raise ValueError("future patch times must be finite and positive")


@dataclass(frozen=True)
class NativeActionBatch:
    values: Tensor
    valid: Tensor
    times: Tensor
    protocol_ids: tuple[str, ...]
    native_dims: Tensor
    normalization_spaces: tuple[str, ...]

    def validate(self, *, batch_size: int, horizon: int) -> None:
        if self.values.ndim != 3 or self.values.shape[:2] != (batch_size, horizon):
            raise ValueError("native actions must have shape [B,horizon,Dmax]")
        if self.valid.shape != self.values.shape or self.times.shape != (batch_size, horizon):
            raise ValueError("native action masks/times disagree with values")
        if len(self.protocol_ids) != batch_size or len(self.normalization_spaces) != batch_size:
            raise ValueError("native action metadata must have one entry per sample")
        if self.native_dims.shape != (batch_size,):
            raise ValueError("native_dims must have shape [B]")
        if (self.native_dims <= 0).any() or (self.native_dims > self.values.shape[-1]).any():
            raise ValueError("native_dims must fit inside the padded action width")
        sanitize_masked_tensor(self.values, self.valid, name="demonstrated actions")
        step_valid = self.valid.any(dim=-1)
        if not torch.isfinite(self.times[step_valid]).all():
            raise ValueError("valid demonstrated action times must be finite")
        for row, dim in enumerate(self.native_dims.tolist()):
            if self.valid[row, :, dim:].any():
                raise ValueError("action dimensions beyond native_dim cannot be valid")


@dataclass(frozen=True)
class ObservedHistory:
    qwen_inputs: Mapping[str, Any]
    raw_states: Tensor
    state_valid: Tensor
    state_times: Tensor
    state_schema_ids: tuple[str, ...]
    action_protocol_ids: tuple[str, ...]
    action_native_dims: Tensor
    action_normalization_spaces: tuple[str, ...]
    committed_actions: NativeActionBatch | None = None

    def validate(self, *, batch_size: int, state_count: int, action_horizon: int) -> None:
        if self.raw_states.ndim != 3 or self.raw_states.shape[:2] != (batch_size, state_count):
            raise ValueError("observed raw_states must have shape [B,state_count,Dmax]")
        if self.state_valid.shape != self.raw_states.shape:
            raise ValueError("observed state mask must match raw_states")
        if self.state_times.shape != (batch_size, state_count):
            raise ValueError("observed state_times must have shape [B,state_count]")
        if len(self.state_schema_ids) != batch_size:
            raise ValueError("state schema IDs must have one entry per sample")
        if len(self.action_protocol_ids) != batch_size or len(self.action_normalization_spaces) != batch_size:
            raise ValueError("action protocol metadata must have one entry per sample")
        if self.action_native_dims.shape != (batch_size,) or (self.action_native_dims <= 0).any():
            raise ValueError("action_native_dims must be positive with shape [B]")
        sanitize_masked_tensor(self.raw_states, self.state_valid, name="observed states")
        time_valid = self.state_valid.any(dim=-1)
        if not torch.isfinite(self.state_times[time_valid]).all():
            raise ValueError("valid observed state times must be finite")
        if (self.state_times[time_valid] > 1e-7).any():
            raise ValueError("observed states cannot be later than the decision time")
        if self.committed_actions is not None:
            self.committed_actions.validate(batch_size=batch_size, horizon=action_horizon)
            if self.committed_actions.protocol_ids != self.action_protocol_ids:
                raise ValueError("committed prefix protocols disagree with the observed request")
            if not torch.equal(
                self.committed_actions.native_dims.to(self.action_native_dims.device),
                self.action_native_dims,
            ):
                raise ValueError("committed prefix native dimensions disagree with the request")
            if self.committed_actions.normalization_spaces != self.action_normalization_spaces:
                raise ValueError("committed prefix normalization spaces disagree with the request")


@dataclass(frozen=True)
class FutureSupervision:
    visual: RaggedPatchBatch
    measured_states: Tensor
    measured_valid: Tensor
    measured_times: Tensor
    measured_source_indices: Tensor
    latent_target_valid: Tensor

    def validate(self, *, batch_size: int, groups: int) -> None:
        self.visual.validate(batch_size=batch_size, groups=groups)
        if self.measured_states.ndim != 3 or self.measured_states.shape[0] != batch_size:
            raise ValueError("future measured_states must have shape [B,T,Dmax]")
        if self.measured_valid.shape != self.measured_states.shape:
            raise ValueError("future measurement mask must match measured_states")
        if self.measured_times.shape != self.measured_states.shape[:2]:
            raise ValueError("future measurement times disagree with values")
        if self.measured_source_indices.shape != self.measured_states.shape[:2]:
            raise ValueError("future measurement source indices disagree with values")
        if self.latent_target_valid.shape != (batch_size, groups):
            raise ValueError("latent_target_valid must have shape [B,groups]")
        sanitize_masked_tensor(self.measured_states, self.measured_valid, name="future measurements")
        valid_time = self.measured_valid.any(dim=-1)
        if not torch.isfinite(self.measured_times[valid_time]).all():
            raise ValueError("valid future measurement times must be finite")
        if (self.measured_times[valid_time] <= 0).any():
            raise ValueError("future measurements must be strictly later than the decision time")


@dataclass(frozen=True)
class SampleMetadata:
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    occurrence_indices: Tensor
    split: str

    def validate(self, *, batch_size: int) -> None:
        if any(len(values) != batch_size for values in (self.sample_ids, self.source_ids, self.episode_ids)):
            raise ValueError("metadata IDs must have one entry per sample")
        if self.occurrence_indices.shape != (batch_size,):
            raise ValueError("occurrence_indices must have shape [B]")
        if self.split not in ("train", "train_calibration", "validation", "test"):
            raise ValueError(f"unsupported data split {self.split!r}")


@dataclass(frozen=True)
class PredictiveWorldBatch:
    observed: ObservedHistory
    future: FutureSupervision
    demonstrated: NativeActionBatch
    metadata: SampleMetadata

    def validate(self, *, state_count: int = 21, groups: int = 3, action_horizon: int = 30) -> None:
        batch_size = self.demonstrated.values.shape[0]
        self.observed.validate(
            batch_size=batch_size, state_count=state_count, action_horizon=action_horizon
        )
        self.future.validate(batch_size=batch_size, groups=groups)
        self.demonstrated.validate(batch_size=batch_size, horizon=action_horizon)
        self.metadata.validate(batch_size=batch_size)

    def deployable_observed(self) -> ObservedHistory:
        """The only object accepted by history/prior/policy model boundaries."""

        return self.observed


def assert_observed_only(value: object) -> ObservedHistory:
    if not isinstance(value, ObservedHistory):
        raise TypeError("deployable model boundaries accept ObservedHistory only")
    return value


__all__ = [
    "FutureSupervision",
    "NativeActionBatch",
    "ObservedHistory",
    "PredictiveWorldBatch",
    "RaggedPatchBatch",
    "SampleMetadata",
    "assert_observed_only",
    "sanitize_masked_tensor",
]
