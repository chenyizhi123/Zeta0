"""Temporal native-action flow expert used by independent teacher and student."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.data.predictive_world.contracts import NativeActionBatch, sanitize_masked_tensor
from psi.models.predictive_world.history import ContextResampler
from psi.models.stable_attention import QKNormEncoderLayer


@dataclass(frozen=True)
class ActionFlowTrainingSample:
    """One RTC-compatible action-flow training state."""

    mixed: Tensor
    target: Tensor
    sigma: Tensor
    model_valid: Tensor
    loss_valid: Tensor
    committed_prefix: NativeActionBatch


def _full_native_step_valid(actions: NativeActionBatch) -> Tensor:
    batch, steps, _ = actions.values.shape
    actions.validate(batch_size=batch, horizon=steps)
    result = torch.zeros(batch, steps, device=actions.valid.device, dtype=torch.bool)
    for row, width in enumerate(actions.native_dims.tolist()):
        result[row] = actions.valid[row, :, :width].all(dim=-1)
    return result


def sample_action_prefix_lengths(
    actions: NativeActionBatch, *, max_delay: int | None = None,
) -> Tensor:
    """Sample 0..min(max_delay,L-1) inclusively for L valid leading steps.

    None retains historical 0..L-1 sampling; no valid steps yields prefix zero.
    Limit the sampling support, not the sampled values (which would bias the cap).
    """

    if max_delay is not None and (type(max_delay) is not int or max_delay < 0):
        raise ValueError("max_delay must be a non-negative integer or None")

    fully_valid = _full_native_step_valid(actions)
    leading_valid = fully_valid.long().cumprod(dim=1).sum(dim=1)
    choices = leading_valid if max_delay is None else leading_valid.clamp(max=max_delay + 1)
    random = torch.rand(leading_valid.shape, device=leading_valid.device)
    # floor(U * N) samples [0,N-1], where N <= L, retaining at least one
    # supervised suffix step for every row with any valid leading action.
    return torch.floor(random * choices.float()).long()


def committed_action_prefix_lengths(actions: NativeActionBatch) -> Tensor:
    """Validate and return lengths for an externally supplied RTC prefix."""

    fully_valid = _full_native_step_valid(actions)
    any_valid = actions.valid.any(dim=-1)
    if (any_valid & ~fully_valid).any():
        raise ValueError("committed prefix steps must contain every native dimension")
    leading = fully_valid.long().cumprod(dim=1).bool()
    if (fully_valid & ~leading).any():
        raise ValueError("committed action prefix must be left-contiguous")
    return fully_valid.sum(dim=1).long()


def make_committed_action_prefix(
    actions: NativeActionBatch, prefix_lengths: Tensor
) -> NativeActionBatch:
    """Materialize a dense-horizon, mask-typed prefix from logged native actions."""

    batch, steps, _ = actions.values.shape
    if prefix_lengths.shape != (batch,) or prefix_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("action prefix lengths must be integer [B]")
    if (prefix_lengths < 0).any() or (prefix_lengths > steps).any():
        raise ValueError("action prefix length is outside the action horizon")
    fully_valid = _full_native_step_valid(actions)
    leading_valid = fully_valid.long().cumprod(dim=1).sum(dim=1)
    if (prefix_lengths.to(leading_valid.device) > leading_valid).any():
        raise ValueError("action prefix extends beyond fully valid logged actions")
    step_prefix = (
        torch.arange(steps, device=actions.values.device)[None]
        < prefix_lengths.to(actions.values.device)[:, None]
    )
    valid = actions.valid.bool() & step_prefix[:, :, None]
    values = torch.where(valid, actions.values, torch.zeros_like(actions.values))
    times = torch.where(step_prefix, actions.times, torch.zeros_like(actions.times))
    return NativeActionBatch(
        values=values,
        valid=valid,
        times=times,
        protocol_ids=actions.protocol_ids,
        native_dims=actions.native_dims,
        normalization_spaces=actions.normalization_spaces,
    )


def prepare_action_flow_training(
    clean: Tensor,
    action_valid: Tensor,
    noise: Tensor,
    sigma: Tensor,
    *,
    committed_prefix: NativeActionBatch,
    prefix_clean: Tensor,
) -> ActionFlowTrainingSample:
    """Build clean-prefix/noisy-suffix inputs and suffix-only targets.

    Action flow owns a time per action step; the world predictor has no flow time.  Scalar
    ``[B]`` sigma remains accepted for old callers and is expanded to the
    canonical ``[B,T]`` training representation.
    """

    if clean.shape != noise.shape or clean.shape != action_valid.shape or clean.ndim != 3:
        raise ValueError("action clean/noise/valid tensors must share shape [B,T,D]")
    if prefix_clean.shape != clean.shape:
        raise ValueError("normalized committed prefix must match action tensor shape")
    batch, steps, _ = clean.shape
    prefix_lengths = committed_action_prefix_lengths(committed_prefix).to(clean.device)
    prefix_valid = committed_prefix.valid.to(device=clean.device, dtype=torch.bool)
    if prefix_valid.shape != clean.shape:
        raise ValueError("committed prefix mask must match action tensor shape")
    if sigma.shape == (batch,):
        sigma_steps = sigma[:, None].expand(-1, steps).clone()
    elif sigma.shape == (batch, steps):
        sigma_steps = sigma.clone()
    else:
        raise ValueError("action flow sigma must have shape [B] or [B,T]")
    sigma_steps = sigma_steps.to(device=clean.device, dtype=torch.float32)
    if not torch.isfinite(sigma_steps).all() or (sigma_steps < 0).any() or (sigma_steps > 1).any():
        raise ValueError("action flow sigma must be finite in [0,1]")
    prefix_steps = torch.arange(steps, device=clean.device)[None] < prefix_lengths[:, None]
    sigma_steps = torch.where(prefix_steps, torch.zeros_like(sigma_steps), sigma_steps)
    sigma_view = sigma_steps.to(clean.dtype)[:, :, None]
    mixed = (1 - sigma_view) * clean + sigma_view * noise
    mixed = torch.where(prefix_valid, prefix_clean.to(mixed.dtype), mixed)
    model_valid = action_valid.bool() | prefix_valid
    loss_valid = action_valid.bool() & ~prefix_steps[:, :, None]
    return ActionFlowTrainingSample(
        mixed=mixed,
        target=noise - clean,
        sigma=sigma_steps,
        model_valid=model_valid,
        loss_valid=loss_valid,
        committed_prefix=committed_prefix,
    )


class _TemporalProtocolAdapter(nn.Module):
    def __init__(self, spec: Mapping[str, Any], hidden_dim: int) -> None:
        super().__init__()
        self.dim = int(spec["dim"])
        mean = torch.as_tensor(spec["mean"], dtype=torch.float32)
        std = torch.as_tensor(spec["std"], dtype=torch.float32)
        if mean.shape != (self.dim,) or std.shape != (self.dim,):
            raise ValueError("action protocol statistics disagree with its dimension")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("action protocol statistics must be finite with positive std")
        self.normalization_space = str(spec.get("normalization_space", "native"))
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.input = nn.Sequential(
            nn.Linear(2 * self.dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.dim)
        )


def _time_embedding(value: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=value.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = value.float().reshape(-1, 1) * frequencies[None]
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return torch.nn.functional.pad(result, (0, dim - result.shape[-1]))


class TemporalNativeActionExpert(nn.Module):
    """Generate coordinated native action chunks from history and raw latent plans."""

    def __init__(
        self,
        protocols: Mapping[str, Mapping[str, Any]],
        state_schema_dims: Mapping[str, int],
        *,
        state_statistics: Mapping[str, tuple[Tensor, Tensor]] | None = None,
        qwen_dim: int = 2048,
        context_tokens: int = 16,
        latent_dim: int = 256,
        latent_groups: int = 3,
        latent_slots: int = 8,
        state_history_count: int = 21,
        action_steps: int = 30,
        hidden_dim: int = 768,
        num_blocks: int = 8,
        num_heads: int = 12,
        ff_dim: int = 3072,
    ) -> None:
        super().__init__()
        if not protocols or not state_schema_dims:
            raise ValueError("native action protocols and state schemas are required")
        if hidden_dim % num_heads or min(action_steps, state_history_count, num_blocks) <= 0:
            raise ValueError("invalid temporal action expert dimensions")
        self.protocol_ids = tuple(sorted(str(name) for name in protocols))
        self.protocol_to_index = {name: index for index, name in enumerate(self.protocol_ids)}
        self.adapters = nn.ModuleDict(
            {name: _TemporalProtocolAdapter(protocols[name], hidden_dim) for name in self.protocol_ids}
        )
        self.state_adapters = StateAdapterBank(
            dict(state_schema_dims), statistics=None if state_statistics is None else dict(state_statistics)
        )
        self.action_steps = int(action_steps)
        self.state_history_count = int(state_history_count)
        self.latent_shape = (int(latent_groups), int(latent_slots), int(latent_dim))
        self.hidden_dim = int(hidden_dim)
        self.context_resampler = ContextResampler(qwen_dim, hidden_dim, context_tokens, num_heads)
        self.state_projection = nn.Sequential(
            nn.Linear(64, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.state_time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.latent_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim))
        self.flow_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.protocol_embedding = nn.Embedding(len(self.protocol_ids), hidden_dim)
        self.state_position = nn.Parameter(torch.empty(state_history_count, hidden_dim))
        self.latent_group_position = nn.Parameter(torch.empty(latent_groups, hidden_dim))
        self.latent_slot_position = nn.Parameter(torch.empty(latent_slots, hidden_dim))
        self.action_position = nn.Parameter(torch.empty(action_steps, hidden_dim))
        self.condition_type = nn.Parameter(torch.empty(4, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                QKNormEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        for parameter in (
            self.state_position,
            self.latent_group_position,
            self.latent_slot_position,
            self.action_position,
            self.condition_type,
        ):
            nn.init.normal_(parameter, std=0.02)

    @property
    def protocol_dims(self) -> dict[str, int]:
        return {name: self.adapters[name].dim for name in self.protocol_ids}

    def _protocol_indices(self, protocol_ids: Sequence[str], device: torch.device) -> Tensor:
        try:
            return torch.tensor(
                [self.protocol_to_index[str(value)] for value in protocol_ids],
                device=device,
                dtype=torch.long,
            )
        except KeyError as error:
            raise ValueError(f"unknown action protocol {error.args[0]!r}") from error

    def normalize_actions(self, actions: NativeActionBatch) -> tuple[Tensor, Tensor]:
        batch, steps, padded_dim = actions.values.shape
        actions.validate(batch_size=batch, horizon=self.action_steps)
        normalized = torch.zeros(batch, steps, padded_dim, device=actions.values.device, dtype=torch.float32)
        exact_valid = torch.zeros_like(actions.valid, dtype=torch.bool)
        for row, protocol in enumerate(actions.protocol_ids):
            if protocol not in self.adapters:
                raise ValueError(f"unknown action protocol {protocol!r}")
            adapter = self.adapters[protocol]
            if int(actions.native_dims[row]) != adapter.dim:
                raise ValueError("native action dimension differs from protocol")
            if actions.normalization_spaces[row] != adapter.normalization_space:
                raise ValueError("native action normalization-space mismatch")
            valid = actions.valid[row, :, : adapter.dim].bool()
            clean = sanitize_masked_tensor(
                actions.values[row, :, : adapter.dim], valid, name="native expert actions"
            ).float()
            normalized[row, :, : adapter.dim] = torch.where(
                valid, (clean - adapter.mean) / adapter.std, torch.zeros_like(clean)
            )
            exact_valid[row, :, : adapter.dim] = valid
        return normalized, exact_valid

    def denormalize_actions(
        self,
        normalized: Tensor,
        protocol_ids: Sequence[str],
        native_dims: Tensor,
    ) -> Tensor:
        if normalized.ndim != 3 or normalized.shape[1] != self.action_steps:
            raise ValueError("normalized actions must have shape [B,action_steps,Dmax]")
        output = torch.zeros_like(normalized, dtype=torch.float32)
        for row, protocol in enumerate(protocol_ids):
            adapter = self.adapters[protocol]
            if int(native_dims[row]) != adapter.dim:
                raise ValueError("native action dimension differs from protocol")
            output[row, :, : adapter.dim] = (
                normalized[row, :, : adapter.dim].float() * adapter.std + adapter.mean
            )
        return output

    def adapt_states(
        self,
        raw_states: Tensor,
        state_dimension_valid: Tensor,
        state_schema_ids: Sequence[str],
        state_time_valid: Tensor,
    ) -> Tensor:
        if state_time_valid.shape != raw_states.shape[:2]:
            raise ValueError("state time validity must have shape [B,T]")
        effective_valid = state_dimension_valid.bool() & state_time_valid[:, :, None].bool()
        adapted = self.state_adapters(
            raw_states,
            effective_valid,
            list(state_schema_ids),
            state_present=state_time_valid.any(dim=1),
        )
        return adapted * state_time_valid[:, :, None].to(adapted.dtype)

    def velocity(
        self,
        *,
        normalized_action_samples: Tensor,
        action_valid: Tensor,
        protocol_ids: Sequence[str],
        qwen_context: Tensor,
        context_valid: Tensor,
        raw_states: Tensor,
        state_dimension_valid: Tensor,
        state_schema_ids: Sequence[str],
        state_time_valid: Tensor,
        state_times: Tensor,
        z_raw: Tensor,
        latent_group_valid: Tensor,
        sigma_a: Tensor,
    ) -> Tensor:
        batch, steps, width = normalized_action_samples.shape
        if steps != self.action_steps or action_valid.shape != normalized_action_samples.shape:
            raise ValueError("action samples/mask disagree with the fixed horizon")
        if raw_states.shape[:2] != (batch, self.state_history_count):
            raise ValueError("raw state history disagrees with configured length")
        if state_dimension_valid.shape != raw_states.shape:
            raise ValueError("state dimension mask must match raw state history")
        if state_time_valid.shape != raw_states.shape[:2] or state_times.shape != raw_states.shape[:2]:
            raise ValueError("state time tensors must have shape [B,T]")
        if z_raw.shape != (batch, *self.latent_shape):
            raise ValueError("raw latent has the wrong shape")
        if latent_group_valid.shape != (batch, self.latent_shape[0]):
            raise ValueError("latent group validity has the wrong shape")
        if sigma_a.ndim == 1:
            if sigma_a.shape != (batch,):
                raise ValueError("scalar action flow time must have shape [B]")
            sigma_steps = sigma_a[:, None].expand(-1, steps)
        elif sigma_a.shape == (batch, steps):
            sigma_steps = sigma_a
        else:
            raise ValueError("action flow time must have shape [B] or [B,action_steps]")
        if not torch.isfinite(state_times[state_time_valid.bool()]).all():
            raise ValueError("valid state times must be finite")
        clean_actions = sanitize_masked_tensor(
            normalized_action_samples, action_valid, name="normalized action samples"
        )
        latent_element_valid = latent_group_valid[:, :, None, None].expand_as(z_raw)
        clean_latent = sanitize_masked_tensor(
            z_raw, latent_element_valid, name="action expert latent"
        )
        adapted_states = self.adapt_states(
            raw_states, state_dimension_valid, state_schema_ids, state_time_valid
        )
        state_tokens = self.state_projection(adapted_states)
        safe_state_times = torch.where(state_time_valid, state_times, torch.zeros_like(state_times))
        state_tokens = state_tokens + self.state_time_projection(
            _time_embedding(safe_state_times, self.hidden_dim)
        ).reshape_as(state_tokens)
        state_tokens = state_tokens + self.state_position[None]
        context_tokens = self.context_resampler(qwen_context, context_valid)
        latent_tokens = self.latent_projection(clean_latent)
        latent_tokens = latent_tokens + self.latent_group_position[None, :, None]
        latent_tokens = latent_tokens + self.latent_slot_position[None, None]
        latent_tokens = latent_tokens.flatten(1, 2)
        protocol_indices = self._protocol_indices(protocol_ids, normalized_action_samples.device)
        protocol = self.protocol_embedding(protocol_indices)
        action_tokens = torch.zeros(
            batch, steps, self.hidden_dim,
            device=normalized_action_samples.device,
            dtype=context_tokens.dtype,
        )
        for name, adapter in self.adapters.items():
            rows = [index for index, value in enumerate(protocol_ids) if value == name]
            if rows:
                row_index = torch.tensor(rows, device=normalized_action_samples.device)
                values = clean_actions.index_select(0, row_index)[..., : adapter.dim]
                valid = action_valid.index_select(0, row_index)[..., : adapter.dim].bool()
                # Zero is a meaningful normalized action.  Concatenating the
                # element-level validity mask lets the expert distinguish a
                # real zero from a missing value that was sanitized to zero.
                features = torch.cat((values, valid.to(values.dtype)), dim=-1)
                action_tokens[row_index] = adapter.input(
                    features.to(adapter.input[0].weight.dtype)
                ).to(action_tokens.dtype)
        action_time = self.flow_time(
            _time_embedding(sigma_steps, self.hidden_dim).to(action_tokens.dtype)
        ).reshape_as(action_tokens)
        action_tokens = action_tokens + action_time + self.action_position[None] + protocol[:, None]
        context_tokens = context_tokens + self.condition_type[0] + protocol[:, None]
        state_tokens = state_tokens + self.condition_type[1] + protocol[:, None]
        latent_tokens = latent_tokens + self.condition_type[2] + protocol[:, None]
        action_tokens = action_tokens + self.condition_type[3]
        hidden = torch.cat((context_tokens, state_tokens, latent_tokens, action_tokens), dim=1)
        latent_step_valid = latent_group_valid[:, :, None].expand(
            -1, -1, self.latent_shape[1]
        ).flatten(1, 2)
        action_step_valid = action_valid.any(dim=-1)
        key_padding = torch.cat(
            (
                torch.zeros(context_tokens.shape[:2], device=hidden.device, dtype=torch.bool),
                ~state_time_valid.bool(),
                ~latent_step_valid.bool(),
                ~action_step_valid.bool(),
            ),
            dim=1,
        )
        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=key_padding)
        action_hidden = self.final_norm(hidden[:, -steps:])
        velocity = torch.zeros(batch, steps, width, device=hidden.device, dtype=action_hidden.dtype)
        for name, adapter in self.adapters.items():
            rows = [index for index, value in enumerate(protocol_ids) if value == name]
            if rows:
                row_index = torch.tensor(rows, device=hidden.device)
                velocity[row_index, :, : adapter.dim] = adapter.output(action_hidden[row_index]).to(
                    velocity.dtype
                )
        return velocity.float() * action_valid.to(dtype=torch.float32)

    forward = velocity

    @torch.no_grad()
    def sample(
        self,
        *,
        qwen_context: Tensor,
        context_valid: Tensor,
        raw_states: Tensor,
        state_dimension_valid: Tensor,
        state_schema_ids: Sequence[str],
        state_time_valid: Tensor,
        state_times: Tensor,
        z_raw: Tensor,
        latent_group_valid: Tensor,
        protocol_ids: Sequence[str],
        native_dims: Tensor,
        noise: Tensor,
        committed_prefix: NativeActionBatch | None = None,
        steps: int = 10,
    ) -> Tensor:
        if steps <= 0:
            raise ValueError("action solver steps must be positive")
        batch = noise.shape[0]
        if noise.ndim != 3 or noise.shape[1] != self.action_steps:
            raise ValueError("action noise must have shape [B,action_steps,Dmax]")
        action_valid = torch.zeros_like(noise, dtype=torch.bool)
        for row, protocol in enumerate(protocol_ids):
            dim = self.adapters[protocol].dim
            if int(native_dims[row]) != dim:
                raise ValueError("native action dimension differs from protocol")
            action_valid[row, :, :dim] = True
        prefix_valid = torch.zeros_like(action_valid)
        prefix_clean = torch.zeros_like(noise, dtype=torch.float32)
        if committed_prefix is not None:
            if committed_prefix.protocol_ids != tuple(protocol_ids):
                raise ValueError("committed prefix protocols do not match the request")
            normalized_prefix, normalized_prefix_valid = self.normalize_actions(committed_prefix)
            if normalized_prefix.shape[:2] != noise.shape[:2]:
                raise ValueError("committed prefix batch/horizon does not match action noise")
            if normalized_prefix.shape[-1] > noise.shape[-1]:
                raise ValueError("committed prefix width exceeds action-noise width")
            prefix_clean[..., : normalized_prefix.shape[-1]] = normalized_prefix
            prefix_valid[..., : normalized_prefix_valid.shape[-1]] = normalized_prefix_valid
            step_valid = prefix_valid.any(dim=-1)
            for row in range(batch):
                seen_gap = False
                for present in step_valid[row].tolist():
                    if not present:
                        seen_gap = True
                    elif seen_gap:
                        raise ValueError("committed action prefix must be left-contiguous")
                dim = int(native_dims[row])
                if (prefix_valid[row, step_valid[row], :dim].sum(dim=-1) != dim).any():
                    raise ValueError("committed prefix steps must contain every native dimension")
        value = torch.where(prefix_valid, prefix_clean, noise.float())
        schedule = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
        for index in range(steps):
            sigma = schedule[index].expand(batch, self.action_steps).clone()
            sigma[prefix_valid.any(dim=-1)] = 0
            value = torch.where(prefix_valid, prefix_clean, value)
            velocity = self.velocity(
                normalized_action_samples=value,
                action_valid=action_valid,
                protocol_ids=protocol_ids,
                qwen_context=qwen_context,
                context_valid=context_valid,
                raw_states=raw_states,
                state_dimension_valid=state_dimension_valid,
                state_schema_ids=state_schema_ids,
                state_time_valid=state_time_valid,
                state_times=state_times,
                z_raw=z_raw,
                latent_group_valid=latent_group_valid,
                sigma_a=sigma,
            )
            value = value + (schedule[index + 1] - schedule[index]) * velocity
            value = torch.where(prefix_valid, prefix_clean, value)
        return self.denormalize_actions(value, protocol_ids, native_dims)


def clone_independent_expert(reference: TemporalNativeActionExpert) -> TemporalNativeActionExpert:
    student = copy.deepcopy(reference)
    assert_independent_experts(reference, student)
    return student


def assert_independent_experts(
    reference: TemporalNativeActionExpert, student: TemporalNativeActionExpert
) -> None:
    reference_parameters = dict(reference.named_parameters())
    student_parameters = dict(student.named_parameters())
    if reference_parameters.keys() != student_parameters.keys():
        raise ValueError("teacher and student parameter structures differ")
    for name in reference_parameters:
        teacher = reference_parameters[name]
        learner = student_parameters[name]
        if teacher is learner or teacher.untyped_storage().data_ptr() == learner.untyped_storage().data_ptr():
            raise ValueError(f"teacher/student parameter storage aliases at {name}")


__all__ = [
    "ActionFlowTrainingSample",
    "TemporalNativeActionExpert",
    "assert_independent_experts",
    "clone_independent_expert",
    "committed_action_prefix_lengths",
    "make_committed_action_prefix",
    "prepare_action_flow_training",
    "sample_action_prefix_lengths",
]
