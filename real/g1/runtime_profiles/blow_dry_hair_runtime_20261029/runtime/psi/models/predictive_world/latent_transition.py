"""Parallel direct prediction of action-useful future process latents."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from psi.data.predictive_world.contracts import NativeActionBatch, sanitize_masked_tensor
from psi.models.predictive_world.temporal_attention import (
    additive_attention_bias,
    prefix_visibility,
)
from psi.models.stable_attention import QKNormAttention


class LatentNormalizer(nn.Module):
    """Immutable-by-contract coordinate transform between raw E and regression space."""

    def __init__(self, latent_shape: tuple[int, int, int], *, std_floor: float = 0.05) -> None:
        super().__init__()
        if len(latent_shape) != 3 or min(latent_shape) <= 0 or std_floor <= 0:
            raise ValueError("latent shape and std floor must be positive")
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.std_floor = float(std_floor)
        self.register_buffer("mean", torch.zeros(self.latent_shape), persistent=True)
        self.register_buffer("std", torch.ones(self.latent_shape), persistent=True)
        self.register_buffer("fitted", torch.tensor(False), persistent=True)

    @torch.no_grad()
    def fit(
        self,
        latent: Tensor,
        group_valid: Tensor,
        *,
        distributed: bool = False,
    ) -> dict[str, float]:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError("latent samples disagree with the configured shape")
        if group_valid.shape != latent.shape[:2]:
            raise ValueError("group_valid must have shape [B,G]")
        mask = group_valid[:, :, None, None].expand_as(latent).bool()
        if not torch.isfinite(latent[mask]).all():
            raise ValueError("valid latent samples must be finite")
        count = mask.sum(dim=0).double()
        clean = torch.where(mask, latent.double(), torch.zeros_like(latent, dtype=torch.float64))
        total = clean.sum(dim=0)
        squared_total = clean.square().sum(dim=0)
        if distributed:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed latent fitting requires an initialized process group")
            for value in (count, total, squared_total):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if (count == 0).any():
            raise ValueError("every latent position/channel needs calibration samples")
        mean = total / count
        variance = (squared_total / count - mean.square()).clamp_min(0)
        raw_std = variance.sqrt().float()
        clipped = raw_std < self.std_floor
        self.mean.copy_(mean.float())
        self.std.copy_(raw_std.clamp_min(self.std_floor))
        self.fitted.fill_(True)
        return {"std_floor_fraction": float(clipped.float().mean().item())}

    @torch.no_grad()
    def update_ema(
        self,
        latent: Tensor,
        group_valid: Tensor,
        *,
        decay: float = 0.999,
        distributed: bool = False,
    ) -> dict[str, float]:
        """Track stable world-model coordinates without backpropagating through statistics."""

        if not 0 <= decay < 1:
            raise ValueError("latent EMA decay must be in [0,1)")
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError("latent samples disagree with the configured shape")
        if group_valid.shape != latent.shape[:2]:
            raise ValueError("group_valid must have shape [B,G]")
        mask = group_valid[:, :, None, None].expand_as(latent).bool()
        if not torch.isfinite(latent[mask]).all():
            raise ValueError("valid latent samples must be finite")
        count = mask.sum(dim=0).float()
        clean = torch.where(mask, latent.float(), torch.zeros_like(latent, dtype=torch.float32))
        total = clean.sum(dim=0)
        squared_total = clean.square().sum(dim=0)
        if distributed:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed latent EMA requires an initialized process group")
            for value in (count, total, squared_total):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        observed = count > 0
        safe_count = count.clamp_min(1)
        batch_mean = total / safe_count
        batch_std = (squared_total / safe_count - batch_mean.square()).clamp_min(0).sqrt()
        batch_std = batch_std.clamp_min(self.std_floor)
        if not bool(self.fitted):
            self.mean.copy_(torch.where(observed, batch_mean, self.mean))
            self.std.copy_(torch.where(observed, batch_std, self.std))
            self.fitted.fill_(True)
        else:
            self.mean.copy_(torch.where(
                observed, self.mean * decay + batch_mean * (1 - decay), self.mean
            ))
            self.std.copy_(torch.where(
                observed, self.std * decay + batch_std * (1 - decay), self.std
            ))
        return {"latent_ema_std_floor_fraction": float(
            (observed & (batch_std <= self.std_floor)).float().sum()
            / observed.float().sum().clamp_min(1)
        )}

    def whiten(self, latent_raw: Tensor) -> Tensor:
        self._validate(latent_raw)
        return (latent_raw.float() - self.mean) / self.std

    def unwhiten(self, latent_white: Tensor) -> Tensor:
        self._validate(latent_white)
        return latent_white.float() * self.std + self.mean

    def _validate(self, latent: Tensor) -> None:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError("latent tensor disagrees with the configured shape")
        if not bool(self.fitted):
            raise RuntimeError("latent normalizer has not been fitted/published")


class _NativeActionConditionAdapter(nn.Module):
    def __init__(self, spec: Mapping[str, Any], hidden_dim: int) -> None:
        super().__init__()
        self.dim = int(spec["dim"])
        mean = torch.as_tensor(spec["mean"], dtype=torch.float32)
        std = torch.as_tensor(spec["std"], dtype=torch.float32)
        if mean.shape != (self.dim,) or std.shape != (self.dim,):
            raise ValueError("action protocol statistics disagree with its dimension")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("action protocol statistics must be finite with positive std")
        self.expected_space = str(spec.get("normalization_space", "native"))
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.projection = nn.Sequential(
            nn.Linear(2 * self.dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, values: Tensor, valid: Tensor, *, normalization_space: str) -> Tensor:
        if values.shape != valid.shape or values.shape[-1] != self.dim:
            raise ValueError("native action condition shape mismatch")
        if normalization_space != self.expected_space:
            raise ValueError(
                f"action normalization space {normalization_space!r} does not match "
                f"protocol space {self.expected_space!r}"
            )
        clean = sanitize_masked_tensor(values, valid, name="latent-transition action condition").float()
        normalized = (clean - self.mean) / self.std
        normalized = torch.where(valid.bool(), normalized, torch.zeros_like(normalized))
        features = torch.cat((normalized, valid.float()), dim=-1)
        return self.projection(features.to(self.projection[0].weight.dtype))


class _TransitionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = QKNormAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_attn = QKNormAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.action_attn = QKNormAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.GELU(approximate="tanh"), nn.Linear(ff_dim, hidden_dim)
        )

    def forward(
        self,
        hidden: Tensor,
        context: Tensor,
        context_valid: Tensor,
        action_tokens: Tensor,
        self_bias: Tensor | None,
        action_bias: Tensor,
    ) -> Tensor:
        query = self.self_norm(hidden)
        hidden = hidden + self.self_attn(query, query, query, attn_mask=self_bias, need_weights=False)[0]
        query = self.context_norm(hidden)
        hidden = hidden + self.context_attn(
            query, context, context, key_padding_mask=~context_valid, need_weights=False
        )[0]
        query = self.action_norm(hidden)
        hidden = hidden + self.action_attn(
            query, action_tokens, action_tokens, attn_mask=action_bias, need_weights=False
        )[0]
        return hidden + self.ff(self.ff_norm(hidden))


class TemporalLatentTransition(nn.Module):
    """Parallel temporal queries with prefix-causal process-latent readouts.

    All segments share a single transformer invocation, not a recurrent rollout. Outputs are
    directly regressed in normalized E coordinates. No target, noise, solver or
    future-label validity is accepted on the predictive path.
    """

    MODES = ("prior", "world")
    SEMANTICS = "parallel_process"

    def __init__(
        self,
        protocols: Mapping[str, Mapping[str, Any]],
        *,
        context_dim: int = 1024,
        latent_shape: tuple[int, int, int] = (3, 8, 256),
        group_ends_s: Sequence[float] = (0.2, 0.4, 0.6),
        hidden_dim: int = 768,
        num_blocks: int = 8,
        num_heads: int = 12,
        ff_dim: int = 3072,
    ) -> None:
        super().__init__()
        groups, slots, latent_dim = latent_shape
        if groups != len(group_ends_s) or min(latent_shape) <= 0:
            raise ValueError("latent shape must agree with future group times")
        if min(hidden_dim, num_heads, num_blocks, ff_dim) <= 0 or hidden_dim % num_heads:
            raise ValueError("invalid transition transformer dimensions")
        if not protocols:
            raise ValueError("at least one native action protocol is required")
        ends = torch.as_tensor(group_ends_s, dtype=torch.float32)
        if not torch.isfinite(ends).all() or (ends <= 0).any() or (ends[1:] <= ends[:-1]).any():
            raise ValueError("future group ends must be finite and strictly increasing")
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.protocol_ids = tuple(sorted(str(name) for name in protocols))
        self.protocol_to_index = {name: index for index, name in enumerate(self.protocol_ids)}
        self.action_adapters = nn.ModuleDict({
            name: _NativeActionConditionAdapter(protocols[name], hidden_dim)
            for name in self.protocol_ids
        })
        self.register_buffer("group_ends_s", ends)
        self.context_projection = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim))
        self.initial_slots = nn.Parameter(torch.empty(slots, hidden_dim))
        self.initial_attention = QKNormAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.segment_time = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.action_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.mode_embedding = nn.Parameter(torch.empty(2, hidden_dim))
        self.protocol_embedding = nn.Embedding(len(self.protocol_ids), hidden_dim)
        self.null_action = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.blocks = nn.ModuleList([
            _TransitionBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_blocks)
        ])
        self.process_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))
        for parameter in (self.initial_slots, self.mode_embedding, self.null_action):
            nn.init.normal_(parameter, std=0.02)
        # E targets are centered; begin with the mean, then learn deviations.
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)

    @staticmethod
    def _time_embedding(value: Tensor, dim: int) -> Tensor:
        half = dim // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=value.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = value.float().reshape(-1, 1) * frequency[None]
        embedded = torch.cat((angle.sin(), angle.cos()), dim=-1)
        return torch.nn.functional.pad(embedded, (0, dim - embedded.shape[-1]))

    def _protocol_indices(self, protocol_ids: Sequence[str], device: torch.device) -> Tensor:
        try:
            return torch.tensor(
                [self.protocol_to_index[str(value)] for value in protocol_ids],
                device=device,
                dtype=torch.long,
            )
        except KeyError as error:
            raise ValueError(f"unknown action protocol {error.args[0]!r}") from error

    def _encode_actions(
        self,
        action_condition: NativeActionBatch | None,
        *,
        batch: int,
        mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if mode == "world" and action_condition is None:
            raise ValueError("world mode requires an explicit native action condition")
        if action_condition is None:
            tokens = self.null_action.to(device=device, dtype=dtype).expand(batch, -1, -1)
            times = torch.empty(batch, 0, device=device, dtype=torch.float32)
            valid = torch.empty(batch, 0, device=device, dtype=torch.bool)
            return tokens, times, valid
        if action_condition.values.shape[0] != batch:
            raise ValueError("action condition batch size mismatch")
        action_condition.validate(batch_size=batch, horizon=action_condition.values.shape[1])
        protocol_indices = self._protocol_indices(action_condition.protocol_ids, device)
        steps = action_condition.values.shape[1]
        encoded = torch.zeros(batch, steps, self.hidden_dim, device=device, dtype=dtype)
        for name, adapter in self.action_adapters.items():
            rows = [index for index, value in enumerate(action_condition.protocol_ids) if value == name]
            if not rows:
                continue
            row_index = torch.tensor(rows, device=device)
            width = adapter.dim
            if (action_condition.native_dims[row_index].to(device) != width).any():
                raise ValueError("native action dimension does not match protocol")
            row_values = action_condition.values.to(device).index_select(0, row_index)[..., :width]
            row_valid = action_condition.valid.to(device).index_select(0, row_index)[..., :width]
            for local_row, source_row in enumerate(rows):
                projected = adapter(
                    row_values[local_row : local_row + 1],
                    row_valid[local_row : local_row + 1],
                    normalization_space=action_condition.normalization_spaces[source_row],
                )
                encoded[source_row] = projected[0].to(dtype)
        action_times = action_condition.times.to(device=device, dtype=torch.float32)
        action_step_valid = action_condition.valid.to(device=device).any(dim=-1)
        safe_action_times = torch.where(
            action_step_valid, action_times, torch.zeros_like(action_times)
        )
        encoded = encoded + self.action_time(
            self._time_embedding(safe_action_times, self.hidden_dim).to(dtype)
        ).reshape_as(encoded)
        encoded = encoded + self.protocol_embedding(protocol_indices)[:, None].to(dtype)
        null = self.null_action.to(device=device, dtype=dtype).expand(batch, -1, -1)
        return (
            torch.cat((null, encoded), dim=1),
            action_times,
            action_step_valid,
        )

    def forward(
        self,
        context: Tensor,
        *,
        mode: str,
        action_condition: NativeActionBatch | None = None,
        context_valid: Tensor | None = None,
    ) -> Tensor:
        if mode not in self.MODES:
            raise ValueError(f"unsupported transition mode {mode!r}")
        if context.ndim != 3:
            raise ValueError("history context must have shape [B,N,C]")
        batch = context.shape[0]
        if context_valid is None:
            context_valid = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        if context_valid.shape != context.shape[:2]:
            raise ValueError("context_valid must have shape [B,N]")
        context_valid = context_valid.to(device=context.device, dtype=torch.bool)
        if (~context_valid).all(dim=1).any():
            raise ValueError("each history context row needs a valid token")
        clean_context = sanitize_masked_tensor(
            context, context_valid[:, :, None].expand_as(context), name="history condition"
        )
        context_tokens = self.context_projection(clean_context)
        action_tokens, action_times, action_valid = self._encode_actions(
            action_condition, batch=batch, mode=mode,
            device=context.device, dtype=context_tokens.dtype,
        )
        slots = self.initial_slots[None].expand(batch, -1, -1).to(context_tokens.dtype)
        memory = slots + self.initial_attention(
            slots, context_tokens, context_tokens, key_padding_mask=~context_valid, need_weights=False
        )[0]
        groups, slot_count, _ = self.latent_shape
        lower = torch.cat((self.group_ends_s.new_zeros(1), self.group_ends_s[:-1]))
        times = torch.cat((
            self._time_embedding(self.group_ends_s, self.hidden_dim),
            self._time_embedding(self.group_ends_s - lower, self.hidden_dim),
        ), dim=-1).to(context_tokens.dtype)
        hidden = memory[:, None] + self.segment_time(times)[None, :, None]
        hidden = hidden + self.mode_embedding[self.MODES.index(mode)].to(hidden.dtype)
        hidden = hidden.reshape(batch, groups * slot_count, self.hidden_dim)
        group_ids = torch.arange(groups, device=context.device).repeat_interleave(slot_count)
        # Prefix mask is needed at EVERY layer: later query states contain later
        # actions and must never relay them back into earlier predictions.
        self_visible = prefix_visibility(group_ids, group_ids)
        self_bias = additive_attention_bias(self_visible, dtype=hidden.dtype)
        ends = self.group_ends_s.repeat_interleave(slot_count)
        visible = (
            action_valid[:, None, :]
            & (action_times[:, None, :] >= 0)
            & (action_times[:, None, :] < ends[None, :, None])
        )
        visible = torch.cat((
            torch.ones(batch, groups * slot_count, 1, device=context.device, dtype=torch.bool),
            visible,
        ), dim=-1)
        bias = additive_attention_bias(visible, dtype=hidden.dtype)
        bias = bias[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            batch * self.num_heads, groups * slot_count, action_tokens.shape[1]
        )
        for block in self.blocks:
            hidden = block(hidden, context_tokens, context_valid, action_tokens, self_bias, bias)
        output = self.output_projection(self.process_norm(hidden)).float()
        latent = output.reshape(batch, groups, slot_count, -1)
        return latent


__all__ = ["LatentNormalizer", "TemporalLatentTransition"]
