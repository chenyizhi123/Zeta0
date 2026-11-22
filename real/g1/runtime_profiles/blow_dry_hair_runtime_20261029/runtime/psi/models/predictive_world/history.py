"""Configurable causal history encoder for predictive-world conditioning."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from psi.data.predictive_world.contracts import sanitize_masked_tensor
from psi.models.predictive_world.attention import StateConditionedAttentionBlock
from psi.models.stable_attention import QK_STABILITY, QKNormAttention


def _time_features(times: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=times.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angle = times.float().reshape(-1, 1) * frequency[None]
    value = torch.cat((angle.sin(), angle.cos()), dim=-1)
    return torch.nn.functional.pad(value, (0, dim - value.shape[-1]))


class ContextResampler(nn.Module):
    """Compress labeled causal Qwen tokens without global mean pooling."""

    def __init__(self, input_dim: int, hidden_dim: int, tokens: int, num_heads: int) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, tokens, num_heads) <= 0 or hidden_dim % num_heads:
            raise ValueError("invalid context resampler dimensions")
        self.queries = nn.Parameter(torch.empty(tokens, hidden_dim))
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attention = QKNormAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, context: Tensor, valid: Tensor) -> Tensor:
        if context.ndim != 3 or valid.shape != context.shape[:2]:
            raise ValueError("Qwen context/mask must have shape [B,L,D]/[B,L]")
        valid = valid.to(device=context.device, dtype=torch.bool)
        if (~valid).all(dim=1).any():
            raise ValueError("every context row needs at least one valid token")
        clean_context = sanitize_masked_tensor(
            context, valid[:, :, None].expand_as(context), name="Qwen context"
        )
        projected = self.input_projection(clean_context)
        query = self.queries[None].expand(context.shape[0], -1, -1)
        update = self.attention(
            self.query_norm(query), projected, projected,
            key_padding_mask=~valid, need_weights=False,
        )[0]
        return self.output_norm(query + update)


class PredictiveHistoryEncoder(nn.Module):
    """Map causal visual/task/state history to 16 reusable condition tokens.

    Defaults use the balanced predictive-world contract. Exact-shape legacy
    migration remains explicit rather than silently selecting an old scale.
    """

    def __init__(
        self,
        *,
        qwen_dim: int = 2048,
        visual_dim: int = 2048,
        state_dim: int = 64,
        state_history_count: int = 21,
        output_tokens: int = 16,
        context_tokens: int = 64,
        hidden_dim: int = 1024,
        num_blocks: int = 16,
        num_heads: int = 16,
        ff_dim: int = 4096,
    ) -> None:
        super().__init__()
        if min(state_history_count, output_tokens, context_tokens, num_blocks) <= 0:
            raise ValueError("history lengths and block count must be positive")
        if hidden_dim % num_heads:
            raise ValueError("history hidden width must be divisible by heads")
        self.state_history_count = int(state_history_count)
        self.output_tokens = int(output_tokens)
        self.hidden_dim = int(hidden_dim)
        self.context_resampler = ContextResampler(
            qwen_dim, hidden_dim, context_tokens, num_heads
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim)
        )
        self.visual_position = nn.Parameter(torch.zeros(output_tokens, hidden_dim))
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.state_time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Last eight entries have an explicit legacy physical-time migration.
        self.state_history_position = nn.Parameter(torch.zeros(state_history_count, hidden_dim))
        self.state_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.null_state = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.state_attention = QKNormAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0, fp32_math=True
        )
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                StateConditionedAttentionBlock(
                    dim=hidden_dim,
                    context_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    stability_version=QK_STABILITY,
                    state_gate_limit=1.0 / num_blocks,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.state_query, std=1 / math.sqrt(hidden_dim))

    @torch.no_grad()
    def load_legacy_state_positions(self, legacy_positions: Tensor) -> None:
        """Map legacy -0.14..0 positions to the final eight physical slots."""

        if legacy_positions.shape != (8, self.hidden_dim):
            raise ValueError("legacy state positions must have shape [8,hidden_dim]")
        self.state_history_position.zero_()
        self.state_history_position[-8:].copy_(legacy_positions)

    def forward(
        self,
        *,
        qwen_context: Tensor,
        context_valid: Tensor,
        current_visual: Tensor,
        adapted_states: Tensor,
        state_valid: Tensor,
        state_times: Tensor,
    ) -> Tensor:
        batch = qwen_context.shape[0]
        if current_visual.ndim != 3 or current_visual.shape[:2] != (batch, self.output_tokens):
            raise ValueError("current visual anchor must have shape [B,16,D]")
        if not torch.isfinite(current_visual).all():
            raise ValueError("current visual anchor must be finite")
        if adapted_states.ndim != 3 or adapted_states.shape[:2] != (
            batch, self.state_history_count
        ):
            raise ValueError("adapted state history has the wrong shape")
        if state_valid.shape != adapted_states.shape[:2] or state_times.shape != state_valid.shape:
            raise ValueError("state history validity/times disagree")
        state_valid = state_valid.to(device=adapted_states.device, dtype=torch.bool)
        if not torch.isfinite(adapted_states[state_valid]).all():
            raise ValueError("valid adapted state values must be finite")
        if not torch.isfinite(state_times[state_valid]).all() or (state_times[state_valid] > 1e-7).any():
            raise ValueError("valid history state times must be finite and causal")
        clean_states = torch.where(
            state_valid[:, :, None], adapted_states, torch.zeros_like(adapted_states)
        )
        state_tokens = self.state_projection(clean_states)
        safe_times = torch.where(state_valid, state_times, torch.zeros_like(state_times))
        state_tokens = state_tokens + self.state_time_projection(
            _time_features(safe_times, self.hidden_dim)
        ).reshape_as(state_tokens)
        state_tokens = state_tokens + self.state_history_position[None]
        null = self.null_state.expand(batch, -1, -1).to(state_tokens.dtype)
        state_tokens = torch.cat((null, state_tokens), dim=1)
        state_key_valid = torch.cat(
            (torch.ones(batch, 1, device=state_valid.device, dtype=torch.bool), state_valid), dim=1
        )
        state_query = self.state_query.expand(batch, -1, -1).to(state_tokens.dtype)
        state_summary = self.state_attention(
            self.state_norm(state_query), state_tokens, state_tokens,
            key_padding_mask=~state_key_valid, need_weights=False,
        )[0][:, 0]
        state_present = state_valid.any(dim=1)
        state_summary = state_summary * state_present[:, None].to(state_summary.dtype)

        context = self.context_resampler(qwen_context, context_valid)
        queries = self.visual_projection(current_visual) + self.visual_position[None]
        context_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        for block in self.blocks:
            queries = block(
                queries, context, state_summary, context_mask, state_present=state_present
            )
        return self.output_norm(queries)


__all__ = ["ContextResampler", "PredictiveHistoryEncoder"]
