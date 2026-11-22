"""State-conditioned attention shared by the causal history encoder's blocks.

Extracted from the retired endpoint model. Tensor names, initialization, masking
and arithmetic are unchanged so existing predictive-world checkpoints still load.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from psi.models.stable_attention import LEGACY_STABILITY, QK_STABILITY, QKNormAttention


class StateConditionedAttentionBlock(nn.Module):
    """One pre-norm query self/cross-attention block."""

    def __init__(
        self,
        *,
        dim: int = 2048,
        context_dim: int | None = None,
        qwen_dim: int | None = None,
        num_heads: int = 16,
        ff_dim: int = 4096,
        stability_version: str = LEGACY_STABILITY,
        state_gate_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if context_dim is None:
            context_dim = qwen_dim
        if context_dim is None:
            raise TypeError("context_dim (or qwen_dim) is required")
        if dim % num_heads:
            raise ValueError("intent width must be divisible by the attention heads")
        self.query_norm = nn.LayerNorm(dim, eps=1e-6)
        attention = QKNormAttention if stability_version == QK_STABILITY else nn.MultiheadAttention
        self.self_attn = attention(
            dim, num_heads, dropout=0.0, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = attention(
            dim, num_heads, dropout=0.0, batch_first=True
        )
        self.state_norm = nn.LayerNorm(dim, eps=1e-6)
        self.state_gate = nn.Parameter(torch.ones(dim))
        self.stability_version = stability_version
        self.state_gate_limit = state_gate_limit
        if stability_version == QK_STABILITY:
            # Each of N blocks injects the same summary. Effective initial
            # gate is 0.1/N, bounded by 1/N even if the raw parameter grows.
            nn.init.constant_(self.state_gate, math.atanh(0.1))
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_in = nn.Linear(dim, ff_dim)
        self.ff_out = nn.Linear(ff_dim, dim)
        self.context_dim = int(context_dim)

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
        state_embedding: Tensor,
        context_mask: Tensor,
        state_present: Tensor | None = None,
    ) -> Tensor:
        q = self.query_norm(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]

        q = self.cross_norm(queries)
        # MultiheadAttention uses True for *padding* in key_padding_mask; the
        # public ABI uses True for valid tokens, so invert exactly once here.
        cross = self.cross_attn(
            q,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )[0]
        queries = queries + cross

        gate = self.state_gate
        if self.stability_version == QK_STABILITY:
            gate = gate.tanh() * self.state_gate_limit
        state_update = gate.view(1, 1, -1) * self.state_norm(
            state_embedding
        ).unsqueeze(1)
        if state_present is not None:
            state_update = state_update * state_present[:, None, None].to(
                device=state_update.device, dtype=state_update.dtype
            )
        queries = queries + state_update

        ff = self.ff_in(self.ff_norm(queries))
        ff = torch.nn.functional.gelu(ff, approximate="tanh")
        queries = queries + self.ff_out(ff)
        return queries
