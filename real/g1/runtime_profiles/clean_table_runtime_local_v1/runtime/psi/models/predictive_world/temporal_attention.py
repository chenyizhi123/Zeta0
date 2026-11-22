"""One additive-mask convention for predictive-world attention."""

from __future__ import annotations

import torch
from torch import Tensor


def additive_attention_bias(visible: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Convert True=visible into 0=visible/-inf=masked additive bias."""

    if visible.dtype != torch.bool:
        raise ValueError("attention visibility must be boolean")
    return torch.zeros(visible.shape, device=visible.device, dtype=dtype).masked_fill(
        ~visible, -torch.inf
    )


def prefix_visibility(query_groups: Tensor, key_groups: Tensor) -> Tensor:
    """A query group may read only its own or earlier temporal groups."""

    if query_groups.ndim != 1 or key_groups.ndim != 1:
        raise ValueError("temporal group IDs must be 1-D")
    return key_groups[None, :] <= query_groups[:, None]


def build_temporal_attention_bias(
    query_groups: Tensor,
    key_groups: Tensor,
    *,
    batch_size: int,
    num_heads: int,
    dtype: torch.dtype,
    query_valid: Tensor | None = None,
    key_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build an MHA-compatible additive bias and valid-query mask.

    A sentinel key should be prepended by the caller when a query can have no
    real keys. This function makes such rows finite only when at least one key
    is visible; it never relies on backend-specific all-masked behavior.
    """

    if batch_size <= 0 or num_heads <= 0:
        raise ValueError("batch size and head count must be positive")
    base = prefix_visibility(query_groups, key_groups)
    q_count, k_count = base.shape
    visible = base[None].expand(batch_size, -1, -1).clone()
    if key_valid is not None:
        if key_valid.shape != (batch_size, k_count):
            raise ValueError("key_valid must have shape [B,K]")
        visible &= key_valid[:, None, :].bool()
    if query_valid is None:
        valid_queries = torch.ones(batch_size, q_count, device=visible.device, dtype=torch.bool)
    else:
        if query_valid.shape != (batch_size, q_count):
            raise ValueError("query_valid must have shape [B,Q]")
        valid_queries = query_valid.bool()
    if (valid_queries & ~visible.any(dim=-1)).any():
        raise ValueError("valid attention query has no visible key; add a sentinel key")
    visible |= ~valid_queries[:, :, None]
    bias = additive_attention_bias(visible, dtype=dtype)
    bias = bias[:, None].expand(-1, num_heads, -1, -1).reshape(
        batch_size * num_heads, q_count, k_count
    )
    return bias, valid_queries


def action_time_visibility(
    query_group_ends: Tensor,
    action_times: Tensor,
    action_step_valid: Tensor,
    *,
    include_sentinel: bool = True,
) -> Tensor:
    """Return [B,Q,T(+1)] with strict action_time < group_end visibility."""

    if query_group_ends.ndim != 1 or action_times.ndim != 2:
        raise ValueError("group ends must be [Q] and action times [B,T]")
    if action_step_valid.shape != action_times.shape:
        raise ValueError("action validity must match action times")
    if not torch.isfinite(action_times[action_step_valid.bool()]).all():
        raise ValueError("valid action times must be finite")
    visible = action_step_valid[:, None, :].bool() & (
        action_times[:, None, :] < query_group_ends[None, :, None]
    )
    if include_sentinel:
        sentinel = torch.ones(
            visible.shape[0], visible.shape[1], 1, device=visible.device, dtype=torch.bool
        )
        visible = torch.cat((sentinel, visible), dim=-1)
    return visible


__all__ = [
    "action_time_visibility",
    "additive_attention_bias",
    "build_temporal_attention_bias",
    "prefix_visibility",
]
