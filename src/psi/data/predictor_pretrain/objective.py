"""Stage-A batch contracts and DDP-correct globally normalized losses."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import Tensor


_REQUIRED_STAGE_A_COMMON_KEYS = frozenset(
    {
        "sample_id",
        "source_id",
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "state_history_mask",
        "state_present",
        "intent_target_vlm_state",
        "intent_target_valid",
    }
)
# Public canonical contract for callers that already provide native state64.
# The validator additionally accepts the complete heterogeneous raw-state tuple.
REQUIRED_STAGE_A_KEYS = _REQUIRED_STAGE_A_COMMON_KEYS | {"states"}
FORBIDDEN_STAGE_A_KEYS = frozenset({"action_samples", "action_history", "traj2ds"})
_ACTION_KEYS = frozenset({
    "actions", "actions_mask", "action_protocol_ids", "action_present", "action_native_dims",
})


def validate_stage_a_batch(batch: Mapping[str, Any]) -> None:
    missing = sorted(_REQUIRED_STAGE_A_COMMON_KEYS - batch.keys())
    forbidden = sorted(FORBIDDEN_STAGE_A_KEYS & batch.keys())
    has_adapted = "states" in batch
    raw_keys = {"raw_states", "state_dimension_mask", "state_schema_ids"}
    has_raw = raw_keys <= batch.keys()
    invalid_state = has_adapted == has_raw
    action_keys = _ACTION_KEYS & batch.keys()
    invalid_action = bool(action_keys) and action_keys != _ACTION_KEYS
    if missing or forbidden or invalid_state or invalid_action:
        raise ValueError(
            "invalid Stage-A batch: "
            f"missing={missing}, forbidden={forbidden}, "
            "state_input=expected exactly one of states or complete raw schema tuple, "
            f"action_input=expected none or the complete action tuple; got={sorted(action_keys)}"
        )


def globally_normalized_future_mse(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    distributed: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """MSE over globally valid samples with DDP-correct gradient scaling.

    DDP averages rank gradients, so each local numerator is multiplied by the
    world size after the global denominator is reduced.  This is not equivalent
    to averaging rank-local means when ranks have different valid counts.
    """

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share shape [B,...]")
    if valid.shape != (prediction.shape[0],):
        raise ValueError("valid must have shape [B]")
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    per_sample = (prediction.float() - target.float()).square().flatten(1).mean(1)
    local_numerator = torch.where(valid, per_sample, torch.zeros_like(per_sample)).sum()
    local_count = valid.sum().to(dtype=torch.float32)
    global_count = local_count.detach().clone()
    global_numerator = local_numerator.detach().clone()
    world_size = 1
    if distributed and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_numerator, op=dist.ReduceOp.SUM)
    if global_count.item() == 0:
        loss = prediction.float().sum() * 0.0
        mean = global_numerator * 0.0
    else:
        loss = local_numerator * world_size / global_count
        mean = global_numerator / global_count
    return loss, {
        "future_mse": mean,
        "valid_samples": global_count,
        "local_valid_samples": local_count.detach(),
    }


def globally_normalized_masked_mse(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    distributed: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """MSE averaged per valid sample, then exactly across the global DDP batch."""

    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target and valid must have identical shapes")
    if prediction.ndim < 2:
        raise ValueError("masked tensors must have shape [B,...]")
    mask = valid.to(device=prediction.device, dtype=torch.bool)
    elementwise = (prediction.float() - target.float()).square()
    counts = mask.flatten(1).sum(1)
    sample_valid = counts > 0
    per_sample = (elementwise * mask).flatten(1).sum(1) / counts.clamp_min(1)
    local_numerator = torch.where(sample_valid, per_sample, torch.zeros_like(per_sample)).sum()
    local_count = sample_valid.sum().float()
    global_numerator = local_numerator.detach().clone()
    global_count = local_count.detach().clone()
    world_size = 1
    if distributed and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(global_numerator, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    loss = (
        local_numerator * world_size / global_count
        if global_count.item() else prediction.float().sum() * 0.0
    )
    mean = global_numerator / global_count if global_count.item() else global_numerator * 0.0
    return loss, {"mse": mean, "valid_samples": global_count}


__all__ = [
    "FORBIDDEN_STAGE_A_KEYS",
    "REQUIRED_STAGE_A_KEYS",
    "globally_normalized_future_mse",
    "globally_normalized_masked_mse",
    "validate_stage_a_batch",
]
