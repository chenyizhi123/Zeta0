"""Mask-safe per-sample losses and update-level DDP normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch import Tensor

from psi.data.predictive_world.contracts import sanitize_masked_tensor


def masked_per_sample_mse(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    element_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape or prediction.shape != valid.shape or prediction.ndim < 2:
        raise ValueError("prediction, target and valid must share shape [B,...]")
    clean_prediction = sanitize_masked_tensor(prediction, valid, name="prediction").float()
    clean_target = sanitize_masked_tensor(target, valid, name="target").float()
    mask = valid.to(device=prediction.device, dtype=torch.bool)
    weights = torch.ones_like(prediction, dtype=torch.float32)
    if element_weight is not None:
        if element_weight.shape != prediction.shape:
            raise ValueError("element weights must share prediction shape")
        weights = element_weight.to(device=prediction.device, dtype=torch.float32)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("element weights must be finite and non-negative")
    effective_weight = weights * mask.float()
    counts = effective_weight.flatten(1).sum(dim=1)
    sample_valid = counts > 0
    squared = (clean_prediction - clean_target).square()
    values = (squared * effective_weight).flatten(1).sum(dim=1) / counts.clamp_min(1)
    return torch.where(sample_valid, values, torch.zeros_like(values)), sample_valid


@dataclass(frozen=True)
class UpdateLossDenominator:
    global_weight: Tensor
    world_size: int

    @property
    def is_empty(self) -> bool:
        return self.global_weight.item() == 0


class UpdateLossNormalizer:
    """Precompute one denominator across all micros in an optimizer update."""

    def __init__(self, *, distributed: bool = True) -> None:
        self.distributed = bool(distributed)

    def denominator(
        self,
        micro_valid: Iterable[Tensor],
        micro_weights: Iterable[Tensor] | None = None,
    ) -> UpdateLossDenominator:
        valid_parts = [value.detach().bool().cpu() for value in micro_valid]
        if not valid_parts:
            raise ValueError("an optimizer update must contain at least one micro-batch")
        if micro_weights is None:
            weight_parts = [torch.ones_like(value, dtype=torch.float32) for value in valid_parts]
        else:
            weight_parts = [value.detach().float().cpu() for value in micro_weights]
            if len(weight_parts) != len(valid_parts):
                raise ValueError("micro weights must align with micro validity")
        local = torch.zeros((), dtype=torch.float64)
        for valid, weight in zip(valid_parts, weight_parts):
            if weight.shape != valid.shape or not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError("sample weights must be finite, non-negative and aligned")
            local += torch.where(valid, weight.double(), torch.zeros_like(weight, dtype=torch.float64)).sum()
        world_size = 1
        if self.distributed and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            backend = dist.get_backend()
            reduce_value = local
            if backend == "nccl":
                reduce_value = local.cuda(torch.cuda.current_device())
            dist.all_reduce(reduce_value, op=dist.ReduceOp.SUM)
            local = reduce_value.cpu()
        return UpdateLossDenominator(global_weight=local.float(), world_size=world_size)

    @staticmethod
    def scale_micro(
        per_sample_loss: Tensor,
        sample_valid: Tensor,
        denominator: UpdateLossDenominator,
        *,
        sample_weight: Tensor | None = None,
        graph_tensor: Tensor | None = None,
    ) -> Tensor:
        if per_sample_loss.shape != sample_valid.shape or per_sample_loss.ndim != 1:
            raise ValueError("per-sample loss and validity must share shape [B]")
        weights = (
            torch.ones_like(per_sample_loss, dtype=torch.float32)
            if sample_weight is None
            else sample_weight.to(device=per_sample_loss.device, dtype=torch.float32)
        )
        if weights.shape != per_sample_loss.shape:
            raise ValueError("sample weights must have shape [B]")
        numerator = torch.where(
            sample_valid.to(per_sample_loss.device), per_sample_loss.float() * weights,
            torch.zeros_like(per_sample_loss, dtype=torch.float32),
        ).sum()
        if denominator.is_empty:
            source = per_sample_loss if graph_tensor is None else graph_tensor
            return source.float().sum() * 0.0
        return numerator * denominator.world_size / denominator.global_weight.to(numerator.device)


def flow_matching_target(clean: Tensor, noise: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
    if clean.shape != noise.shape:
        raise ValueError("clean sample and noise must have identical shapes")
    if sigma.shape != (clean.shape[0],):
        raise ValueError("flow sigma must have shape [B]")
    view = (clean.shape[0],) + (1,) * (clean.ndim - 1)
    sigma_view = sigma.to(device=clean.device, dtype=clean.dtype).reshape(view)
    return (1 - sigma_view) * clean + sigma_view * noise, noise - clean


__all__ = [
    "UpdateLossDenominator",
    "UpdateLossNormalizer",
    "flow_matching_target",
    "masked_per_sample_mse",
]
