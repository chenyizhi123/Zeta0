"""Phase ownership and teacher-gradient rules for predictive-world training."""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable

import torch
from torch import Tensor, nn

from psi.data.predictive_world.objective import masked_per_sample_mse


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def module_state_sha256(module: nn.Module) -> str:
    return tensor_mapping_sha256(module.state_dict())


def tensor_mapping_sha256(tensors) -> str:
    """Hash named tensor values, including shape/dtype, in canonical key order."""
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_optimizer_partition(
    optimizer: torch.optim.Optimizer,
    *,
    trainable: Iterable[nn.Module],
    forbidden: Iterable[nn.Module] = (),
) -> None:
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for module in trainable
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    forbidden_ids = {
        id(parameter) for module in forbidden for parameter in module.parameters()
    }
    if optimizer_ids != expected_ids:
        missing = len(expected_ids - optimizer_ids)
        unexpected = len(optimizer_ids - expected_ids)
        raise ValueError(
            f"optimizer/trainable partition mismatch: missing={missing}, unexpected={unexpected}"
        )
    if optimizer_ids & forbidden_ids:
        raise ValueError("optimizer contains frozen/reference parameters")


class ReferenceModules(nn.Module):
    """The single reference coordinate published after representation training."""

    def __init__(
        self,
        encoder: nn.Module,
        readout: nn.Module,
        teacher: nn.Module,
        normalizer: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.readout = readout
        self.teacher = teacher
        self.normalizer = normalizer

    def freeze(self) -> dict[str, str]:
        freeze_module(self)
        return {
            "encoder": module_state_sha256(self.encoder),
            "readout": module_state_sha256(self.readout),
            "teacher": module_state_sha256(self.teacher),
            "normalizer": module_state_sha256(self.normalizer),
        }


class ForecastModules(nn.Module):
    def __init__(
        self,
        history_state_adapters: nn.Module,
        history_encoder: nn.Module,
        latent_transition: nn.Module,
    ) -> None:
        super().__init__()
        self.history_state_adapters = history_state_adapters
        self.history_encoder = history_encoder
        self.latent_transition = latent_transition


class PolicyModules(nn.Module):
    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student


def local_teacher_alignment(
    *,
    predicted_latent_white: Tensor,
    gold_latent_white: Tensor,
    unwhiten: Callable[[Tensor], Tensor],
    teacher_velocity: Callable[[Tensor], Tensor],
    action_valid: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compare paired latents through a frozen teacher while retaining dz gradients."""

    if predicted_latent_white.shape != gold_latent_white.shape:
        raise ValueError("predicted and gold latent coordinates disagree")
    predicted_raw = unwhiten(predicted_latent_white)
    gold_raw = unwhiten(gold_latent_white.detach())
    # Teacher parameters are frozen, but this call intentionally tracks input
    # derivatives back to predicted_raw.
    predicted_velocity = teacher_velocity(predicted_raw)
    with torch.no_grad():
        gold_velocity = teacher_velocity(gold_raw)
    per_sample, sample_valid = masked_per_sample_mse(
        predicted_velocity, gold_velocity, action_valid
    )
    if sample_valid.any():
        loss = per_sample[sample_valid].mean()
    else:
        loss = predicted_velocity.float().sum() * 0.0
    return loss, {
        "alignment_flow_mse": per_sample[sample_valid].mean().detach()
        if sample_valid.any()
        else per_sample.sum().detach() * 0,
        "alignment_valid_samples": sample_valid.sum().detach(),
    }


__all__ = [
    "ForecastModules",
    "PolicyModules",
    "ReferenceModules",
    "assert_optimizer_partition",
    "freeze_module",
    "local_teacher_alignment",
    "module_state_sha256",
]
