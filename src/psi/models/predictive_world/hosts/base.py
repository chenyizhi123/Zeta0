"""Layered / dual-system VLA host contracts for Zeta-0.

Zeta-0 sits as a foresight bridge (System 1.5) between a System 2
vision-language backbone and a System 1 flow-matching action expert:
- ``zeta_native``: Frozen Qwen3-VL + TemporalNativeActionExpert (Psi0-derived)
- ``gr00t_n1d7``: Exported GR00T N1.7 VLM + native Gr00tN1d7ActionHead
- ``pi05`` (reserved contract): PaliGemma KV-cache + SigLIP patches + Gemma action expert
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import nn


@dataclass(frozen=True)
class BackboneLoadResult:
    """Loaded System 2 observation encoder and feature geometry."""

    observation_encoder: nn.Module
    context_dim: int
    visual_dim: int
    initialization: dict[str, Any] | None = None


@runtime_checkable
class LayeredVLAHostSpec(Protocol):
    """Unified training, export, and deployment contract for a layered VLA host."""

    name: str
    status: str  # "supported" | "planned"
    backbone_family: str
    supports_kv_cache: bool
    requires_image_mask: bool
    default_action_horizon: int
    default_solver_steps: int

    def metadata(self) -> dict[str, Any]: ...

    def load_backbone(
        self,
        *,
        backbone_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        host_model_path: Path | None = None,
        expected_config_sha256: str | None = None,
        observation_encoder_cls: Any | None = None,
    ) -> BackboneLoadResult: ...

    def build_action_expert(
        self,
        *,
        config: Any,
        protocols: Mapping[str, Mapping[str, Any]],
        state_dims: Mapping[str, int],
        state_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        context_dim: int,
    ) -> nn.Module: ...

    def build_deploy_action_expert(
        self,
        *,
        manifest: Mapping[str, Any],
        protocols: Mapping[str, Mapping[str, Any]],
        state_dims: Mapping[str, int],
        state_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        context_dim: int,
        host_model_path: Path | None = None,
    ) -> nn.Module: ...


_REGISTRY: dict[str, LayeredVLAHostSpec] = {}


def register_host(spec: LayeredVLAHostSpec, *, overwrite: bool = False) -> LayeredVLAHostSpec:
    """Register a layered VLA host implementation by its architecture name."""
    name = str(spec.name).strip()
    if not name:
        raise ValueError("host name must be non-empty")
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"layered VLA host is already registered: {name!r}")
    _REGISTRY[name] = spec
    return spec


def get_host(name: str) -> LayeredVLAHostSpec:
    """Resolve a registered layered VLA host by ``model.host_architecture``."""
    key = str(name or "zeta_native")
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unsupported layered VLA host {key!r}; registered hosts: {available}")
    return _REGISTRY[key]


def list_hosts(*, include_planned: bool = True) -> tuple[str, ...]:
    """Return registered host names in deterministic order."""
    return tuple(
        name
        for name in sorted(_REGISTRY)
        if include_planned or _REGISTRY[name].status == "supported"
    )


def describe_hosts(*, include_planned: bool = True) -> dict[str, dict[str, Any]]:
    """Return machine-readable metadata for registered layered VLA hosts."""
    return {
        name: _REGISTRY[name].metadata()
        for name in list_hosts(include_planned=include_planned)
    }
