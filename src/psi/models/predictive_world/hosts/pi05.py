"""Reserved layered VLA host contract for future pi0.5 (``pi05``) integration.

Design contract (zero-surgery extension over Zeta-0's shared System 1.5 core):
1. System 2 (PaliGemma 2B):
   - SigLIP visual patch tokens populate ``ObservationFeatures.current_visual``
     and supervise future scene latents via ``FutureTrajectoryEncoder(visual_dim=...)``
     and ``WorldReadout(visual_dim=...)``.
   - PaliGemma layer-wise key/value states populate ``ObservationFeatures.kv_cache``
     and ``kv_attention_mask`` once per step so both teacher ``T`` and student ``G``
     share one System 2 forward pass during dual-response transfer.
2. System 1 (Gemma-300M Flow-Matching Action Expert):
   - Projects causal future latents ``z_raw [B, K, N, d]`` into expert-side
     future conditioning tokens with block-bidirectional attention over
     ``[z_tokens, state_token, noisy_action_tokens]`` and read-only cross/causal
     attention to the frozen PaliGemma ``kv_cache``.
   - Uses Beta(1.5, 1.0) flow-time sampling with matching Zeta velocity sign
     (``x_t = t * eps + (1 - t) * a``, ``v* = eps - a``) and RTC clean prefix
     inpainting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from psi.models.predictive_world.hosts.base import BackboneLoadResult


class Pi05PlannedHost:
    """Contract placeholder for upcoming pi0.5 (PaliGemma + Gemma Action Expert) host."""

    name = "pi05"
    status = "planned"
    backbone_family = "paligemma_siglip"
    supports_kv_cache = True
    requires_image_mask = False
    default_action_horizon = 50
    default_solver_steps = 10

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "backbone_family": self.backbone_family,
            "supports_kv_cache": self.supports_kv_cache,
            "requires_image_mask": self.requires_image_mask,
            "default_action_horizon": self.default_action_horizon,
            "default_solver_steps": self.default_solver_steps,
            "action_expert_class": "Pi05ActionExpert (planned)",
        }

    def _not_implemented(self, stage: str) -> None:
        raise NotImplementedError(
            f"Layered host 'pi05' is reserved for v0.3 ({stage} not yet implemented). "
            "Use host_architecture='zeta_native' or 'gr00t_n1d7' today, or follow "
            "src/psi/models/predictive_world/hosts/pi05.py to bind a PaliGemma + Gemma "
            "action expert onto Zeta-0's ObservationFeatures(kv_cache=...) contract."
        )

    def load_backbone(
        self,
        *,
        backbone_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        host_model_path: Path | None = None,
        expected_config_sha256: str | None = None,
        observation_encoder_cls: Any | None = None,
    ) -> BackboneLoadResult:
        del backbone_path, device, dtype, host_model_path, expected_config_sha256, observation_encoder_cls
        self._not_implemented("load_backbone")
        raise AssertionError("unreachable")

    def build_action_expert(
        self,
        *,
        config: Any,
        protocols: Mapping[str, Mapping[str, Any]],
        state_dims: Mapping[str, int],
        state_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        context_dim: int,
    ) -> nn.Module:
        del config, protocols, state_dims, state_stats, context_dim
        self._not_implemented("build_action_expert")
        raise AssertionError("unreachable")

    def build_deploy_action_expert(
        self,
        *,
        manifest: Mapping[str, Any],
        protocols: Mapping[str, Mapping[str, Any]],
        state_dims: Mapping[str, int],
        state_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        context_dim: int,
        host_model_path: Path | None = None,
    ) -> nn.Module:
        del manifest, protocols, state_dims, state_stats, context_dim, host_model_path
        self._not_implemented("build_deploy_action_expert")
        raise AssertionError("unreachable")
