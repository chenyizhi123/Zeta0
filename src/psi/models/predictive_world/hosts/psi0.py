"""Psi0-derived (``zeta_native``) layered VLA host."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from psi.models.predictive_world.hosts.base import BackboneLoadResult


class Psi0NativeHost:
    """Frozen Qwen3-VL System 2 + TemporalNativeActionExpert System 1 host."""

    name = "zeta_native"
    status = "supported"
    backbone_family = "qwen3_vl"
    supports_kv_cache = False
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
            "action_expert_class": "TemporalNativeActionExpert",
        }

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
        del host_model_path, expected_config_sha256
        from transformers import Qwen3VLForConditionalGeneration

        from psi.models.predictive_world.observation import FrozenQwenObservationEncoder

        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            str(backbone_path),
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        context_dim = int(getattr(qwen.config, "text_config", qwen.config).hidden_size)
        vision_cfg = getattr(qwen.config, "vision_config", None)
        visual_dim = int(getattr(vision_cfg, "out_hidden_size", 2048))
        encoder_cls = FrozenQwenObservationEncoder if observation_encoder_cls is None else observation_encoder_cls
        encoder = encoder_cls(qwen)
        return BackboneLoadResult(
            observation_encoder=encoder,
            context_dim=context_dim,
            visual_dim=visual_dim,
            initialization=None,
        )

    def build_action_expert(
        self,
        *,
        config: Any,
        protocols: Mapping[str, Mapping[str, Any]],
        state_dims: Mapping[str, int],
        state_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        context_dim: int,
    ) -> nn.Module:
        from psi.models.predictive_world.action_expert import TemporalNativeActionExpert

        model = config.model
        return TemporalNativeActionExpert(
            protocols,
            state_dims,
            state_statistics=state_stats,
            qwen_dim=context_dim,
            context_tokens=model.expert_context_tokens,
            latent_dim=model.latent_shape[2],
            latent_groups=model.latent_shape[0],
            latent_slots=model.latent_shape[1],
            state_history_count=config.data.state_history_count,
            action_steps=config.data.action_horizon,
            hidden_dim=model.expert_hidden_dim,
            num_blocks=model.expert_blocks,
            num_heads=model.expert_heads,
            ff_dim=model.expert_ff_dim,
        )

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
        del host_model_path
        from psi.models.predictive_world.action_expert import TemporalNativeActionExpert

        arch = manifest["architecture"]
        temporal = manifest["temporal_abi"]
        return TemporalNativeActionExpert(
            protocols,
            state_dims,
            state_statistics=state_stats,
            qwen_dim=context_dim,
            context_tokens=arch["expert_context_tokens"],
            latent_dim=arch["latent_shape"][2],
            latent_groups=arch["latent_shape"][0],
            latent_slots=arch["latent_shape"][1],
            state_history_count=temporal["state_history_count"],
            action_steps=temporal["action_horizon"],
            hidden_dim=arch["expert_hidden_dim"],
            num_blocks=arch["expert_blocks"],
            num_heads=arch["expert_heads"],
            ff_dim=arch["expert_ff_dim"],
        )
