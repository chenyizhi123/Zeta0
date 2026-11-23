"""Native GR00T N1.7 (``gr00t_n1d7``) layered VLA host."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from psi.data.predictive_world.cache import file_sha256
from psi.models.predictive_world.hosts.base import BackboneLoadResult


class Gr00tN1d7Host:
    """Exported GR00T N1.7 VLM System 2 + native Gr00tN1d7ActionHead System 1 host."""

    name = "gr00t_n1d7"
    status = "supported"
    backbone_family = "gr00t_n1d7_vl"
    supports_kv_cache = False
    requires_image_mask = True
    default_action_horizon = 40
    default_solver_steps = 4

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "backbone_family": self.backbone_family,
            "supports_kv_cache": self.supports_kv_cache,
            "requires_image_mask": self.requires_image_mask,
            "default_action_horizon": self.default_action_horizon,
            "default_solver_steps": self.default_solver_steps,
            "action_expert_class": "Gr00tN1d7ActionExpert",
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
        from psi.models.predictive_world.gr00t_assets import load_backbone, verify_backbone_export
        from psi.models.predictive_world.observation import FrozenQwenObservationEncoder

        initialization = None
        if host_model_path is not None and expected_config_sha256 is not None:
            initialization = verify_backbone_export(
                backbone_path,
                host_model_path,
                expected_config_sha256=expected_config_sha256,
            )
        qwen = load_backbone(backbone_path, dtype=dtype, device=device)
        context_dim = int(getattr(qwen.config, "text_config", qwen.config).hidden_size)
        vision_cfg = getattr(qwen.config, "vision_config", None)
        visual_dim = int(getattr(vision_cfg, "out_hidden_size", 2048))
        encoder_cls = FrozenQwenObservationEncoder if observation_encoder_cls is None else observation_encoder_cls
        encoder = encoder_cls(qwen, include_image_mask=True)
        return BackboneLoadResult(
            observation_encoder=encoder,
            context_dim=context_dim,
            visual_dim=visual_dim,
            initialization=initialization,
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
        from psi.data.predictive_world.assets import resolve_asset_path
        from psi.models.predictive_world.gr00t import load_action_expert

        model = config.model
        path = resolve_asset_path(model.gr00t_model, config.runtime.asset_path_map)
        if file_sha256(path / "config.json") != model.gr00t_config_sha256:
            raise ValueError("GR00T base configuration identity mismatch")
        expert = load_action_expert(
            path,
            protocols,
            state_dims,
            state_stats,
            latent_shape=model.latent_shape,
        )
        expert.activation_checkpointing = config.runtime.gr00t_activation_checkpointing
        if (
            expert.action_steps != config.data.action_horizon
            or expert.head.config.backbone_embedding_dim != context_dim
        ):
            raise ValueError(
                "GR00T checkpoint dimensions disagree with the resolved training configuration"
            )
        return expert

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
        """Build an uninitialized native N1.7 student graph ready for ``load_inference_weights``."""
        from psi.models.predictive_world.gr00t import Gr00tN1d7ActionExpert, native_types

        arch = manifest["architecture"]
        temporal = manifest["temporal_abi"]
        if host_model_path is None:
            raise ValueError(
                "GR00T N1.7 deployment requires --gr00t-model (local base directory with config.json)"
            )
        model_path = Path(host_model_path).expanduser().resolve()
        expected_sha = arch.get("gr00t_config_sha256")
        if expected_sha and file_sha256(model_path / "config.json") != expected_sha:
            raise ValueError("GR00T base configuration identity mismatch at deployment")
        config_type, head_type = native_types()
        native_cfg = config_type.from_pretrained(str(model_path), local_files_only=True)
        if native_cfg.model_type != "Gr00tN1d7":
            raise ValueError("expected a GR00T N1.7 base configuration for deployment")
        native_cfg.max_num_embodiments = 1
        native_cfg.strict_action_padding_mask = native_cfg.strict_vl_padding_mask = True
        head = head_type(native_cfg)
        expert = Gr00tN1d7ActionExpert(
            head,
            protocols,
            state_dims,
            state_statistics=state_stats,
            latent_shape=tuple(arch["latent_shape"]),
        )
        if (
            expert.action_steps != temporal["action_horizon"]
            or expert.head.config.backbone_embedding_dim != context_dim
        ):
            raise ValueError(
                "GR00T deployment head dimensions disagree with the packaged manifest"
            )
        return expert
