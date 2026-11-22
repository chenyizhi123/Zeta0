"""GR00T N1.7 action architecture behind Zeta's native-action interface.

The pinned upstream DiT, VL preprocessing, state/action encoders and decoder
remain the computation owners. This adapter supplies Zeta conditions, normalized
task coordinates, explicit flow inputs and clean committed-prefix semantics.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

from psi.data.predictive_world.contracts import sanitize_masked_tensor
from psi.models.predictive_world.action_expert import TemporalNativeActionExpert


SOURCE_REVISION = "3b49067943eb1ab2eb2203b93d461c4390c78da3"
ADAPTER_FORMAT = "zeta0.gr00t_n1d7.native_head.v1"
PROJECTOR_PREFIXES = ("state_encoder.", "action_encoder.", "action_decoder.")


def native_source() -> Path:
    """Resolve pinned source for both checkout and installed-wheel callers."""
    configured = os.environ.get("ZETA0_GR00T_SOURCE")
    source = (Path(configured).expanduser() if configured else
              Path(__file__).resolve().parents[4] / "third_party/Isaac-GR00T")
    source = source.resolve()
    if not (source / "gr00t").is_dir():
        raise ValueError(
            "N1.7 source is missing; acquire the pinned Isaac-GR00T checkout and "
            "set ZETA0_GR00T_SOURCE to its root when using an installed wheel"
        )
    try:
        def git(*arguments):
            return subprocess.check_output(
                ["git", "-C", str(source), *arguments], text=True,
                stderr=subprocess.DEVNULL, timeout=30,
            ).strip()

        if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
            raise ValueError("N1.7 source must be its own Git checkout")
        if git("rev-parse", "HEAD") != SOURCE_REVISION:
            raise ValueError("N1.7 source differs from the supported pinned revision")
        if git("status", "--porcelain", "--untracked-files=all"):
            raise ValueError("N1.7 source checkout must be clean")
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("could not verify the pinned N1.7 source checkout") from error
    return source


@lru_cache(maxsize=1)
def native_types():
    """Explicit construction-time import; no import-time downloads or GPU work."""
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS", "").lower() not in {"", "0", "false", "no", "off"}:
        raise ValueError("GR00T test-only weight skipping cannot be used by the Zeta backend")
    source = native_source()
    sys.path.insert(0, str(source))
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    import gr00t
    if not Path(gr00t.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("a different GR00T package was imported before the pinned source")
    return Gr00tN1d7Config, Gr00tN1d7ActionHead


class _Statistics(nn.Module):
    def __init__(self, dim, mean, std, normalization_space="native"):
        super().__init__()
        self.dim = int(dim)
        self.normalization_space = str(normalization_space)
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if mean.shape != (self.dim,) or std.shape != mean.shape:
            raise ValueError("normalization statistics disagree with task dimensions")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("normalization statistics must be finite with positive std")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)


class Gr00tN1d7ActionExpert(nn.Module):
    """One task's native N1.7 head with independent trainable Zeta future tokens.

    The external sigma convention remains Zeta's noise fraction. Internally
    native t=1-sigma and v_native=action-noise; returning -v_native makes both
    training targets and the existing decreasing-sigma Euler solver agree.
    RTC retains the clean prefix while the native head receives the common
    suffix time, matching N1.7's global-time inpainting convention.
    """

    normalize_actions = TemporalNativeActionExpert.normalize_actions
    denormalize_actions = TemporalNativeActionExpert.denormalize_actions
    protocol_dims = TemporalNativeActionExpert.protocol_dims
    sample = TemporalNativeActionExpert.sample

    def __init__(self, native_head, protocols, state_dims, *, state_statistics,
                 latent_shape=(3, 8, 256)):
        super().__init__()
        if len(protocols) != 1 or len(state_dims) != 1:
            raise ValueError("N1.7 v1 requires one shared action protocol and state schema")
        config = native_head.config
        if config.max_num_embodiments != 1 or config.state_history_length != 1:
            raise ValueError("N1.7 task head requires one new embodiment and one current state")
        if not config.strict_action_padding_mask or not config.strict_vl_padding_mask:
            raise ValueError("Zeta N1.7 requires strict action and VL padding masks")
        if (not isinstance(native_head.vl_self_attention, nn.Identity)
                and not native_head.mask_vl_padding):
            raise ValueError("N1.7 VL preprocessing must honor the observation padding mask")
        self.head = native_head
        self.activation_checkpointing = False
        self.protocol_ids = tuple(protocols)
        self.adapters = nn.ModuleDict({name: _Statistics(
            spec["dim"], spec["mean"], spec["std"], spec.get("normalization_space", "native")
        ) for name, spec in protocols.items()})
        self.state_schema_id, state_dim = next(iter(state_dims.items()))
        mean, std = state_statistics[self.state_schema_id]
        self.state_statistics = _Statistics(state_dim, mean, std)
        if state_dim > config.max_state_dim or max(self.protocol_dims.values()) > config.max_action_dim:
            raise ValueError("task dimensions exceed native N1.7 padding widths")
        self.action_steps = int(config.action_horizon)
        self.state_history_count = 1
        self.latent_shape = tuple(latent_shape)
        self.future_projection = nn.Sequential(
            nn.LayerNorm(self.latent_shape[-1]),
            nn.Linear(self.latent_shape[-1], config.backbone_embedding_dim),
        )
        self.future_position = nn.Parameter(torch.empty(
            self.latent_shape[0], self.latent_shape[1], config.backbone_embedding_dim))
        nn.init.normal_(self.future_position, std=0.02)

    def sample_sigma(self, batch_size, device):
        return 1 - self.head.sample_time(batch_size, device, torch.float32)

    def condition(self, context, context_valid, image_mask, z_raw, latent_group_valid):
        from transformers.feature_extraction_utils import BatchFeature

        if context_valid.shape != context.shape[:2] or image_mask.shape != context_valid.shape:
            raise ValueError("context and image masks must share shape [B,S]")
        if not context_valid.bool().any(dim=1).all():
            raise ValueError("each observation needs at least one valid context token")
        if z_raw.shape != (context.shape[0], *self.latent_shape):
            raise ValueError("future latent shape mismatch")
        if latent_group_valid.shape != z_raw.shape[:2]:
            raise ValueError("future group mask shape mismatch")
        context = sanitize_masked_tensor(context, context_valid[..., None].expand_as(context),
                                         name="GR00T observation context")
        observed = self.head.process_backbone_output(BatchFeature(data={
            "backbone_features": context,
            "backbone_attention_mask": context_valid.bool(),
        }))["backbone_features"]
        clean_z = sanitize_masked_tensor(z_raw, latent_group_valid[:, :, None, None].expand_as(z_raw),
                                        name="GR00T future latent")
        future = (self.future_projection(clean_z) + self.future_position).flatten(1, 2).to(observed.dtype)
        valid = latent_group_valid[:, :, None].expand(-1, -1, self.latent_shape[1]).flatten(1, 2).bool()
        # Upstream uses complementary image/non-image masks. Two masked views of
        # the SAME 24 logical tokens make them available in both kinds of block,
        # without modifying upstream attention or learning a second future bank.
        # Each block attends to exactly one view; no future is counted twice.
        features = torch.cat((observed, future, future), dim=1)
        attention = torch.cat((context_valid.bool(), valid, valid), dim=1)
        images = torch.cat((image_mask.bool(), torch.ones_like(valid), torch.zeros_like(valid)), dim=1)
        return features, attention, images

    def _velocity(self, *, normalized_action_samples, action_valid, protocol_ids,
                 qwen_context, context_valid, image_mask, raw_states,
                 state_dimension_valid, state_schema_ids, state_time_valid,
                 state_times, z_raw, latent_group_valid, sigma_a):
        batch, steps, width = normalized_action_samples.shape
        config = self.head.config
        if steps != self.action_steps or action_valid.shape != normalized_action_samples.shape:
            raise ValueError("N1.7 action samples and masks must match its fixed horizon")
        if width > config.max_action_dim or width < self.adapters[self.protocol_ids[0]].dim:
            raise ValueError("action width is incompatible with the task protocol")
        if tuple(protocol_ids) != self.protocol_ids * batch:
            raise ValueError("unknown N1.7 task action protocol")
        if tuple(state_schema_ids) != (self.state_schema_id,) * batch:
            raise ValueError("unknown N1.7 task state schema")
        if (raw_states.shape[:2] != (batch, 1) or state_dimension_valid.shape != raw_states.shape
                or state_time_valid.shape != (batch, 1) or state_times.shape != (batch, 1)):
            raise ValueError("N1.7 requires a single current state and aligned masks/times")
        if raw_states.shape[-1] != self.state_statistics.dim:
            raise ValueError("state width differs from the registered schema")
        if not torch.isfinite(state_times[state_time_valid.bool()]).all():
            raise ValueError("valid state times must be finite")
        if (state_times[state_time_valid.bool()] > 0).any():
            raise ValueError("N1.7 current state must be causal")
        if sigma_a.shape == (batch,):
            sigma = sigma_a
        elif sigma_a.shape == (batch, steps):
            sigma = sigma_a.max(dim=1).values
            # Only clean prefix (sigma=0) plus a common suffix time are supported.
            if ((sigma_a != 0) & (sigma_a != sigma[:, None])).any():
                raise ValueError("N1.7 requires one common suffix flow time")
            zero = sigma_a.eq(0)
            if ((~zero).long().cumsum(1).gt(0) & zero & sigma[:, None].ne(0)).any():
                raise ValueError("zero-time committed actions must form a leading prefix")
        else:
            raise ValueError("action flow time must be [B] or [B,T]")
        if not torch.isfinite(sigma).all() or (sigma < 0).any() or (sigma > 1).any():
            raise ValueError("action flow time must be finite in [0,1]")
        t = ((1 - sigma.float()) * self.head.num_timestep_buckets).long()
        valid = action_valid.bool()
        # A padded width supplied by callers must never activate unused dimensions.
        if valid[..., self.adapters[self.protocol_ids[0]].dim:].any():
            raise ValueError("action padding cannot contain valid task coordinates")
        clean = sanitize_masked_tensor(normalized_action_samples, valid, name="GR00T noisy action")
        clean = F.pad(clean, (0, config.max_action_dim - width))
        native_valid = F.pad(valid, (0, config.max_action_dim - width))
        state_valid = state_dimension_valid.bool() & state_time_valid[:, :, None].bool()
        state = sanitize_masked_tensor(raw_states, state_valid, name="GR00T current state")
        state = torch.where(state_valid, (state - self.state_statistics.mean) / self.state_statistics.std, 0.)
        state = F.pad(state, (0, config.max_state_dim - state.shape[-1]))
        embodiment = torch.zeros(batch, dtype=torch.long, device=clean.device)
        dtype = next(self.head.parameters()).dtype
        features, attention, images = self.condition(
            qwen_context.to(dtype), context_valid, image_mask, z_raw, latent_group_valid)
        state_features = self.head.state_encoder(state.to(dtype), embodiment)
        if self.training and self.head.state_dropout_prob > 0:
            keep = torch.rand(batch, device=state.device) >= self.head.state_dropout_prob
            state_features = state_features * keep[:, None, None]
        state_features = state_features * state_time_valid[:, :, None]
        action_features = self.head.action_encoder(clean.to(dtype), t, embodiment)
        if config.add_pos_embed:
            action_features = action_features + self.head.position_embedding(
                torch.arange(steps, device=clean.device))[None]
        action_tokens_valid = native_valid.any(-1)
        action_features = action_features * action_tokens_valid[..., None]
        hidden_valid = torch.cat((state_time_valid.bool(), action_tokens_valid), dim=1)
        kwargs = {"image_mask": images, "backbone_attention_mask": attention} if config.use_alternate_vl_dit else {}
        output = self.head.model(
            hidden_states=torch.cat((state_features, action_features), dim=1),
            encoder_hidden_states=features, encoder_attention_mask=attention,
            timestep=t, hidden_attention_mask=hidden_valid, **kwargs)
        native_velocity = self.head.action_decoder(output, embodiment)[:, -steps:, :width]
        return -native_velocity.float() * valid

    def velocity(self, **arguments):
        # Checkpoint a complete ordinary reader call. Frozen response readers
        # checkpoint outside functional_call in phases._response_velocity, so
        # recomputation reinstates their detached state and eval mode together.
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            # Positional nested tensors let checkpoint discover the CUDA device
            # and preserve its dropout RNG, not only the CPU RNG.
            return checkpoint(lambda values: self._velocity(**values), arguments, use_reentrant=False)
        return self._velocity(**arguments)

    forward = velocity


def load_action_expert(model_path, protocols, state_dims, state_statistics, *, latent_shape):
    """Strictly load shared native weights; new task projectors are explicit.

    Arena semantic40/state64 are not any pretrained relative-EEF embodiment.
    Use a new native embodiment adapter, then copy the entire initialized T to G.
    """
    from safetensors import safe_open

    config_type, head_type = native_types()
    model_path = Path(model_path)
    config = config_type.from_pretrained(str(model_path), local_files_only=True)
    if config.model_type != "Gr00tN1d7":
        raise ValueError("expected a GR00T N1.7 checkpoint")
    config.max_num_embodiments = 1
    config.strict_action_padding_mask = config.strict_vl_padding_mask = True
    head = head_type(config)
    parameters = head.state_dict()
    index = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {name for name in parameters if not name.startswith(PROJECTOR_PREFIXES)}
    found = {name.removeprefix("action_head.") for name in index if name.startswith("action_head.")
             and not name.removeprefix("action_head.").startswith(PROJECTOR_PREFIXES)}
    if expected != found:
        raise ValueError(f"N1.7 shared-head checkpoint keys differ: {sorted(expected ^ found)[:12]}")
    for shard in sorted({index["action_head." + name] for name in expected}):
        if Path(shard).name != shard:
            raise ValueError("unsafe checkpoint shard path")
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for name in sorted(expected):
                if index["action_head." + name] == shard:
                    value = handle.get_tensor("action_head." + name)
                    if value.shape != parameters[name].shape:
                        raise ValueError(f"N1.7 checkpoint shape mismatch for {name}")
                    parameters[name].copy_(value)
    return Gr00tN1d7ActionExpert(head, protocols, state_dims, state_statistics=state_statistics,
                                latent_shape=latent_shape)
