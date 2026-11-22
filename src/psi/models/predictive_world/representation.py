"""Future process encoder, causal world readout, and representation losses."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

from psi.data.predictive_world.contracts import sanitize_masked_tensor
from psi.models.predictive_world.temporal_attention import (
    additive_attention_bias,
    prefix_visibility,
)
from psi.models.stable_attention import QKNormAttention


def _sinusoidal_time(value: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=value.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angle = value.float().reshape(-1, 1) * frequency[None]
    result = torch.cat((angle.sin(), angle.cos()), dim=-1)
    return torch.nn.functional.pad(result, (0, dim - result.shape[-1]))


class _PrefixAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = QKNormAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = QKNormAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.GELU(approximate="tanh"), nn.Linear(ff_dim, hidden_dim)
        )

    def forward(self, query: Tensor, context: Tensor, self_bias: Tensor, cross_bias: Tensor) -> Tensor:
        normalized = self.self_norm(query)
        query = query + self.self_attn(
            normalized, normalized, normalized, attn_mask=self_bias, need_weights=False
        )[0]
        query = query + self.cross_attn(
            self.cross_norm(query), context, context, attn_mask=cross_bias, need_weights=False
        )[0]
        return query + self.ff(self.ff_norm(query))


class FutureTrajectoryEncoder(nn.Module):
    """Encode future evidence into prefix-causal [B,G,S,Dz] raw latents.

    Inputs are padded tensors at this boundary. Ragged patch storage is
    materialized by the collator without pooling or fabricating extra patches.
    No task text, future action, or history-encoder condition is accepted.
    """

    def __init__(
        self,
        *,
        visual_dim: int = 2048,
        measurement_dim: int,
        latent_shape: tuple[int, int, int] = (3, 8, 256),
        scene_slots: int = 4,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
    ) -> None:
        super().__init__()
        groups, slots, latent_dim = latent_shape
        if slots != 8 or scene_slots <= 0 or scene_slots >= slots:
            raise ValueError("future encoder requires a non-empty scene/body partition of 8 slots")
        if measurement_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("invalid future encoder dimensions")
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.scene_slots = int(scene_slots)
        self.num_heads = int(num_heads)
        self.visual_projection = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim))
        self.visual_position_projection = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.camera_embedding = nn.Embedding(64, hidden_dim)
        self.measurement_projection = nn.Sequential(
            nn.Linear(2 * measurement_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.latent_queries = nn.Parameter(torch.empty(groups, slots, hidden_dim))
        self.slot_type = nn.Parameter(torch.empty(2, hidden_dim))
        self.null_context = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [_PrefixAttentionBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_blocks)]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))
        nn.init.normal_(self.latent_queries, std=0.02)
        nn.init.normal_(self.slot_type, std=0.02)
        nn.init.normal_(self.null_context, std=0.02)

    def forward(
        self,
        *,
        anchor_patches: Tensor,
        anchor_patch_valid: Tensor,
        current_measurements: Tensor,
        current_measurement_valid: Tensor,
        future_patches: Tensor,
        future_patch_valid: Tensor,
        future_scene_times: Tensor,
        future_measurements: Tensor,
        future_measurement_valid: Tensor,
        future_measurement_times: Tensor,
        group_valid: Tensor,
        future_patch_positions: Tensor | None = None,
        future_patch_camera_indices: Tensor | None = None,
    ) -> Tensor:
        if anchor_patches.ndim != 3 or anchor_patch_valid.shape != anchor_patches.shape[:2]:
            raise ValueError("anchor patches/mask must have shape [B,P,D]/[B,P]")
        batch = anchor_patches.shape[0]
        groups, slots, _ = self.latent_shape
        if future_patches.ndim != 4 or future_patches.shape[:2] != (batch, groups):
            raise ValueError("future patches must have shape [B,G,P,D]")
        if future_patch_valid.shape != future_patches.shape[:3]:
            raise ValueError("future patch validity must have shape [B,G,P]")
        if future_patch_positions is None or future_patch_camera_indices is None:
            future_patch_positions = torch.zeros(
                *future_patches.shape[:3], 2, device=future_patches.device
            )
            future_patch_camera_indices = torch.zeros(
                future_patches.shape[:3], dtype=torch.long, device=future_patches.device
            )
        if (
            future_patch_positions.shape != (*future_patches.shape[:3], 2)
            or future_patch_camera_indices.shape != future_patches.shape[:3]
            or (future_patch_camera_indices < 0).any()
            or (future_patch_camera_indices >= self.camera_embedding.num_embeddings).any()
        ):
            raise ValueError("future patch position/camera metadata has an invalid shape or value")
        if future_scene_times.shape != (batch, groups):
            raise ValueError("future scene times must have shape [B,G]")
        if current_measurements.ndim != 2 or current_measurement_valid.shape != current_measurements.shape:
            raise ValueError("current measurements/mask must have shape [B,Dm]")
        if current_measurements.shape[0] != batch:
            raise ValueError("current measurement batch size mismatch")
        if future_measurements.ndim != 4 or future_measurements.shape[:2] != (batch, groups):
            raise ValueError("future measurements must have shape [B,G,M,Dm]")
        if future_measurements.shape[-1] != current_measurements.shape[-1]:
            raise ValueError("current and future measurement widths must match")
        if future_measurement_valid.shape != future_measurements.shape:
            raise ValueError("future measurement validity must match values")
        if future_measurement_times.shape != future_measurements.shape[:3]:
            raise ValueError("future measurement times must have shape [B,G,M]")
        if group_valid.shape != (batch, groups):
            raise ValueError("group_valid must have shape [B,G]")
        if (group_valid & ~future_patch_valid.any(dim=-1)).any():
            raise ValueError("a canonical valid latent group needs future visual evidence")

        anchor_clean = sanitize_masked_tensor(
            anchor_patches,
            anchor_patch_valid[:, :, None].expand_as(anchor_patches),
            name="anchor patches",
        )
        future_patch_clean = sanitize_masked_tensor(
            future_patches,
            future_patch_valid[:, :, :, None].expand_as(future_patches),
            name="future patches",
        )
        current_clean = sanitize_masked_tensor(
            current_measurements, current_measurement_valid, name="current measurements"
        )
        future_clean = sanitize_masked_tensor(
            future_measurements, future_measurement_valid, name="future measurements"
        )
        # Missing measurements carry NaN timestamps in the release. Attention
        # masks cannot remove NaNs after time encoding has contaminated K/V.
        future_measurement_times = sanitize_masked_tensor(
            future_measurement_times,
            future_measurement_valid.any(dim=-1),
            name="future measurement times",
        )
        future_scene_times = sanitize_masked_tensor(
            future_scene_times, future_patch_valid.any(dim=-1), name="future scene times"
        )
        anchor_tokens = self.visual_projection(anchor_clean)
        future_scene = (
            self.visual_projection(future_patch_clean)
            + self.visual_position_projection(future_patch_positions.float())
            + self.camera_embedding(future_patch_camera_indices)
        )
        current_features = torch.cat((current_clean, current_measurement_valid.float()), dim=-1)
        current_token = self.measurement_projection(current_features)[:, None]
        future_features = torch.cat((future_clean, future_measurement_valid.float()), dim=-1)
        future_body = self.measurement_projection(future_features)
        future_scene = future_scene + self.time_projection(
            _sinusoidal_time(future_scene_times, future_scene.shape[-1])
        ).reshape(batch, groups, 1, -1)
        future_body = future_body + self.time_projection(
            _sinusoidal_time(future_measurement_times, future_body.shape[-1])
        ).reshape_as(future_body)

        null = self.null_context.expand(batch, -1, -1)
        context = torch.cat(
            (
                null,
                anchor_tokens,
                current_token,
                future_scene.flatten(1, 2),
                future_body.flatten(1, 2),
            ),
            dim=1,
        )
        anchor_count = anchor_tokens.shape[1] + 2  # null + current measurement
        scene_per_group = future_scene.shape[2]
        body_per_group = future_body.shape[2]
        key_groups = torch.cat(
            (
                torch.full((anchor_count,), -1, device=context.device, dtype=torch.long),
                torch.arange(groups, device=context.device).repeat_interleave(scene_per_group),
                torch.arange(groups, device=context.device).repeat_interleave(body_per_group),
            )
        )
        key_valid = torch.cat(
            (
                torch.ones(batch, 1, device=context.device, dtype=torch.bool),
                anchor_patch_valid.bool(),
                current_measurement_valid.any(dim=-1, keepdim=True),
                future_patch_valid.flatten(1, 2).bool(),
                future_measurement_valid.any(dim=-1).flatten(1, 2).bool(),
            ),
            dim=1,
        )
        query_groups = torch.arange(groups, device=context.device).repeat_interleave(slots)
        self_bias = additive_attention_bias(
            prefix_visibility(query_groups, query_groups), dtype=context.dtype
        )
        cross_visible = prefix_visibility(query_groups, key_groups)[None] & key_valid[:, None]
        cross_bias = additive_attention_bias(cross_visible, dtype=context.dtype)
        cross_bias = cross_bias[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            batch * self.num_heads, groups * slots, context.shape[1]
        )
        queries = self.latent_queries[None].expand(batch, -1, -1, -1).clone()
        queries[:, :, : self.scene_slots] += self.slot_type[0]
        queries[:, :, self.scene_slots :] += self.slot_type[1]
        queries = queries.flatten(1, 2)
        for block in self.blocks:
            queries = block(queries, context, self_bias, cross_bias)
        latent = self.output(queries).reshape(batch, groups, slots, -1).float()
        return latent * group_valid[:, :, None, None].to(latent.dtype)


class EMAFutureTrajectoryEncoder(nn.Module):
    """Checkpointed online/target pair with a stable target-coordinate forward API.

    ``forward`` deliberately publishes the EMA target encoder.  Representation
    training must opt into ``encode_online`` so every downstream consumer (F,
    latent caching and policy training) sees the same stable coordinate system.
    Update markers live in this module because phase-objective state is not part
    of the predictive-world checkpoint module manifest.
    """

    def __init__(self, online_encoder: FutureTrajectoryEncoder, *, decay: float = 0.99) -> None:
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError("target encoder EMA decay must be in [0,1)")
        self.online_encoder = online_encoder
        self.target_encoder = copy.deepcopy(online_encoder)
        self.decay = float(decay)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()
        self.register_buffer("last_target_update", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("last_normalizer_update", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("normalizer_frozen", torch.tensor(False, dtype=torch.bool))

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self.online_encoder.latent_shape

    def train(self, mode: bool = True):
        super().train(mode)
        self.online_encoder.train(mode)
        self.target_encoder.eval()
        return self

    def encode_online(self, **arguments: Tensor) -> Tensor:
        return self.online_encoder(**arguments)


    def forward(self, **arguments: Tensor) -> Tensor:
        return self.target_encoder(**arguments)

    @torch.no_grad()
    def update_target(self, optimizer_update: int) -> bool:
        """Advance EMA at most once per optimizer update.

        Called at the start of an update, this makes the target a one-update-lag
        EMA of the online encoder and is invariant to gradient accumulation.
        """

        update = int(optimizer_update)
        previous = int(self.last_target_update.item())
        if update < previous:
            raise ValueError("target encoder update index moved backwards")
        if update == previous:
            return False
        for target, online in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True
        ):
            target.lerp_(online.detach().to(dtype=target.dtype), 1.0 - self.decay)
        for target, online in zip(
            self.target_encoder.buffers(), self.online_encoder.buffers(), strict=True
        ):
            if target.dtype.is_floating_point:
                target.lerp_(online.detach().to(dtype=target.dtype), 1.0 - self.decay)
            else:
                target.copy_(online.detach())
        self.last_target_update.fill_(update)
        return True

    @torch.no_grad()
    def mark_normalizer_update(self, optimizer_update: int) -> bool:
        update = int(optimizer_update)
        previous = int(self.last_normalizer_update.item())
        if update < previous:
            raise ValueError("latent normalizer update index moved backwards")
        if update == previous:
            return False
        self.last_normalizer_update.fill_(update)
        return True

    @torch.no_grad()
    def freeze_normalizer_stage(self) -> None:
        self.normalizer_frozen.fill_(True)


class WorldReadout(nn.Module):
    """Decode visual and measured-body trajectories from raw prefix latents."""

    def __init__(
        self,
        *,
        visual_dim: int = 2048,
        measurement_dim: int,
        latent_shape: tuple[int, int, int] = (3, 8, 256),
        max_scene_patches: int = 160,
        body_points_per_group: int = 10,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        num_heads: int = 8,
        ff_dim: int = 2048,
    ) -> None:
        super().__init__()
        groups, _, latent_dim = latent_shape
        if min(measurement_dim, max_scene_patches, body_points_per_group) <= 0:
            raise ValueError("world readout dimensions must be positive")
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.max_scene_patches = int(max_scene_patches)
        self.body_points_per_group = int(body_points_per_group)
        self.num_heads = int(num_heads)
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.anchor_projection = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim))
        self.null_context = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.scene_queries = nn.Parameter(torch.empty(groups, max_scene_patches, hidden_dim))
        self.body_queries = nn.Parameter(torch.empty(groups, body_points_per_group, hidden_dim))
        self.query_type = nn.Parameter(torch.empty(2, hidden_dim))
        self.blocks = nn.ModuleList(
            [_PrefixAttentionBlock(hidden_dim, num_heads, ff_dim) for _ in range(num_blocks)]
        )
        self.scene_output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, visual_dim))
        self.body_output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, measurement_dim))
        nn.init.normal_(self.null_context, std=0.02)
        nn.init.normal_(self.scene_queries, std=0.02)
        nn.init.normal_(self.body_queries, std=0.02)
        nn.init.normal_(self.query_type, std=0.02)


    def forward(
        self,
        *,
        anchor_patches: Tensor,
        anchor_patch_valid: Tensor,
        z_raw: Tensor,
        group_valid: Tensor,
        scene_patch_count: int,
    ) -> tuple[Tensor, Tensor]:
        if scene_patch_count <= 0 or scene_patch_count > self.max_scene_patches:
            raise ValueError("scene_patch_count exceeds the configured readout capacity")
        if z_raw.ndim != 4 or tuple(z_raw.shape[1:]) != self.latent_shape:
            raise ValueError("z_raw disagrees with the configured latent shape")
        batch, groups, slots, _ = z_raw.shape
        if anchor_patches.ndim != 3 or anchor_patches.shape[0] != batch:
            raise ValueError("anchor patches must have shape [B,P,D]")
        if anchor_patch_valid.shape != anchor_patches.shape[:2] or group_valid.shape != (batch, groups):
            raise ValueError("readout masks disagree with inputs")
        anchor_clean = sanitize_masked_tensor(
            anchor_patches,
            anchor_patch_valid[:, :, None].expand_as(anchor_patches),
            name="world readout anchor",
        )
        anchor = self.anchor_projection(anchor_clean)
        latent_valid = group_valid[:, :, None, None].expand_as(z_raw)
        clean_latent = sanitize_masked_tensor(z_raw, latent_valid, name="world readout latent")
        latent = self.latent_projection(clean_latent)
        context = torch.cat((self.null_context.expand(batch, -1, -1), anchor, latent.flatten(1, 2)), dim=1)
        key_groups = torch.cat(
            (
                torch.full((anchor.shape[1] + 1,), -1, device=z_raw.device, dtype=torch.long),
                torch.arange(groups, device=z_raw.device).repeat_interleave(slots),
            )
        )
        key_valid = torch.cat(
            (
                torch.ones(batch, 1, device=z_raw.device, dtype=torch.bool),
                anchor_patch_valid.bool(),
                group_valid[:, :, None].expand(-1, -1, slots).flatten(1, 2),
            ),
            dim=1,
        )
        scene = self.scene_queries[:, :scene_patch_count] + self.query_type[0]
        body = self.body_queries + self.query_type[1]
        queries = torch.cat((scene.flatten(0, 1), body.flatten(0, 1)), dim=0)
        query_groups = torch.cat(
            (
                torch.arange(groups, device=z_raw.device).repeat_interleave(scene_patch_count),
                torch.arange(groups, device=z_raw.device).repeat_interleave(self.body_points_per_group),
            )
        )
        query = queries[None].expand(batch, -1, -1)
        self_bias = additive_attention_bias(prefix_visibility(query_groups, query_groups), dtype=query.dtype)
        cross_visible = prefix_visibility(query_groups, key_groups)[None] & key_valid[:, None]
        cross_bias = additive_attention_bias(cross_visible, dtype=query.dtype)
        cross_bias = cross_bias[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            batch * self.num_heads, query.shape[1], context.shape[1]
        )
        for block in self.blocks:
            query = block(query, context, self_bias, cross_bias)
        scene_count = groups * scene_patch_count
        scene_hidden = query[:, :scene_count].reshape(batch, groups, scene_patch_count, -1)
        body_hidden = query[:, scene_count:].reshape(
            batch, groups, self.body_points_per_group, -1
        )
        scene_prediction = self.scene_output(scene_hidden).float()
        body_prediction = self.body_output(body_hidden).float()
        valid = group_valid[:, :, None, None]
        return scene_prediction * valid, body_prediction * valid


def representation_regularizer(
    z_raw: Tensor,
    group_valid: Tensor,
    *,
    rms_limit: float = 4.0,
    token_std_floor: float = 0.1,
) -> Tensor:
    """Mean upper-scale and within-group token-diversity regularizer."""

    per_sample, sample_valid = representation_regularizer_per_sample(
        z_raw,
        group_valid,
        rms_limit=rms_limit,
        token_std_floor=token_std_floor,
    )
    return torch.where(sample_valid, per_sample, torch.zeros_like(per_sample)).sum() / (
        sample_valid.float().sum().clamp_min(1)
    )


def representation_regularizer_per_sample(
    z_raw: Tensor,
    group_valid: Tensor,
    *,
    rms_limit: float = 4.0,
    token_std_floor: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Return a regularizer and validity bit for every sample.

    The scale term prevents an unbounded latent coordinate system.  The slot
    diversity term keeps the eight tokens in each valid temporal group from
    silently collapsing to one repeated code.
    """

    if z_raw.ndim != 4 or group_valid.shape != z_raw.shape[:2]:
        raise ValueError("latent and group validity shapes disagree")
    if rms_limit <= 0 or token_std_floor <= 0:
        raise ValueError("regularizer limits must be positive")
    # sqrt has an infinite derivative at exactly zero. Clamping the squared
    # mean before sqrt keeps fully masked/zero toy batches differentiable and
    # prevents the later zero mask from receiving ``NaN * 0`` gradients.
    rms = z_raw.float().square().mean(dim=(-1, -2)).clamp_min(1e-12).sqrt()
    token_std = z_raw.float().std(dim=-2, unbiased=False).mean(dim=-1)
    per_group = torch.relu(rms - rms_limit).square() + torch.relu(
        token_std_floor - token_std
    ).square()
    weights = group_valid.float()
    sample_valid = group_valid.any(dim=-1)
    per_sample = (per_group * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1)
    return torch.where(sample_valid, per_sample, torch.zeros_like(per_sample)), sample_valid


__all__ = [
    "EMAFutureTrajectoryEncoder",
    "FutureTrajectoryEncoder",
    "WorldReadout",
    "representation_regularizer",
    "representation_regularizer_per_sample",
]
