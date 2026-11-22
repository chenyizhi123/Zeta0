"""Schema-specific state adapters for heterogeneous predictor pretraining."""

from __future__ import annotations

import torch
from torch import Tensor, nn

STATE_DIM = 64  # Fixed common state embedding width in predictive-world checkpoints.


class StateSchemaAdapter(nn.Module):
    """Map one declared raw state schema plus dimension mask to state64."""

    def __init__(
        self,
        raw_dim: int,
        *,
        mean: Tensor | None = None,
        std: Tensor | None = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if raw_dim <= 0 or hidden_dim <= 0:
            raise ValueError("raw_dim and hidden_dim must be positive")
        if hidden_dim != 128:
            raise ValueError("the frozen scaling protocol fixes adapter hidden_dim=128")
        self.raw_dim = int(raw_dim)
        self.register_buffer("mean", self._stat(mean, 0.0, "mean"))
        self.register_buffer("std", self._stat(std, 1.0, "std"))
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("state normalization statistics must be finite")
        if (self.std <= 0).any():
            raise ValueError("state normalization std must be positive")
        self.projection = nn.Sequential(
            nn.Linear(2 * self.raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def _stat(self, value: Tensor | None, default: float, name: str) -> Tensor:
        result = torch.full((self.raw_dim,), default) if value is None else torch.as_tensor(value).float()
        if result.shape != (self.raw_dim,):
            raise ValueError(f"{name} must have shape [{self.raw_dim}]")
        # A registry statistic can initialize several independent adapters.
        # Each module owns its buffers, including before a CPU checkpoint save.
        return result.detach().clone()

    @property
    def expected_parameter_count(self) -> int:
        return 256 * self.raw_dim + 8_384

    def forward(
        self,
        raw_state: Tensor,
        dimension_mask: Tensor,
        *,
        state_present: Tensor | None = None,
    ) -> Tensor:
        if raw_state.shape != dimension_mask.shape or raw_state.shape[-1] != self.raw_dim:
            raise ValueError(
                f"raw_state and dimension_mask must share shape [...,{self.raw_dim}]"
            )
        mask = dimension_mask.to(device=raw_state.device, dtype=torch.bool)
        if state_present is None:
            state_present = mask.any(dim=-1)
        elif state_present.shape != raw_state.shape[:-1]:
            raise ValueError("state_present must match raw_state leading dimensions")
        state_present = state_present.to(device=raw_state.device, dtype=torch.bool)
        if mask[~state_present].any():
            raise ValueError("missing-state entries cannot mark dimensions valid")
        normalized = (raw_state.float() - self.mean) / self.std
        normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
        features = torch.cat((normalized, mask.to(dtype=normalized.dtype)), dim=-1)
        adapted = self.projection(features.to(dtype=self.projection[0].weight.dtype))
        return adapted * state_present[..., None].to(dtype=adapted.dtype)


class StateAdapterBank(nn.Module):
    """Apply schema-specific adapters to a padded heterogeneous batch."""

    def __init__(
        self,
        schema_dims: dict[str, int],
        *,
        statistics: dict[str, tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        super().__init__()
        if not schema_dims:
            raise ValueError("at least one state schema is required")
        statistics = statistics or {}
        self.schema_dims = {str(key): int(value) for key, value in schema_dims.items()}
        self.adapters = nn.ModuleDict()
        for schema, raw_dim in self.schema_dims.items():
            stats = statistics.get(schema)
            self.adapters[schema] = StateSchemaAdapter(
                raw_dim,
                mean=None if stats is None else stats[0],
                std=None if stats is None else stats[1],
            )

    def forward(
        self,
        raw_states: Tensor,
        dimension_mask: Tensor,
        schema_ids: list[str] | tuple[str, ...],
        *,
        state_present: Tensor,
    ) -> Tensor:
        if raw_states.ndim != 3 or raw_states.shape != dimension_mask.shape:
            raise ValueError("raw_states and dimension_mask must share shape [B,T,Dmax]")
        batch, history, _ = raw_states.shape
        if len(schema_ids) != batch or state_present.shape != (batch,):
            raise ValueError("schema_ids/state_present must have one value per batch row")
        output_dtype = next(self.adapters.parameters()).dtype
        output = torch.zeros(
            (batch, history, STATE_DIM), device=raw_states.device, dtype=output_dtype
        )
        for schema in sorted(set(schema_ids)):
            indices = [index for index, value in enumerate(schema_ids) if value == schema]
            present_indices = [index for index in indices if bool(state_present[index])]
            if not present_indices:
                continue
            if schema not in self.adapters:
                raise KeyError(f"no state adapter registered for schema {schema!r}")
            raw_dim = self.schema_dims[schema]
            index_tensor = torch.tensor(present_indices, device=raw_states.device)
            if dimension_mask.index_select(0, index_tensor)[..., raw_dim:].any():
                raise ValueError(f"schema {schema!r} marks dimensions beyond raw_dim={raw_dim}")
            selected_states = raw_states.index_select(0, index_tensor)[..., :raw_dim]
            selected_mask = dimension_mask.index_select(0, index_tensor)[..., :raw_dim]
            selected_present = torch.ones(
                (len(present_indices), history), device=raw_states.device, dtype=torch.bool
            )
            adapted = self.adapters[schema](
                selected_states, selected_mask, state_present=selected_present
            )
            output[index_tensor] = adapted.to(dtype=output.dtype)
        return output


__all__ = ["StateAdapterBank", "StateSchemaAdapter"]
