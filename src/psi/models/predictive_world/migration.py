"""Explicit tensor-by-tensor legacy migration with a machine-readable audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MigrationReport:
    copied: tuple[tuple[str, str], ...]
    unused_source: tuple[str, ...]
    initialized_target: tuple[str, ...]


def migrate_state_dict(
    source: Mapping[str, Tensor],
    modules: Mapping[str, nn.Module],
    mapping: Mapping[str, str],
    *,
    allow_initialized_targets: frozenset[str] = frozenset(),
    allow_unused_sources: frozenset[str] = frozenset(),
) -> MigrationReport:
    """Apply only declared exact-shape copies; every exception must be allowlisted."""

    targets = {
        f"{module_name}.{key}": value
        for module_name, module in modules.items()
        for key, value in module.state_dict().items()
    }
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("multiple source tensors cannot map to one target tensor")
    unknown_source = set(mapping) - set(source)
    unknown_target = set(mapping.values()) - set(targets)
    if unknown_source or unknown_target:
        raise KeyError(
            f"migration mapping references unknown tensors: source={sorted(unknown_source)}, "
            f"target={sorted(unknown_target)}"
        )
    migrated = dict(targets)
    copied = []
    for source_key, target_key in sorted(mapping.items()):
        tensor = source[source_key]
        if tensor.shape != targets[target_key].shape:
            raise ValueError(
                f"migration shape mismatch {source_key}->{target_key}: "
                f"{tuple(tensor.shape)} != {tuple(targets[target_key].shape)}"
            )
        migrated[target_key] = tensor.to(dtype=targets[target_key].dtype)
        copied.append((source_key, target_key))
    initialized = set(targets) - set(mapping.values())
    unused = set(source) - set(mapping)
    if initialized != set(allow_initialized_targets):
        raise ValueError(
            "initialized target allowlist mismatch: "
            f"actual={sorted(initialized)}, allowed={sorted(allow_initialized_targets)}"
        )
    if unused != set(allow_unused_sources):
        raise ValueError(
            f"unused source allowlist mismatch: actual={sorted(unused)}, "
            f"allowed={sorted(allow_unused_sources)}"
        )
    for module_name, module in modules.items():
        prefix = f"{module_name}."
        module.load_state_dict({
            key[len(prefix):]: value for key, value in migrated.items() if key.startswith(prefix)
        }, strict=True)
    return MigrationReport(tuple(copied), tuple(sorted(unused)), tuple(sorted(initialized)))


def load_tensor_file(path: str) -> dict[str, Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, Mapping) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, Mapping) or not all(isinstance(item, Tensor) for item in value.values()):
        raise ValueError("legacy checkpoint must be a tensor mapping or contain state_dict")
    return dict(value)


__all__ = ["MigrationReport", "load_tensor_file", "migrate_state_dict"]
