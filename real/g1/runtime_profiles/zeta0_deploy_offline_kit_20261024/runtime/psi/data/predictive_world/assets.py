"""Explicit runtime relocation of immutable source assets (no filesystem writes)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def validate_asset_path_map(mapping: Mapping[str, str]) -> None:
    if not isinstance(mapping, dict) or any(
        not isinstance(source, str) or not isinstance(target, str)
        or not Path(source).is_absolute() or not Path(target).is_absolute()
        or '..' in Path(source).parts or '..' in Path(target).parts
        for source, target in mapping.items()
    ):
        raise ValueError('asset_path_map must map exact absolute paths to absolute paths')


def load_asset_path_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError('asset path map must be a JSON object of path strings')
    # Relative destinations make a site manifest portable with its asset bundle.
    mapping = {key: str((path.parent / value).resolve()) for key, value in payload.items()}
    validate_asset_path_map(mapping)
    return mapping


def resolve_asset_path(path: str | Path, mapping: Mapping[str, str] | None = None) -> Path:
    mapping = {} if mapping is None else mapping
    validate_asset_path_map(mapping)
    # No prefix rewriting, existence-based fallback, or recursive remapping.
    return Path(mapping.get(str(path), str(path)))


def verify_asset_boundary(paths, boundary: str | Path) -> int:
    """Check active input trees, following nested links without looping.

    This is a startup deployment check, not a content hash or a security sandbox.
    Historical path strings inside immutable provenance JSON are not rewritten.
    """
    root = Path(boundary).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('asset boundary must be a directory')
    pending = [Path(path) for path in paths]
    visited = set()
    while pending:
        path = pending.pop()
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f'asset escapes configured shared boundary: {path} -> {resolved}')
        if resolved in visited:
            continue
        visited.add(resolved)
        if resolved.is_dir():
            pending.extend(resolved.iterdir())
        elif not resolved.is_file():
            raise ValueError(f'asset is not a regular file or directory: {path}')
    return len(visited)
