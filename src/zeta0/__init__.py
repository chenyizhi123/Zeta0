"""Zeta0 public Python SDK namespace (backed by backward-compatible ``psi`` core).

Keep top-level imports lightweight so CLI and metadata callers do not import
``torch``, ``numpy``, or ``transformers`` until ``zeta0.core`` or ``zeta0.hosts``
is accessed.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("zeta0-humanoid")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = [
    "__version__",
    "core",
    "hosts",
]


def __getattr__(name: str) -> Any:
    if name in {"core", "hosts"}:
        return importlib.import_module(f"zeta0.{name}")
    raise AttributeError(f"module 'zeta0' has no attribute {name!r}")
