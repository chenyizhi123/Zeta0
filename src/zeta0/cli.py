"""Public CLI entrypoint mirror for ``python -m zeta0.cli``."""

from __future__ import annotations

from psi.cli import main, parser

__all__ = ["main", "parser"]

if __name__ == "__main__":
    raise SystemExit(main())
