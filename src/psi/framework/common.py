"""Shared filesystem and report helpers; importing these never opens hardware."""
from __future__ import annotations

import json
from pathlib import Path


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def checkout_root(explicit: str | None = None) -> Path:
    candidates = [Path(explicit).expanduser()] if explicit else [Path.cwd(), Path(__file__).parent]
    for start in candidates:
        for root in (start, *start.parents) if explicit is None else (start,):
            if (root / "scripts/train/predictive_world/train.py").is_file() and (root / "pyproject.toml").is_file():
                return root.resolve()
    raise ValueError("A Zeta source checkout is required. Use --root /path/to/Zeta0.")


def save_report(report: dict, output: str | None) -> None:
    if output:
        # Reports are new artifacts; never overwrite a previous audit.
        with Path(output).expanduser().open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
