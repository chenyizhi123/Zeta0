"""Inspect explicitly selected/existing environments without installing packages."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from psi.data.predictive_world.assets import load_asset_path_map

# Presence checks only. Version/ABI validation remains with the lock, pip check,
# native imports and the relevant acceptance tests; do not report these as passed.
PROFILES = {
    "core": ((3, 10), ("torch", "numpy", "safetensors")),
    "train": ((3, 10), ("torch", "numpy", "safetensors", "pydantic", "transformers", "pyarrow", "datasets", "torchcodec", "lerobot")),
    "gr00t": ((3, 10), ("torch", "numpy", "safetensors", "pydantic", "transformers", "pyarrow", "datasets", "torchcodec", "lerobot", "diffusers", "omegaconf", "dm-tree")),
    "capture": ((3, 12), ("numpy", "lerobot", "pyzmq", "msgpack", "lerobot-robot-sonic")),
}
PROBE = """import importlib.metadata as m, json, sys
packages = {}
for name in json.loads(sys.argv[1]):
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
print(json.dumps(dict(version=list(sys.version_info[:3]), executable=sys.executable,
                     prefix=sys.prefix, packages=packages)))
"""


def candidates(root: Path | None, selected: list[str]) -> list[str]:
    if selected:
        values = selected
    else:
        values = [sys.executable]
        if os.environ.get("VIRTUAL_ENV"):
            values.append(str(Path(os.environ["VIRTUAL_ENV"]) / "bin/python"))
        if os.environ.get("CONDA_PREFIX"):
            values.append(str(Path(os.environ["CONDA_PREFIX"]) / "bin/python"))
        if root:
            values.extend(str(root / name / "bin/python") for name in (
                ".venv", ".venv-check", ".venv-psi0", ".venv-gr00t",
                "real/g1/.venv_lerobot", "real/g1/.venv_policy",
            ) if (root / name / "bin/python").is_file())
    # Preserve venv symlinks: resolving bin/python changes its environment.
    return list(dict.fromkeys(str(Path(shutil.which(value) or value).expanduser().absolute()) for value in values))


def inspect_environment(python: str, profile: str) -> dict:
    version, packages = PROFILES[profile]
    report = {"python": python, "metadata_checks_passed": False}
    try:
        result = subprocess.run([python, "-I", "-c", PROBE, json.dumps(packages)],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            return {**report, "error": f"Interpreter probe exited with {result.returncode}"}
        probe = json.loads(result.stdout)
        missing = [name for name, installed in probe["packages"].items() if installed is None]
        report.update(probe, missing_packages=missing, expected_python=".".join(map(str, version)),
                      metadata_checks_passed=tuple(probe["version"][:2]) == version and not missing)
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as error:
        report["error"] = f"Interpreter probe failed: {type(error).__name__}"
    return report


def doctor(root: Path | None, *, selected: list[str], profile: str, asset_path_map: str | None) -> dict:
    environments = [inspect_environment(python, profile) for python in candidates(root, selected)]
    mapping = load_asset_path_map(asset_path_map)
    assets = [{"source": source, "local_path": target, "exists": Path(target).exists()}
              for source, target in mapping.items()]
    suitable = [item["python"] for item in environments if item["metadata_checks_passed"]]
    return {
        "format": "zeta0.doctor.v1", "profile": profile,
        "ok": bool(suitable) and all(item["exists"] for item in assets),
        "scope": "Python/package metadata and mapped path existence only",
        "environments": environments, "reuse_candidates": suitable, "assets": assets,
        "unchecked": ["package constraints/ABI", "native imports", "CUDA", "asset hashes", "hardware"],
        "next_steps": ["Select an existing candidate with workflow --python; no environment is changed.",
                       "Run uv pip check --python <selected-python>, then the relevant acceptance tier.",
                       "Missing mapped assets require the matching revision; a different cached model is not a substitute."],
        "downloads": 0, "environments_modified": False,
    }
