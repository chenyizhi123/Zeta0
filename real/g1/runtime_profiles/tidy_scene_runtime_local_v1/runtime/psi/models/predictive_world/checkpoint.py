"""Strict, immutable checkpoint bundles for predictive-world phases."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping

from torch import nn


FORMAT_ID = "oraclevla.predictive_world.parallel_process"
# Latent-flow/recurrent checkpoints must never be resumed as parallel predictors.
SCHEMA_REVISION = 1
PHASE_MODULES = {
    "world_model": frozenset(
        {
            "encoder", "readout", "teacher", "normalizer",
            "history_state_adapters", "history_encoder", "latent_transition",
        }
    ),
    "joint": frozenset(
        {
            "encoder", "readout", "teacher", "normalizer",
            "history_state_adapters", "history_encoder", "latent_transition", "student",
        }
    ),
    "representation_training": frozenset({"encoder", "readout", "teacher", "normalizer"}),
    "reference": frozenset({"encoder", "readout", "teacher", "normalizer"}),
    "forecast": frozenset({"history_state_adapters", "history_encoder", "latent_transition"}),
    "policy": frozenset({"student"}),
    "deploy": frozenset(
        {
            "history_state_adapters",
            "history_encoder",
            "latent_transition",
            "latent_normalizer",
            "student",
        }
    ),
}
REQUIRED_METADATA = frozenset(
    {
        "architecture",
        "temporal_abi",
        "encoder_id",
        "processor_id",
        "protocol_registry_sha256",
        "measurement_registry_sha256",
        "normalization_id",
        "reference_id",
        "data_fingerprint",
        "resolved_config_sha256",
        "code_revision",
        "code_fingerprint",
        "environment_fingerprint",
    }
)


def predictive_world_code_fingerprint(start: str | Path) -> str:
    """Hash the exact source snapshot used by the predictive-world runtime.

    A Git commit alone is insufficient for research runs made from a dirty
    worktree.  This content hash includes tracked, modified, deleted and
    untracked (non-ignored) files under the runtime source/script trees.
    """

    location = Path(start).expanduser().resolve()
    if location.is_file():
        location = location.parent
    root = next(
        (candidate for candidate in (location, *location.parents) if (candidate / ".git").exists()),
        None,
    )
    if root is None:
        raise ValueError(f"cannot locate Git repository from {location}")
    result = subprocess.run(
        [
            "git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
            "--exclude-standard", "--", "src/psi", "scripts", "pyproject.toml",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    relative_paths = sorted(path for path in result.stdout.split(b"\0") if path)
    digest = hashlib.sha256()
    digest.update(b"oraclevla.predictive_world.code.v1\0")
    for encoded_path in relative_paths:
        relative = encoded_path.decode("utf-8")
        path = root / relative
        digest.update(encoded_path)
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"missing\0")
            continue
        digest.update(b"executable\0" if path.stat().st_mode & 0o111 else b"regular\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictive_world_bundle_identity(path: str | Path) -> str:
    """Hash the verified immutable sidecar that identifies one exact bundle."""

    sidecar = read_predictive_world_bundle(path)
    encoded = json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_METADATA - metadata.keys())
    placeholders = [key for key, value in metadata.items() if isinstance(value, str) and "REQUIRED_" in value]
    if missing or placeholders:
        raise ValueError(
            f"predictive-world checkpoint metadata is incomplete: missing={missing}, "
            f"placeholders={placeholders}"
        )


def _validate_phase_metadata(phase: str, metadata: Mapping[str, Any]) -> None:
    if phase != "joint":
        return
    parent = metadata.get("parent_world_model")
    initialization = metadata.get("initialization", {})
    if isinstance(initialization, Mapping) and initialization.get("kind") == "random_task_modules":
        from psi.config.predictive_world import PredictiveWorldTrainingConfig
        from psi.models.predictive_world.scratch import random_joint_initialization

        if parent is not None:
            raise ValueError("scratch joint checkpoint cannot claim a pretrained world parent")
        optimization = metadata.get("optimization_contract")
        if not isinstance(optimization, Mapping):
            raise ValueError("scratch joint checkpoint requires its explicit optimization contract")
        training = PredictiveWorldTrainingConfig(**optimization)
        if dict(initialization) != random_joint_initialization(training):
            raise ValueError("scratch checkpoint initialization disagrees with its training contract")
        return
    if not isinstance(parent, Mapping):
        raise ValueError("joint checkpoint metadata requires parent_world_model provenance")
    if parent.get("phase") != "world_model":
        raise ValueError("joint checkpoint parent must be a world_model bundle")
    updates = parent.get("completed_updates")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
        raise ValueError("joint checkpoint parent completed_updates is invalid")
    digest = parent.get("weights_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("joint checkpoint parent weights_sha256 is invalid")
    for key in ("reference_id", "normalization_id"):
        if not isinstance(parent.get(key), str) or not parent[key]:
            raise ValueError(f"joint checkpoint parent {key} is invalid")


def _validate_modules(phase: str, modules: Mapping[str, nn.Module]) -> None:
    if phase not in PHASE_MODULES:
        raise ValueError(f"unsupported predictive-world checkpoint phase {phase!r}")
    names = frozenset(modules)
    if names != PHASE_MODULES[phase]:
        raise ValueError(
            f"phase={phase} requires modules={sorted(PHASE_MODULES[phase])}, got={sorted(names)}"
        )
    if len({id(module) for module in modules.values()}) != len(modules):
        raise ValueError("checkpoint module names cannot alias the same module object")


def save_predictive_world_bundle(
    output_dir: str | Path,
    *,
    phase: str,
    modules: Mapping[str, nn.Module],
    completed_updates: int,
    metadata: Mapping[str, Any],
    auxiliary_json: Mapping[str, Mapping[str, Any]] | None = None,
    publish_complete: bool = True,
) -> Path:
    """Publish tensors and sidecar, then write `.complete` last."""

    from safetensors.torch import save_file

    _validate_modules(phase, modules)
    _validate_metadata(metadata)
    _validate_phase_metadata(phase, metadata)
    if completed_updates < 0:
        raise ValueError("completed_updates must be non-negative")
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for module_name, module in sorted(modules.items()):
        for key, value in module.state_dict().items():
            tensors[f"{module_name}.{key}"] = value.detach().cpu().contiguous()
    weights_tmp = directory / ".predictive_world.safetensors.tmp"
    weights_path = directory / "predictive_world.safetensors"
    save_file(tensors, str(weights_tmp))
    weights_tmp.replace(weights_path)
    sidecar = {
        "format_id": FORMAT_ID,
        "schema_revision": SCHEMA_REVISION,
        "phase": phase,
        "completed_updates": int(completed_updates),
        "module_names": sorted(modules),
        "weights_file": weights_path.name,
        "weights_sha256": _sha256(weights_path),
        "metadata": dict(metadata),
    }
    sidecar_tmp = directory / ".predictive_world_checkpoint.json.tmp"
    sidecar_path = directory / "predictive_world_checkpoint.json"
    sidecar_tmp.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar_tmp.replace(sidecar_path)
    for name, payload in sorted((auxiliary_json or {}).items()):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError(f"invalid checkpoint auxiliary JSON name {name!r}")
        auxiliary_tmp = directory / f".{name}.tmp"
        auxiliary_tmp.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        auxiliary_tmp.replace(directory / name)
    if publish_complete:
        complete_predictive_world_bundle(directory)
    return sidecar_path


def complete_predictive_world_bundle(path: str | Path) -> Path:
    """Atomically mark a fully materialized bundle complete after verifying its bytes."""

    directory = Path(path).expanduser().resolve()
    marker = directory / ".complete"
    if marker.exists():
        raise FileExistsError(f"predictive-world bundle is already complete: {directory}")
    sidecar_path = directory / "predictive_world_checkpoint.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError("predictive-world checkpoint sidecar is missing")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    weights = directory / str(sidecar.get("weights_file", ""))
    if not weights.is_file() or _sha256(weights) != sidecar.get("weights_sha256"):
        raise ValueError("cannot complete predictive-world bundle with missing/corrupt weights")
    temporary = directory / f".complete.{uuid.uuid4().hex}.tmp"
    temporary.write_text("complete\n", encoding="utf-8")
    temporary.replace(marker)
    return marker


def read_predictive_world_bundle(path: str | Path, *, allow_diagnostic: bool = False) -> dict[str, Any]:
    directory = Path(path).expanduser().resolve()
    if not (directory / ".complete").is_file():
        raise FileNotFoundError("predictive-world checkpoint is incomplete")
    sidecar_path = directory / "predictive_world_checkpoint.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("format_id") != FORMAT_ID or sidecar.get("schema_revision") != SCHEMA_REVISION:
        raise ValueError("predictive-world checkpoint format/schema mismatch")
    phase = sidecar.get("phase")
    if phase not in PHASE_MODULES:
        raise ValueError("predictive-world checkpoint phase is invalid")
    if frozenset(sidecar.get("module_names", ())) != PHASE_MODULES[phase]:
        raise ValueError("checkpoint module manifest disagrees with its phase")
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("predictive-world checkpoint metadata is missing")
    if metadata.get('artifact_use') == 'diagnostic_only' and not allow_diagnostic:
        raise ValueError('diagnostic-only checkpoint requires explicit diagnostic loading; not for production/export')
    _validate_metadata(metadata)
    _validate_phase_metadata(str(phase), metadata)
    weights_path = (directory / str(sidecar.get("weights_file", ""))).resolve()
    if weights_path.parent != directory or not weights_path.is_file():
        raise FileNotFoundError("checkpoint weights are missing or escape their directory")
    if _sha256(weights_path) != sidecar.get("weights_sha256"):
        raise ValueError("predictive-world checkpoint weight hash mismatch")
    return sidecar


def load_predictive_world_bundle(
    path: str | Path,
    *,
    phase: str,
    modules: Mapping[str, nn.Module],
    expected_metadata: Mapping[str, Any] | None = None,
    allow_diagnostic: bool = False,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    _validate_modules(phase, modules)
    directory = Path(path).expanduser().resolve()
    sidecar = read_predictive_world_bundle(directory, allow_diagnostic=allow_diagnostic)
    if sidecar["phase"] != phase:
        raise ValueError(f"checkpoint phase mismatch: saved={sidecar['phase']}, expected={phase}")
    metadata = sidecar["metadata"]
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in (expected_metadata or {}).items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"predictive-world checkpoint metadata mismatch: {mismatches}")
    tensors = load_file(str(directory / sidecar["weights_file"]), device="cpu")
    expected_keys = {
        f"{module_name}.{key}"
        for module_name, module in modules.items()
        for key in module.state_dict()
    }
    actual_keys = set(tensors)
    if expected_keys != actual_keys:
        raise RuntimeError(
            "strict predictive-world checkpoint keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)[:8]}, "
            f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
        )
    for module_name, module in modules.items():
        prefix = f"{module_name}."
        state = {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}
        expected = module.state_dict()
        wrong_shape = sorted(
            key for key in expected if tuple(expected[key].shape) != tuple(state[key].shape)
        )
        if wrong_shape:
            raise RuntimeError(f"strict predictive-world checkpoint shape mismatch: {wrong_shape[:8]}")
        module.load_state_dict(state, strict=True)
    return sidecar


__all__ = [
    "FORMAT_ID",
    "PHASE_MODULES",
    "SCHEMA_REVISION",
    "complete_predictive_world_bundle",
    "load_predictive_world_bundle",
    "predictive_world_code_fingerprint",
    "predictive_world_bundle_identity",
    "read_predictive_world_bundle",
    "save_predictive_world_bundle",
]


def prune_completed_checkpoints(output, *, keep: int):
    """Retain recent recovery points in this run only; preserve partial artifacts."""
    import shutil
    root = Path(output).resolve()
    if keep < 1:
        raise ValueError("checkpoint retention must be positive")
    complete = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if (suffix.isdigit() and path.is_dir() and not path.is_symlink()
                and (path / ".complete").is_file() and (path / ".trainer_complete").is_file()):
            complete.append((int(suffix), path))
    removed = []
    for _, path in sorted(complete)[:-keep]:
        shutil.rmtree(path)
        removed.append(str(path))
    return removed
