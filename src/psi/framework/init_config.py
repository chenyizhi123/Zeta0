"""Generate resolved Zeta-0 joint training configurations from maintained host templates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HOST_TEMPLATES = {
    "psi0": "configs/psi0/joint.template.json",
    "zeta_native": "configs/psi0/joint.template.json",
    "gr00t_n1d7": "configs/gr00t_n1d7/joint.template.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_required_placeholders(node: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            sub = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_find_required_placeholders(value, sub))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            found.extend(_find_required_placeholders(value, f"{prefix}[{idx}]"))
    elif isinstance(node, str) and node.startswith("REQUIRED_"):
        found.append(f"{prefix}={node}")
    return found


def init_training_config(
    root: Path,
    *,
    host: str,
    output: str | Path,
    qwen_model: str | None = None,
    gr00t_model: str | None = None,
    prepared_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_updates: int | None = None,
    gpus_per_node: int | None = None,
    micro_batch: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Scaffold a v6/raw training JSON config without hand-editing 90+ lines of JSON."""
    key = host.strip().lower()
    if key == "pi05":
        raise NotImplementedError(
            "Host 'pi05' is planned for v0.3. Choose --host psi0 or --host gr00t_n1d7 today."
        )
    if key not in HOST_TEMPLATES:
        raise ValueError(f"Unsupported host {host!r}; choose from: psi0, zeta_native, gr00t_n1d7")
    template_path = (root / HOST_TEMPLATES[key]).resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Template config not found: {template_path}")
    destination = Path(output).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing config: {destination}")

    payload = json.loads(template_path.read_text(encoding="utf-8"))
    if qwen_model:
        payload["model"]["qwen_model"] = str(Path(qwen_model).expanduser().resolve())
    if gr00t_model:
        gr00t_path = Path(gr00t_model).expanduser().resolve()
        payload["model"]["gr00t_model"] = str(gr00t_path)
        config_json = gr00t_path / "config.json"
        if config_json.is_file():
            payload["model"]["gr00t_config_sha256"] = _sha256(config_json)
    if prepared_dir:
        prep = Path(prepared_dir).expanduser().resolve()
        candidates = {
            "measurement_registry": prep / "registries/measurement_registry.json",
            "canonical_future_capabilities": prep / "registries/capability_registry.json",
            "action_protocol_registry": prep / "registries/action_protocol_registry.json",
            "state_schema_registry": prep / "registries/state_schema_registry.json",
            "index_path": prep / "predictive_world_index.jsonl",
            "feature_cache": prep / "feature_cache",
        }
        for field_name, path in candidates.items():
            payload["data"][field_name] = str(path)
    if output_dir:
        payload["runtime"]["output_dir"] = str(Path(output_dir).expanduser().resolve())
    if max_updates is not None:
        if max_updates <= 0:
            raise ValueError("max_updates must be positive")
        payload["training"]["max_updates"] = int(max_updates)
    if gpus_per_node is not None:
        if gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        payload["runtime"]["gpus_per_node"] = int(gpus_per_node)
    if micro_batch is not None:
        if micro_batch <= 0:
            raise ValueError("micro_batch must be positive")
        payload["training"]["micro_batch"] = int(micro_batch)
        accum = int(payload["training"].get("gradient_accumulation_steps", 1))
        nodes = int(payload["runtime"].get("nodes", 1))
        gpus = int(payload["runtime"].get("gpus_per_node", 1))
        payload["training"]["effective_global_batch"] = int(micro_batch) * accum * nodes * gpus

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    remaining = _find_required_placeholders(payload)
    return {
        "format": "zeta0.init_config.v1",
        "ok": True,
        "host_architecture": payload["model"].get("host_architecture", "zeta_native"),
        "template": str(template_path),
        "output": str(destination),
        "remaining_required_fields": remaining,
        "ready_for_dry_run": len(remaining) == 0,
        "next_step": (
            f"zeta workflow train -- --config {destination} --dry-run"
            if not remaining
            else "Fill the remaining REQUIRED_* fields in the generated JSON, then run train --dry-run."
        ),
    }
