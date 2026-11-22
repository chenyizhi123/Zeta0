#!/usr/bin/env python3
"""Publish immutable resolved configs for the production two-stage run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

from psi.config.predictive_world import load_predictive_world_config


RELEASE_FILES = {
    "measurement_registry": "measurement_registry.json",
    "canonical_future_capabilities": "capability_registry.json",
    "action_protocol_registry": "action_protocol_registry.json",
    "state_schema_registry": "state_schema_registry.json",
    "index_path": "predictive_world_index.jsonl",
    "feature_cache": "feature_cache",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--joint-run-root", help="Separate joint configs and outputs; keeps the source world run intact")
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--joint-micro-batch", type=int)
    parser.add_argument("--joint-gradient-accumulation-steps", type=int)
    parser.add_argument("--world-updates", type=int, default=15_000)
    parser.add_argument("--joint-updates", type=int, default=20_000)
    parser.add_argument("--observed-visual-mode", choices=("history", "current_multiview"), default="history")
    parser.add_argument("--seed", type=int, default=101)
    return parser


def _read_template(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config template is not a JSON object: {path}")
    return payload


def _immutable_write(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(
                f"immutable resolved config already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _resolve_release(artifact_root: Path, qwen_model: Path) -> dict[str, str]:
    resolved = {
        name: str((artifact_root / relative).resolve())
        for name, relative in RELEASE_FILES.items()
    }
    required = [
        Path(resolved["measurement_registry"]),
        Path(resolved["canonical_future_capabilities"]),
        Path(resolved["action_protocol_registry"]),
        Path(resolved["state_schema_registry"]),
        Path(resolved["index_path"]),
        Path(resolved["feature_cache"]) / "feature_manifest.json",
        artifact_root / "source_audit.json",
        qwen_model / "config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "predictive-world data release is incomplete; missing:\n  - " + formatted
        )
    return resolved


def _resolve_config(
    template: dict,
    *,
    release: dict[str, str],
    qwen_model: Path,
    phase: str,
    output_dir: Path,
    world_model: Path,
    nodes: int,
    gpus_per_node: int,
    micro_batch: int,
    accumulation: int,
    updates: int,
    seed: int,
) -> dict:
    payload = json.loads(json.dumps(template))
    payload["data"].update(release)
    payload["model"]["qwen_model"] = str(qwen_model)
    payload["training"].update(
        {
            "phase": phase,
            "seed": seed,
            "max_updates": updates,
            "micro_batch": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_global_batch": nodes * gpus_per_node * micro_batch * accumulation,
        }
    )
    if phase == "joint":
        payload["training"]["world_model"] = str(world_model)
    else:
        payload["training"].pop("world_model", None)
    payload["runtime"].update(
        {
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "output_dir": str(output_dir),
        }
    )
    return payload


def main() -> None:
    args = _parser().parse_args()
    joint_micro_batch = args.micro_batch if args.joint_micro_batch is None else args.joint_micro_batch
    joint_accumulation = (
        args.gradient_accumulation_steps
        if args.joint_gradient_accumulation_steps is None
        else args.joint_gradient_accumulation_steps
    )
    numeric = (
        args.nodes,
        args.gpus_per_node,
        args.micro_batch,
        args.gradient_accumulation_steps,
        joint_micro_batch,
        joint_accumulation,
        args.world_updates,
        args.joint_updates,
    )
    if min(numeric) <= 0 or args.seed < 0:
        raise ValueError("topology, batch sizes and update budgets must be positive")

    project_root = Path(args.project_root).expanduser().resolve()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    qwen_model = Path(args.qwen_model).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    joint_run_root = (
        run_root if args.joint_run_root is None
        else Path(args.joint_run_root).expanduser().resolve()
    )
    templates = project_root / "configs/predictive_world"
    release = _resolve_release(artifact_root, qwen_model)

    world_output = run_root / "world-model"
    joint_output = joint_run_root / "joint"
    if (
        joint_output == world_output
        or world_output in joint_output.parents
        or joint_output in world_output.parents
    ):
        raise ValueError("world-model and joint outputs must be separate")
    world_bundle = world_output / "world-model-final"
    locations = {
        "world_preflight": run_root / "preflight/world-model",
        "world": world_output,
        "joint_preflight": joint_run_root / "preflight/joint",
        "joint": joint_output,
    }
    world_template = _read_template(templates / "world_model.template.json")
    joint_template = _read_template(templates / "joint.template.json")
    specs = {
        "world_preflight": (world_template, "world_model", args.world_updates),
        "world": (world_template, "world_model", args.world_updates),
        "joint_preflight": (joint_template, "joint", args.joint_updates),
        "joint": (joint_template, "joint", args.joint_updates),
    }
    published: dict[str, str] = {}
    for name, (template, phase, updates) in specs.items():
        payload = _resolve_config(
            template,
            release=release,
            qwen_model=qwen_model,
            phase=phase,
            output_dir=locations[name],
            world_model=world_bundle,
            nodes=args.nodes,
            gpus_per_node=args.gpus_per_node,
            micro_batch=joint_micro_batch if phase == "joint" else args.micro_batch,
            accumulation=joint_accumulation if phase == "joint" else args.gradient_accumulation_steps,
            updates=updates,
            seed=args.seed,
        )
        payload["data"]["observed_visual_mode"] = args.observed_visual_mode
        config_root = joint_run_root if phase == "joint" else run_root
        path = config_root / "configs" / f"{name}.json"
        _immutable_write(path, payload)
        # This validates all semantic fields while deliberately allowing a
        # same-contract retry to find an already populated output directory.
        load_predictive_world_config(path, allow_existing_output_override=True)
        published[name] = str(path)

    print(
        json.dumps(
            {
                "event": "predictive_world_two_stage_run_resolved",
                "artifact_root": str(artifact_root),
                "run_root": str(run_root),
                "joint_run_root": str(joint_run_root),
                "world_model_bundle": str(world_bundle),
                "effective_global_batch": (
                    args.nodes
                    * args.gpus_per_node
                    * args.micro_batch
                    * args.gradient_accumulation_steps
                ),
                "joint_effective_global_batch": (
                    args.nodes * args.gpus_per_node * joint_micro_batch * joint_accumulation
                ),
                "configs": published,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
