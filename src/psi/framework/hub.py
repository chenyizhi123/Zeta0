"""Dataset catalog and safe download planner for official Zeta0 HuggingFace datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REAL_G1_TASK_ARCHIVES: dict[str, str] = {
    "blow_dry_hair": "real/blow_dry_hair.zip",
    "clean_table_trash": "real/clean_table_trash.zip",
    "place_chair": "real/place_chair.zip",
    "place_doll": "real/place_doll.zip",
    "place_tape_on_table": "real/place_tape_on_table.zip",
    "tidy_scene": "real/tidy_scene.zip",
    "flatten_box": "additional/flatten_box.zip",
}

OFFICIAL_DATASETS: dict[str, dict[str, Any]] = {
    "real": {
        "repo_id": "SII-yizhi/zeta0-real-dataset",
        "kind": "real_robot",
        "format": "lerobot_v3_sonic_g1_brainco",
        "fps": 30,
        "state_dim": 41,
        "action_dim": 76,
        "tasks": list(REAL_G1_TASK_ARCHIVES),
        "task_patterns": dict(REAL_G1_TASK_ARCHIVES),
        "base_patterns": ["README.md", "SHA256SUMS"],
        "description": "Unitree G1 + BrainCo real-world demonstrations (6 primary + 1 additional task, 739 episodes, 872,409 frames).",
    },
}


def resolve_dataset_spec(dataset: str) -> dict[str, Any]:
    """Resolve an official alias (``real``) or explicit ``org/repo``."""
    key = dataset.strip()
    if not key:
        raise ValueError("dataset alias or HuggingFace repo_id must be non-empty")
    if key in OFFICIAL_DATASETS:
        return {"alias": key, **OFFICIAL_DATASETS[key]}
    if "/" not in key or len(key.split("/")) != 2 or any(not part.strip() for part in key.split("/")):
        aliases = ", ".join(sorted(OFFICIAL_DATASETS))
        raise ValueError(
            f"dataset must be an official alias ({aliases}) or a HuggingFace repo_id 'owner/name'"
        )
    return {
        "alias": None,
        "repo_id": key,
        "kind": "custom",
        "format": "external_huggingface_dataset",
        "tasks": [],
        "description": f"Custom HuggingFace dataset repository {key}.",
    }


def build_pull_plan(
    dataset: str,
    *,
    output: str | Path,
    tasks: list[str] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build a read-only HuggingFace pull plan without contacting the network."""
    spec = resolve_dataset_spec(dataset)
    destination = Path(output).expanduser().resolve()
    allow_patterns: list[str] | None = None
    selected_tasks = [
        part.strip()
        for item in (tasks or [])
        for part in item.split(",")
        if part.strip()
    ]
    if selected_tasks:
        known = set(spec.get("tasks", []))
        if known:
            unknown = sorted(set(selected_tasks) - known)
            if unknown:
                raise ValueError(
                    f"Unknown task(s) {unknown} for {spec['repo_id']}; available: {sorted(known)}"
                )
        task_patterns = spec.get("task_patterns") or {}
        base_patterns = list(spec.get("base_patterns") or ["README.md"])
        allow_patterns = base_patterns + [
            task_patterns.get(task, f"{task}/**") for task in selected_tasks
        ]
    return {
        "format": "zeta0.dataset_pull_plan.v1",
        "ok": True,
        "alias": spec["alias"],
        "repo_id": spec["repo_id"],
        "kind": spec["kind"],
        "revision": revision or "main",
        "tasks": selected_tasks,
        "allow_patterns": allow_patterns,
        "output": str(destination),
        "token_configured": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "executed": False,
        "next_step": "Review the plan and re-run with --execute to download via huggingface_hub.",
    }


def execute_pull(plan: dict[str, Any]) -> dict[str, Any]:
    """Download the planned dataset snapshot using ``huggingface_hub``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for --execute (install with: pip install huggingface_hub)"
        ) from exc

    destination = Path(plan["output"])
    destination.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=plan["repo_id"],
        repo_type="dataset",
        revision=plan["revision"],
        local_dir=str(destination),
        allow_patterns=plan["allow_patterns"],
    )
    return {
        **plan,
        "executed": True,
        "local_dir": str(Path(local_dir).resolve()),
        "next_step": f"Inspect the downloaded dataset with: zeta data inspect {local_dir}",
    }
