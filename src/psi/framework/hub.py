"""Dataset catalog and safe download planner for official Zeta0 HuggingFace datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HUMANOIDARENA_TASKS = (
    "HOI_double_desk",
    "HOI_football",
    "HOI_pp_box",
    "HSI_boxing",
    "HSI_open_door",
    "HSI_sit_sofa",
    "HSI_vision_navi",
)

OFFICIAL_DATASETS: dict[str, dict[str, Any]] = {
    "humanoidarena": {
        "repo_id": "SII-yizhi/HumanoidArenaV3.1_SONIC40",
        "kind": "simulation",
        "format": "lerobot_sonic_semantic40_v3.1",
        "fps": 50,
        "state_dim": 64,
        "action_dim": 40,
        "tasks": list(HUMANOIDARENA_TASKS),
        "description": "7 HumanoidArena tasks (700 episodes, 546,460 frames at 50 Hz, SONIC semantic-40).",
    },
    "real": {
        "repo_id": "SII-yizhi/zeta0-real-dataset",
        "kind": "real_robot",
        "format": "lerobot_v3_sonic_g1_brainco",
        "fps": 30,
        "state_dim": 41,
        "action_dim": 76,
        "tasks": [],
        "description": "Unitree G1 + BrainCo real-world demonstrations collected with SONIC.",
    },
}


def resolve_dataset_spec(dataset: str) -> dict[str, Any]:
    """Resolve an official alias (``humanoidarena`` / ``real``) or explicit ``org/repo``."""
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
    selected_tasks = [t.strip() for t in (tasks or []) if t.strip()]
    if selected_tasks:
        known = set(spec.get("tasks", []))
        if known:
            unknown = sorted(set(selected_tasks) - known)
            if unknown:
                raise ValueError(
                    f"Unknown task(s) {unknown} for {spec['repo_id']}; available: {sorted(known)}"
                )
        allow_patterns = ["README.md"] + [f"{task}/**" for task in selected_tasks]
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
