"""Inspect captured data using the training adapter as the numerical authority."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import read_object


def inspect_data(dataset: str, *, full: bool = False, task: str = "inspection",
                 validation_fraction: float = 0.0, seed: int = 292285) -> dict:
    root = Path(dataset).expanduser().resolve()
    info = read_object(root / "meta/info.json")
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("Dataset features must be a JSON object")
    candidate = info.get("codebase_version") == "v3.0" and info.get("robot_type") == "sonic_g1"
    report = {
        "format": "zeta0.data_inspection.v1", "dataset": str(root), "ok": True,
        "scope": "metadata", "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"), "fps": info.get("fps"),
        "episodes": info.get("total_episodes"), "frames": info.get("total_frames"),
        "features": {key: {field: value.get(field) for field in ("dtype", "shape", "names")}
                     for key, value in features.items() if isinstance(value, dict)},
        "candidate_adapter": "sonic_lerobot" if candidate else None,
        "numerical_contract_verified": False, "training_admission_decided": False,
        "video_decoding_verified": False,
        "next_step": ("Run data inspect --full with a training environment, then workflow prepare."
                      if candidate else "Use an explicit dataset adapter; metadata similarity does not establish compatibility."),
    }
    if full:
        if not candidate:
            raise ValueError("Full inspection currently supports native sonic_g1 LeRobot v3; use the matching existing adapter for other formats.")
        from psi.data.predictive_world.sonic_lerobot_post_training import inspect_dataset
        _, instruction, _, audit, _ = inspect_dataset(root, task=task, seed=seed,
                                                     validation_fraction=validation_fraction)
        outcomes = Counter("success" if ep.get("task_success") is True else
                           "failure" if ep.get("task_success") is False else "unknown"
                           for ep in audit["episodes"])
        report.update(scope="training-adapter numerical/provenance audit", instruction=instruction,
                      numerical_contract_verified=True, outcomes=dict(outcomes), audit=audit,
                      next_step="Review episode outcomes and the split, then prepare using the same task/seed/split. Video decoding is checked separately during materialization.")
    return report
