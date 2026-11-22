"""Dispatch to maintained entrypoints without duplicating model/data logic."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

WORKFLOWS = {
    "demo": ("scripts/examples/quickstart.py", ".", "core", "synthetic installation example"),
    "prepare": ("scripts/train/predictive_world/prepare_sonic_lerobot_task.py", ".", "train", "native SONIC v3 preparation; publishes new artifacts"),
    "train": ("scripts/train/predictive_world/train.py", ".", "train", "shared trainer; may allocate GPUs"),
    "export": ("scripts/export_real_sonic_research_policy.py", ".", "train", "immutable research inference package"),
    "record": ("real/g1/scripts/record_lerobot.py", "real/g1", "capture", "live collection; connects to robot sensors"),
    "deploy": ("real/g1/scripts/deploy_online_model.py", "real/g1", "core", "hardware runtime; target --execute can actuate the robot"),
}


def workflow_plan(root: Path, name: str, arguments: list[str], python: str | None) -> dict:
    script, cwd, profile, effect = WORKFLOWS[name]
    interpreter = str(Path(shutil.which(python) or python).expanduser().absolute()) if python else sys.executable
    argv = [interpreter, str(root / script), *arguments]
    return {"format": "zeta0.workflow_plan.v1", "ok": (root / script).is_file(), "workflow": name,
            "argv": argv, "command": shlex.join(argv), "cwd": str(root / cwd),
            "environment_profile": profile, "effect": effect, "executed": False,
            "entrypoint_exists": (root / script).is_file(),
            "next_step": "Inspect the plan; add workflow --execute to run this exact entrypoint. Target flags after -- retain their original meaning."}


def execute_plan(plan: dict, root: Path) -> int:
    if not plan["entrypoint_exists"]:
        raise ValueError("Workflow entrypoint is missing from this source checkout")
    env = dict(os.environ)
    if plan["workflow"] in {"record", "deploy"}:
        # Hardware runtimes own their isolated psi imports. Do not inject the
        # active training tree or inherit a different checkout's PYTHONPATH.
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(root / "src")
    # No shell, no installation, no submission to a resource platform.
    return subprocess.call(plan["argv"], cwd=plan["cwd"], env=env)
