#!/usr/bin/env python3
"""Export a deploy-only policy bundle from a strict joint/policy checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from psi.config.predictive_world import load_predictive_world_config, predictive_world_loss_contract
from psi.data.predictive_world.cache import file_sha256
from psi.data.predictive_world.evaluation import (
    checkpoint_evaluation_binding,
    evaluation_release_from_index,
    validate_passed_evaluation_report,
)
from psi.models.predictive_world.checkpoint import (
    load_predictive_world_bundle,
    predictive_world_code_fingerprint,
    read_predictive_world_bundle,
    save_predictive_world_bundle,
)
from psi.models.predictive_world.factory import build_phase


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def export_expected_metadata(config) -> dict:
    """Check the objective for both routes; only policy-only needs a world source."""
    expected = {"loss_contract": predictive_world_loss_contract(config.training)}
    if config.training.phase == "policy":
        if not config.training.world_model:
            raise ValueError("policy export requires a world-model source")
        source = read_predictive_world_bundle(config.training.world_model)
        expected["reference_id"] = source["metadata"]["reference_id"]
    elif config.training.phase != "joint":
        raise ValueError("deployment export requires a joint or policy phase")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved phase=joint/policy config")
    parser.add_argument("--policy", required=True, help="Completed strict action-policy bundle")
    parser.add_argument(
        "--evaluation-report",
        required=True,
        help="Passed report bound to this exact policy/config/data/code bundle",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, payload = load_predictive_world_config(
        args.config, allow_existing_output_override=True
    )
    if config.training.phase not in {"joint", "policy"}:
        raise ValueError("deployment export requires a phase=joint or phase=policy config")
    evaluation_path = Path(args.evaluation_report)
    evaluation_report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(evaluation_report, dict):
        raise ValueError("evaluation report must be a JSON object")
    reported_binding = evaluation_report.get("evaluation_binding")
    reported_release = (
        reported_binding.get("evaluation_release")
        if isinstance(reported_binding, dict) else None
    )
    if not isinstance(reported_release, dict):
        raise ValueError("evaluation report lacks an immutable release binding")
    _, release = evaluation_release_from_index(
        config.data.index_path,
        split=reported_release.get("split"),
        count=reported_release.get("sample_count"),
        seed=reported_release.get("seed"),
    )
    if release != reported_release:
        raise ValueError("evaluation report release binding is stale or malformed")
    binding = checkpoint_evaluation_binding(
        args.policy,
        args.config,
        config,
        payload,
        code_revision=_git_revision(),
        code_fingerprint=predictive_world_code_fingerprint(Path(__file__)),
        evaluation_release=release,
    )
    validate_passed_evaluation_report(evaluation_report, binding)
    evaluation_sha256 = file_sha256(evaluation_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built = build_phase(config, device=device)
    policy_sidecar = load_predictive_world_bundle(
        args.policy,
        phase=config.training.phase,
        modules=built.checkpoint_modules,
        expected_metadata=export_expected_metadata(config),
    )
    modules = {
        "history_state_adapters": built.forecast_modules["history_state_adapters"],
        "history_encoder": built.forecast_modules["history_encoder"],
        "latent_transition": built.forecast_modules["latent_transition"],
        "latent_normalizer": built.reference_modules["normalizer"],
        "student": built.checkpoint_modules["student"],
    }
    metadata = dict(policy_sidecar["metadata"])
    metadata["qwen_reference"] = str(config.model.qwen_model)
    metadata["export_contract"] = "observed_history_only"
    metadata["evaluation_report_sha256"] = evaluation_sha256
    deployment_contract = {
        "format": "zeta0.predictive_world.deployment.parallel_process",
        "route": "predictive_world",
        "history_visual_offsets_s": list(config.data.observed_visual_offsets_s),
        **config.data.observation_contract,
        "state_history_count": config.data.state_history_count,
        "state_hz": config.data.state_query_hz,
        "action_horizon": config.data.action_horizon,
        "action_hz": config.data.action_hz,
        "prediction_semantics": config.model.latent_transition.semantics,
        "student_latent_mode": config.training.student_latent_mode,
        "loss_contract": predictive_world_loss_contract(config.training),
        "future_group_ends_s": list(config.data.future_group_ends_s),
        "action_solver_steps": config.model.action_solver_steps,
        "qwen_reference": str(config.model.qwen_model),
        "reference_id": metadata["reference_id"],
        "evaluation_binding": binding,
        "evaluation_report_sha256": evaluation_sha256,
        "requires_clock_mapping": True,
        "requires_queue_revision_guard": True,
        "requires_execution_backend_validation": True,
    }
    save_predictive_world_bundle(
        args.output, phase="deploy", modules=modules,
        completed_updates=policy_sidecar["completed_updates"], metadata=metadata,
        auxiliary_json={
            "deployment_contract.json": deployment_contract,
            "evaluation_report.json": evaluation_report,
        },
    )
    verified = read_predictive_world_bundle(args.output)
    forbidden = {"encoder", "readout", "teacher"} & set(verified["module_names"])
    if forbidden:
        raise RuntimeError(f"deploy bundle contains training-only modules: {sorted(forbidden)}")
    print(json.dumps({
        "event": "predictive_world_policy_exported",
        "output": args.output,
        "modules": verified["module_names"],
        "reference_id": metadata["reference_id"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
