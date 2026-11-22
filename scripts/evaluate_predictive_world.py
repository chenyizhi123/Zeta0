#!/usr/bin/env python3
"""Validate bound external evaluation records and enforce quality gates.

This command does not run model inference. The producer of the JSONL records remains
an external evaluation step; every row must declare the exact immutable subject it
evaluated so unrelated or stale records fail closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import uuid

from psi.config.predictive_world import load_predictive_world_config
from psi.data.predictive_world.cache import file_sha256
from psi.data.predictive_world.evaluation import (
    EVALUATION_FORMAT,
    REQUIRED_VIEWS,
    aggregate_metrics,
    checkpoint_evaluation_binding,
    evaluation_release_from_index,
    evaluate_gates,
    fixed_sample,
    validate_passed_evaluation_report,
    validate_record_bindings,
    validate_release_record_ids,
)
from psi.models.predictive_world.checkpoint import predictive_world_code_fingerprint


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Exact completed action-policy bundle")
    parser.add_argument("--config", required=True, help="Exact resolved joint/policy config")
    parser.add_argument("--records", required=True)
    parser.add_argument("--gates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    config, payload = load_predictive_world_config(
        args.config, allow_existing_output_override=True
    )
    if config.training.phase not in {"joint", "policy"}:
        raise ValueError("quality evaluation requires a phase=joint or phase=policy config")
    expected_sample_ids, release = evaluation_release_from_index(
        config.data.index_path, split=args.split, count=args.samples, seed=args.seed
    )
    binding = checkpoint_evaluation_binding(
        args.checkpoint,
        args.config,
        config,
        payload,
        code_revision=_git_revision(),
        code_fingerprint=predictive_world_code_fingerprint(Path(__file__)),
        evaluation_release=release,
    )
    records_path = Path(args.records)
    records = [
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    validate_record_bindings(records, binding)
    validate_release_record_ids(records, expected_sample_ids)
    selected = fixed_sample(
        records, args.samples, args.seed, required_views=REQUIRED_VIEWS
    )
    metrics = aggregate_metrics(selected)
    gates_path = Path(args.gates)
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    if not isinstance(gates, list) or not gates:
        raise ValueError("quality gates must be a non-empty JSON list")
    gate_views = {
        str(gate.get("view")) for gate in gates if isinstance(gate, dict)
    }
    if not REQUIRED_VIEWS.issubset(gate_views):
        raise ValueError(
            f"quality gates must cover every required view: {sorted(REQUIRED_VIEWS)}"
        )
    outcomes = evaluate_gates(metrics, gates)
    report = {
        "format": EVALUATION_FORMAT,
        "evaluation_kind": "bound_external_records_only",
        "inference_generated_by_this_command": False,
        "evaluation_binding": binding,
        "artifact_hashes": {
            "records_sha256": file_sha256(records_path),
            "gates_sha256": file_sha256(gates_path),
        },
        "required_views": sorted(REQUIRED_VIEWS),
        "record_count": len(selected), "sample_count": args.samples,
        "seed": args.seed,
        "metrics": metrics, "gates": outcomes,
        "passed": all(item["passed"] for item in outcomes),
    }
    if report["passed"]:
        validate_passed_evaluation_report(report, binding)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise ValueError("immutable evaluation report already exists with different content")
    if not output.exists():
        temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({"event": "predictive_world_evaluation", **report}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
