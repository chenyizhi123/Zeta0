#!/usr/bin/env python3
"""Create a strict predictive-world bundle from an explicitly mapped legacy state."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from psi.config.predictive_world import load_predictive_world_config
from psi.data.predictive_world.cache import file_sha256
from psi.models.predictive_world.checkpoint import save_predictive_world_bundle
from psi.models.predictive_world.factory import build_phase
from psi.models.predictive_world.migration import load_tensor_file, migrate_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--completed-updates", type=int, default=0)
    args = parser.parse_args()
    config, _ = load_predictive_world_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built = build_phase(config, device=device)
    modules = built.checkpoint_modules
    phase = config.training.phase
    plan = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    report = migrate_state_dict(
        load_tensor_file(args.source), modules, plan["mapping"],
        allow_initialized_targets=frozenset(plan.get("allow_initialized_targets", ())),
        allow_unused_sources=frozenset(plan.get("allow_unused_sources", ())),
    )
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    migration_report = {
        "source": str(Path(args.source).resolve()),
        "source_sha256": file_sha256(Path(args.source)),
        "plan": str(Path(args.mapping).resolve()),
        **asdict(report),
    }
    metadata["migration"] = migration_report
    save_predictive_world_bundle(
        args.output, phase=phase, modules=modules,
        completed_updates=args.completed_updates, metadata=metadata,
        auxiliary_json={"migration_report.json": migration_report},
    )
    print(json.dumps({"event": "predictive_world_checkpoint_migrated", **asdict(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
