#!/usr/bin/env python3
"""Prepare an N1.7 config/index and fresh feature-cache identity; never launch training."""
import argparse
import json
from pathlib import Path

from psi.data.predictive_world.gr00t_preparation import prepare_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-data", "base", "backbone", "output", "run-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--template", type=Path,
        default=Path(__file__).resolve().parents[3] / "configs/gr00t_n1d7/joint.template.json")
    args = parser.parse_args()
    print(json.dumps(prepare_training(args.source_data, json.loads(args.template.read_text()),
        base=args.base, backbone=args.backbone, output=args.output, run_output=args.run_output)))


if __name__ == "__main__":
    main()
