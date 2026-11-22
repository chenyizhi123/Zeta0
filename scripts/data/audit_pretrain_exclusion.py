#!/usr/bin/env python3
"""Fail closed when a proposed Stage-A manifest overlaps downstream data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psi.data.predictor_pretrain.registry import (
    DEFAULT_FORBIDDEN_PT_SOURCES,
    audit_cross_stage_overlap,
    iter_manifest_rows,
    load_manifest_rows,
)


DEFAULT_FT_ROOT = str(Path(__file__).resolve().parents[2] / "data/humanoidarena")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-manifest", required=True)
    parser.add_argument("--downstream-exclusion-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--forbidden-root", action="append", default=[DEFAULT_FT_ROOT])
    parser.add_argument("--forbidden-source", action="append", default=[])
    parser.add_argument(
        "--skip-duplicate-sample-check", action="store_true",
        help="Skip the audit's second SQLite identity set when a concurrently built "
             "anchor index enforces the same primary-key invariant.",
    )
    parser.add_argument("--progress-every-rows", type=int, default=0)
    args = parser.parse_args()

    kwargs = {"forbidden_pt_roots": args.forbidden_root}
    kwargs["forbidden_pt_sources"] = (
        *DEFAULT_FORBIDDEN_PT_SOURCES,
        *args.forbidden_source,
    )
    report = audit_cross_stage_overlap(
        iter_manifest_rows(args.pretrain_manifest),
        load_manifest_rows(args.downstream_exclusion_manifest),
        **kwargs,
        check_duplicate_sample_ids=not args.skip_duplicate_sample_check,
        progress_every_rows=args.progress_every_rows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    if not report.passed:
        raise SystemExit(f"pretraining exclusion audit failed with {len(report.findings)} finding(s)")


if __name__ == "__main__":
    main()
