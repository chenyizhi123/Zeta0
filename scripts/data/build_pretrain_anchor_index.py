#!/usr/bin/env python3
"""Build the SQLite anchor index, optionally sharing its scan with leakage audit."""

import argparse
import json
from pathlib import Path

from psi.data.predictor_pretrain.registry import (
    CrossStageAuditAccumulator,
    DEFAULT_FORBIDDEN_PT_SOURCES,
    load_manifest_rows,
)
from psi.data.predictor_pretrain.runtime import build_anchor_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-every-rows", type=int, default=0)
    parser.add_argument("--audit-downstream-exclusion-manifest")
    parser.add_argument("--audit-output")
    parser.add_argument("--forbidden-root", action="append", default=[])
    parser.add_argument("--forbidden-source", action="append", default=[])
    args = parser.parse_args()
    if bool(args.audit_downstream_exclusion_manifest) != bool(args.audit_output):
        raise ValueError("audit manifest and output must be provided together")
    auditor = None
    if args.audit_output:
        auditor = CrossStageAuditAccumulator(
            load_manifest_rows(args.audit_downstream_exclusion_manifest),
            forbidden_pt_roots=args.forbidden_root,
            forbidden_pt_sources=(
                *DEFAULT_FORBIDDEN_PT_SOURCES, *args.forbidden_source,
            ),
            # The anchors table has a PRIMARY KEY on sample_id and therefore
            # supplies the same fail-closed duplicate invariant in this fused pass.
            check_duplicate_sample_ids=False,
            # The index ingest reports the same shared scan; avoid duplicate lines.
            progress_every_rows=0,
        )
    try:
        result = build_anchor_index(
            args.anchor_manifest, args.output,
            progress_every_rows=args.progress_every_rows,
            row_observer=auditor.add if auditor is not None else None,
        )
        if auditor is not None:
            report = auditor.finish()
            output = Path(args.audit_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
            temporary.replace(output)
            if not report.passed:
                raise SystemExit(
                    f"pretraining exclusion audit failed with {len(report.findings)} finding(s)"
                )
        print(result)
    finally:
        if auditor is not None:
            auditor.close()


if __name__ == "__main__":
    main()
