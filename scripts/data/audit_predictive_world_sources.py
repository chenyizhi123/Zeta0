#!/usr/bin/env python3
"""Cross-check source capabilities, physical schemas, protocols and index coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.index import verify_predictive_world_index
from psi.data.predictive_world.measurement_schema import (
    canonical_body_target_signature,
    canonical_body_target_width,
    load_capability_registry,
    load_measurement_registry,
    validate_predictive_world_row_capability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-registry", required=True)
    parser.add_argument("--capability-registry", required=True)
    parser.add_argument("--action-protocol-registry", required=True)
    parser.add_argument("--state-schema-registry", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    measurements = load_measurement_registry(args.measurement_registry)
    protocols = json.loads(Path(args.action_protocol_registry).read_text(encoding="utf-8"))
    state_schemas = json.loads(Path(args.state_schema_registry).read_text(encoding="utf-8"))
    if not isinstance(protocols, dict) or not protocols:
        raise ValueError("action protocol registry must be a non-empty flat JSON object")
    if not isinstance(state_schemas, dict) or not state_schemas:
        raise ValueError("state schema registry must be a non-empty flat JSON object")
    for name, spec in protocols.items():
        dim = int(spec["dim"])
        if (
            len(spec["mean"]) != dim or len(spec["std"]) != dim
            or dim <= 0
            or not all(math.isfinite(float(value)) for value in spec["mean"])
            or not all(math.isfinite(float(value)) and float(value) > 0 for value in spec["std"])
            or not spec.get("normalization_space")
            or "REQUIRED_" in str(spec.get("normalization_space"))
        ):
            raise ValueError(f"action protocol {name!r} lacks exact statistics/space")
    capabilities = load_capability_registry(
        args.capability_registry, measurement_schemas=measurements,
        action_protocol_ids=set(protocols),
    )
    index_meta = verify_predictive_world_index(args.index)
    sources: set[str] = set()
    row_count = 0
    with Path(args.index).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_count += 1
            sources.add(str(row["source_id"]))
            capability, _ = validate_predictive_world_row_capability(
                row, capabilities=capabilities, measurement_schemas=measurements
            )
            materialization = row["materialization"]
            protocol = protocols[capability.action_protocol_id]
            if str(materialization.get("action_normalization_space")) != str(
                protocol["normalization_space"]
            ):
                raise ValueError(
                    "index action normalization space disagrees with action protocol registry"
                )
    if sources != set(capabilities):
        raise ValueError(
            f"index/capability source mismatch: index_only={sorted(sources-set(capabilities))}, "
            f"registry_only={sorted(set(capabilities)-sources)}"
        )
    registry_binding = {
        "measurement_registry_sha256": file_sha256(Path(args.measurement_registry)),
        "capability_registry_sha256": file_sha256(Path(args.capability_registry)),
        "action_protocol_registry_sha256": file_sha256(Path(args.action_protocol_registry)),
        "state_schema_registry_sha256": file_sha256(Path(args.state_schema_registry)),
    }
    indexed_binding = index_meta.get("provenance", {}).get("registry_binding")
    if not isinstance(indexed_binding, dict):
        raise ValueError("predictive-world index has no immutable registry binding")
    for key, digest in registry_binding.items():
        if indexed_binding.get(key) != digest:
            raise ValueError(f"predictive-world index registry binding mismatch: {key}")
    signature = canonical_body_target_signature(capabilities, measurements)
    report = {
        "format": "zeta0.predictive_world.source_audit.v1",
        "sources": sorted(sources), "rows": row_count,
        "world_action_sources": sorted(
            name for name, value in capabilities.items() if value.supports_world_action
        ),
        "index_identity": index_meta["semantic_identity"],
        **registry_binding,
        "canonical_body_target_width": canonical_body_target_width(signature),
        "canonical_body_target_signature_sha256": canonical_sha256(
            {"fields": list(signature)}
        ),
        "shared_visual_index_sources": sorted(
            name
            for name, capability in capabilities.items()
            if capability.shared_frame_index_across_visual_keys
        ),
        "independent_visual_time_sources": sorted(
            name
            for name, capability in capabilities.items()
            if capability.independent_timestamps_across_visual_keys
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != report:
        raise ValueError("immutable source audit already exists with different content")
    if not output.exists():
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "predictive_world_source_audit_passed", **report}, sort_keys=True))


if __name__ == "__main__":
    main()
