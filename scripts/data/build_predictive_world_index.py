#!/usr/bin/env python3
"""Build an immutable multi-time predictive-world index from audited episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psi.data.predictive_world.index import (
    build_episode_windows,
    predictive_world_temporal_abi,
    publish_predictive_world_index,
    verify_predictive_world_index,
)
from psi.data.predictive_world.cache import file_sha256
from psi.data.predictive_world.measurement_schema import (
    load_capability_registry,
    load_measurement_registry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, help="Audited episode JSONL")
    parser.add_argument("--config", required=True, help="Index semantics/provenance JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--excluded-episodes", help="Optional JSON array")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.verify_only:
        print(json.dumps(verify_predictive_world_index(args.output), sort_keys=True))
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    required = {
        "history_visual_offsets_s", "state_history_count", "state_query_hz",
        "future_group_ends_s", "action_horizon", "action_hz", "visual_max_age_s",
        "state_max_age_s", "action_max_age_s", "provenance",
        "measurement_registry", "canonical_future_capabilities",
        "action_protocol_registry", "state_schema_registry",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"predictive-world index config is missing: {missing}")
    registry_paths = {
        "measurement_registry_sha256": Path(config["measurement_registry"]).expanduser().resolve(),
        "capability_registry_sha256": Path(
            config["canonical_future_capabilities"]
        ).expanduser().resolve(),
        "action_protocol_registry_sha256": Path(
            config["action_protocol_registry"]
        ).expanduser().resolve(),
        "state_schema_registry_sha256": Path(
            config["state_schema_registry"]
        ).expanduser().resolve(),
    }
    if any(not path.is_file() for path in registry_paths.values()):
        missing_paths = sorted(str(path) for path in registry_paths.values() if not path.is_file())
        raise ValueError(f"predictive-world index registry files are missing: {missing_paths}")
    measurements = load_measurement_registry(registry_paths["measurement_registry_sha256"])
    protocols = json.loads(
        registry_paths["action_protocol_registry_sha256"].read_text(encoding="utf-8")
    )
    if not isinstance(protocols, dict) or not protocols:
        raise ValueError("action protocol registry must be a non-empty JSON object")
    capabilities = load_capability_registry(
        registry_paths["capability_registry_sha256"],
        measurement_schemas=measurements,
        action_protocol_ids=set(map(str, protocols)),
    )
    temporal_abi = predictive_world_temporal_abi(
        history_visual_offsets_s=config["history_visual_offsets_s"],
        state_history_count=int(config["state_history_count"]),
        state_query_hz=float(config["state_query_hz"]),
        future_group_ends_s=config["future_group_ends_s"],
        action_horizon=int(config["action_horizon"]),
        action_hz=float(config["action_hz"]),
        visual_max_age_s=float(config["visual_max_age_s"]),
        state_max_age_s=float(config["state_max_age_s"]),
        action_max_age_s=float(config["action_max_age_s"]),
    )
    state_offsets = temporal_abi["state_offsets_s"]
    action_offsets = temporal_abi["action_offsets_s"]
    future_measurement_offsets = temporal_abi["future_measurement_offsets_s"]
    rows = []
    for line_number, line in enumerate(Path(args.episodes).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            episode = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid episode JSON at line {line_number}") from error
        source_id = str(episode.get("source_id", ""))
        if source_id not in capabilities:
            raise ValueError(
                f"episode source {source_id!r} at line {line_number} is absent from the "
                "capability registry"
            )
        capability = capabilities[source_id]
        rows.extend(build_episode_windows(
            episode,
            history_visual_offsets_s=config["history_visual_offsets_s"],
            state_offsets_s=state_offsets,
            future_group_ends_s=config["future_group_ends_s"],
            action_offsets_s=action_offsets,
            future_measurement_offsets_s=future_measurement_offsets,
            visual_max_age_s=float(config["visual_max_age_s"]),
            state_max_age_s=float(config["state_max_age_s"]),
            action_max_age_s=float(config["action_max_age_s"]),
            capability=capability,
            measurement_schema=measurements[capability.measurement_schema_id],
        ))
    excluded = set()
    if args.excluded_episodes:
        payload = json.loads(Path(args.excluded_episodes).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("excluded episode file must be a JSON array")
        excluded = {str(value) for value in payload}
    provenance = dict(config["provenance"])
    binding = {name: file_sha256(path) for name, path in registry_paths.items()}
    if "registry_binding" in provenance and provenance["registry_binding"] != binding:
        raise ValueError("index provenance registry_binding disagrees with registry files")
    provenance["registry_binding"] = binding
    index, meta = publish_predictive_world_index(
        args.output, rows=rows, provenance=provenance, temporal_abi=temporal_abi,
        excluded_episode_ids=excluded,
    )
    print(json.dumps({"event": "predictive_world_index_published", "index": str(index), "meta": str(meta), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
