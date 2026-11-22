"""Explicit joint weight transfer with additive native protocol adapters.

Exact resume still uses checkpoint.py's strict loader. This migration preserves
every old tensor, remaps protocol embedding rows by name, and permits only new
native action/state adapters. No optimizer state is transferred.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.assets import resolve_asset_path
from psi.models.predictive_world.checkpoint import read_predictive_world_bundle, PHASE_MODULES


def exported_state_registry_digest(checkpoint, config_path, state_path, source):
    """Verify the portable export's checksum manifest instead of requiring its training dataset.

    A recognized export contains config/ and checkpoint directories under one
    SHA256SUMS.txt. A present but invalid manifest never falls back to an index.
    """
    root = Path(config_path).resolve().parent.parent
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        return None
    checksums = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or relative in checksums or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise ValueError("invalid exported source checksum manifest")
        checksums[relative] = digest
    for path in (Path(config_path), Path(checkpoint) / "predictive_world_checkpoint.json", Path(state_path)):
        relative = str(path.resolve().relative_to(root))
        if checksums.get(relative) != file_sha256(path):
            raise ValueError(f"exported source identity mismatch: {relative}")
    weight_relative = str((Path(checkpoint) / source["weights_file"]).resolve().relative_to(root))
    if checksums.get(weight_relative) != source["weights_sha256"]:
        raise ValueError("exported source manifest disagrees with checkpoint weights")
    return checksums[str(Path(state_path).resolve().relative_to(root))]


def read_initialization_source(checkpoint, config_path, *, asset_path_map=None):
    source = read_predictive_world_bundle(checkpoint)
    if source["phase"] != "joint":
        raise ValueError("joint initialization requires a complete joint checkpoint")
    payload = json.loads(Path(config_path).read_text())
    if canonical_sha256(payload) != source["metadata"]["resolved_config_sha256"]:
        raise ValueError("initialization source config identity mismatch")
    data = payload["data"]
    for field, metadata_key in (
        ("action_protocol_registry", "protocol_registry_sha256"),
        ("measurement_registry", "measurement_registry_sha256"),
    ):
        if file_sha256(resolve_asset_path(data[field], asset_path_map)) != source["metadata"][metadata_key]:
            raise ValueError(f"initialization source registry identity mismatch: {field}")
    state_path = resolve_asset_path(data["state_schema_registry"], asset_path_map)
    states_hash = file_sha256(state_path)
    expected_states_hash = exported_state_registry_digest(checkpoint, config_path, state_path, source)
    if expected_states_hash is None:
        meta = json.loads(resolve_asset_path(data["index_path"] + ".meta.json", asset_path_map).read_text())
        expected_states_hash = meta["provenance"]["registry_binding"]["state_schema_registry_sha256"]
    if states_hash != expected_states_hash:
        raise ValueError("initialization source state registry identity mismatch")
    registries = {name: json.loads(resolve_asset_path(data[name], asset_path_map).read_text()) for name in (
        "action_protocol_registry", "state_schema_registry", "measurement_registry",
    )}
    return source, payload, registries


def transfer_joint_tensors(tensors, modules, *, source_protocols, target_protocols,
                           source_states, target_states,
                           current_state_source_count=None):
    """Validate the complete transfer before changing any destination tensor."""
    if set(modules) != PHASE_MODULES["joint"]:
        raise ValueError("initialization requires every joint module, including student")
    for old, new in ((source_protocols, target_protocols), (source_states, target_states)):
        if any(name not in new or new[name] != spec for name, spec in old.items()):
            raise ValueError("additive initialization cannot remove/change existing registry entries")
    new_protocols = sorted(set(target_protocols) - set(source_protocols))
    new_states = sorted(set(target_states) - set(source_states))
    prefixes = []
    for protocol in new_protocols:
        prefixes += [f"{name}.adapters.{protocol}." for name in ("teacher", "student")]
        prefixes += [f"latent_transition.action_adapters.{protocol}."]
    for schema in new_states:
        prefixes += [f"{name}.state_adapters.adapters.{schema}." for name in ("teacher", "student")]
        prefixes += [f"history_state_adapters.adapters.{schema}."]
    target = {f"{name}.{key}": value for name, module in modules.items()
              for key, value in module.state_dict().items()}
    state_positions = {
        "history_encoder.state_history_position", "teacher.state_position", "student.state_position",
    }
    if current_state_source_count is not None:
        if type(current_state_source_count) is not int or current_state_source_count <= 1:
            raise ValueError("current-state transfer needs an explicit source history count > 1")
        for key in state_positions:
            if (key not in tensors or key not in target or target[key].ndim != 2
                    or target[key].shape[0] != 1
                    or tensors[key].shape != (current_state_source_count, target[key].shape[1])):
                raise ValueError(f"current-state position transfer mismatch: {key}")
    added = {key for key in target if any(key.startswith(prefix) for prefix in prefixes)}
    expected = set(target) - added
    if set(tensors) != expected:
        raise ValueError(f"joint transfer keys mismatch: missing={sorted(expected-set(tensors))[:8]}, "
                         f"unexpected={sorted(set(tensors)-expected)[:8]}")
    source_names, target_names = sorted(source_protocols), sorted(target_protocols)
    for key in expected:
        value = tensors[key]
        if key in {"teacher.protocol_embedding.weight", "student.protocol_embedding.weight", "latent_transition.protocol_embedding.weight"}:
            if value.shape != (len(source_names), target[key].shape[1]):
                raise ValueError(f"joint protocol embedding shape mismatch: {key}")
            expanded = target[key].clone()
            for row, name in enumerate(source_names):
                expanded[target_names.index(name)].copy_(value[row])
            target[key] = expanded
        else:
            if current_state_source_count is not None and key in state_positions:
                value = value[-1:].clone()
            if value.shape != target[key].shape:
                raise ValueError(f"joint transfer shape mismatch: {key}")
            target[key] = value
    from psi.models.predictive_world.training import tensor_mapping_sha256
    added_initialization = {key: target[key] for key in added}
    for name in ("teacher", "student", "latent_transition"):
        key = f"{name}.protocol_embedding.weight"
        for protocol in new_protocols:
            added_initialization[f"{key}:{protocol}"] = target[key][target_names.index(protocol)]
    for name, module in modules.items():
        prefix = name + "."
        module.load_state_dict({key[len(prefix):]: value for key, value in target.items()
                                if key.startswith(prefix)}, strict=True)
    return {"new_protocols": new_protocols, "new_state_schemas": new_states,
            "goal_intent_added": False,
            "new_adapter_initialization_sha256": tensor_mapping_sha256(added_initialization),
            "new_tensor_keys": sorted(added), "transferred_tensor_count": len(expected),
            "new_protocol_embedding_rows": {name: new_protocols for name in ("teacher", "student", "latent_transition")}}


def initialize_joint(config, modules, expected_metadata, *, allow_diagnostic_migration=False):
    from safetensors.torch import load_file
    source, payload, registries = read_initialization_source(
        config.training.init_joint, config.training.init_source_config,
        asset_path_map=config.runtime.asset_path_map)
    metadata = source["metadata"]
    raw_migration = getattr(config.training, "init_latent_prediction_migration", None)
    source_space = payload.get("training", {}).get("latent_prediction_space", "whitened")
    target_space = getattr(config.training, "latent_prediction_space", "whitened")
    if source_space != metadata.get("loss_contract", {}).get("latent_prediction_space", "whitened"):
        raise ValueError("joint source latent prediction space disagrees with its loss contract")
    if source_space != target_space and not (
        source_space == "whitened" and target_space == "raw" and raw_migration is not None
    ):
        raise ValueError("joint latent prediction space change requires an explicit coordinate migration")
    if raw_migration is not None:
        from psi.models.predictive_world.posttrain_transition import verify_raw_latent_transition
        expected_metadata = verify_raw_latent_transition(raw_migration, source, payload, expected_metadata)
    posttrain_migration = getattr(config.training, "init_posttrain_migration", None)
    if posttrain_migration is not None:
        if raw_migration is not None and posttrain_migration.get("calibration_batches") != 0:
            raise ValueError("raw latent migration cannot fit post-training statistics")
        from psi.models.predictive_world.posttrain_transition import verify_posttrain_transition
        expected_metadata = verify_posttrain_transition(
            posttrain_migration, source, payload, expected_metadata)
    diagnostic = config.training.init_processor_diagnostic
    migration = config.training.init_processor_migration
    objective_migration = config.training.init_objective_migration
    state_migration = config.training.init_state_history_migration
    observation_migration = getattr(config.training, "init_observation_contract_migration", None)
    vlm_action_migration = getattr(config.training, "init_vlm_action_migration", None)
    state_target_temporal = expected_metadata["temporal_abi"]
    if vlm_action_migration is not None:
        from psi.models.predictive_world.vlm_action_transition import verify_vlm_action_transition
        vlm_action_migration = verify_vlm_action_transition(
            vlm_action_migration, source, payload, expected_metadata,
            processor_migration=migration, source_config=config.training.init_source_config)
        state_target_temporal = dict(state_target_temporal)
        for key in ("action_horizon", "action_hz"):
            state_target_temporal[key] = metadata["temporal_abi"][key]
        if (state_target_temporal["state_history_count"] != metadata["temporal_abi"]["state_history_count"]
                and state_migration is None):
            raise ValueError("VLM/action migration also requires explicit current-state migration")
    if observation_migration is not None:
        from psi.models.predictive_world.observation_contract_transition import (
            REAL_TEMPORAL_FIELDS, verify_observation_contract_transition,
        )
        observation_migration = verify_observation_contract_transition(
            observation_migration, source, payload, expected_metadata["temporal_abi"], target_data=config.data)
        if (metadata["temporal_abi"]["state_history_count"] != expected_metadata["temporal_abi"]["state_history_count"]
                and state_migration is None):
            raise ValueError("observation migration also requires the explicit current-state position migration")
        state_target_temporal = {
            key: value for key, value in state_target_temporal.items() if key not in REAL_TEMPORAL_FIELDS
        }
    if state_migration is not None:
        from psi.models.predictive_world.state_history_transition import verify_current_state_transition
        state_migration = verify_current_state_transition(
            state_migration, source, payload, state_target_temporal)
    if objective_migration is not None:
        from psi.models.predictive_world.objective_transition import verify_objective_transition
        objective_migration = verify_objective_transition(
            objective_migration, source, payload, config.training
        )
        target_loss_contract = dict(objective_migration["target_loss_contract"])
        if raw_migration is not None:
            target_loss_contract.pop("latent_prediction_space", None)
        if target_loss_contract != expected_metadata["loss_contract"]:
            raise ValueError("objective transition target metadata mismatch")
    if migration is not None:
        from psi.data.predictive_world.processor_migration import verify_processor_migration
        migration = verify_processor_migration(
            migration, source, expected_metadata['processor_id'], config.training.init_source_config)
    if diagnostic is not None:
        if not allow_diagnostic_migration:
            raise ValueError('processor transition requires explicit diagnostic execution')
        from psi.data.predictive_world.processor_diagnostic import verify_processor_diagnostic
        diagnostic = verify_processor_diagnostic(
            diagnostic, source, expected_metadata['processor_id'], config.training.init_source_config)
    for key in ("architecture", "temporal_abi", "encoder_id", "processor_id", "loss_contract"):
        if vlm_action_migration is not None and key in {"architecture", "temporal_abi", "encoder_id", "processor_id"}:
            continue  # Exact source/target and the only permitted deltas were checked above.
        if key == "temporal_abi" and (state_migration is not None or observation_migration is not None):
            continue  # The exact source-bound history -> current-state delta was checked above.
        if key == 'processor_id' and (diagnostic is not None or migration is not None):
            continue  # This exact source->target pair was checked above, no wildcard.
        if key == "loss_contract" and objective_migration is not None:
            continue  # Only the source-bound declared objective delta was checked above.
        if metadata.get(key) != expected_metadata[key]:
            raise ValueError(f"joint initialization metadata mismatch: {key}")
    if metadata["optimization_contract"]["target_encoder_ema_decay"] != config.training.target_encoder_ema_decay:
        raise ValueError("joint initialization EMA decay mismatch")
    target = {name: json.loads(Path(getattr(config.data, name)).read_text())
              for name in registries}
    if any(target["measurement_registry"].get(key) != value
           for key, value in registries["measurement_registry"].items()):
        raise ValueError("joint initialization cannot change existing body schemas")
    tensors = load_file(str(Path(config.training.init_joint) / source["weights_file"]), device="cpu")
    if raw_migration is not None:
        from psi.models.predictive_world.posttrain_transition import migrate_raw_latent_tensors
        tensors = migrate_raw_latent_tensors(
            tensors, modules["latent_transition"],
            source_output_affine=metadata["architecture"]["latent_transition"].get("output_affine", False),
        )
    elif posttrain_migration is not None:
        from psi.models.predictive_world.posttrain_transition import add_output_coordinate_tensors
        tensors = add_output_coordinate_tensors(tensors, modules["latent_transition"])
    if vlm_action_migration is not None:
        from psi.models.predictive_world.vlm_action_transition import migrate_action_position_tensors
        tensors = migrate_action_position_tensors(tensors, vlm_action_migration)
    body_migration = config.training.init_body_target_migration
    if body_migration is not None:
        from psi.models.predictive_world.body_target_transition import (
            migrate_body_target_tensors, verify_body_target_transition,
        )
        body_migration = verify_body_target_transition(
            body_migration, source, payload, target["measurement_registry"],
            registries["measurement_registry"],
        )
        tensors = migrate_body_target_tensors(tensors)
    report = transfer_joint_tensors(
        tensors, modules,
        source_protocols=registries["action_protocol_registry"],
        target_protocols=target["action_protocol_registry"],
        source_states=registries["state_schema_registry"],
        target_states=target["state_schema_registry"],
        current_state_source_count=(None if state_migration is None
                                    else state_migration["source_state_history_count"]),
    )
    from psi.models.predictive_world.training import module_state_sha256
    if module_state_sha256(modules["normalizer"]) != metadata["normalization_id"]:
        raise ValueError("joint initialization changed the fitted normalizer identity")
    encoder = modules["encoder"]
    if (getattr(config.training, "latent_prediction_space", "whitened") == "whitened"
            and (not bool(modules["normalizer"].fitted) or not bool(encoder.normalizer_frozen))):
        raise ValueError("joint initialization requires a fitted, frozen normalizer")
    offset = int(encoder.last_target_update) + 1
    if offset <= 0:
        raise ValueError("joint initialization lacks a valid EMA update history")
    report.update({
        "format": "zeta0.predictive_world.joint_initialization.v1",
        "path": str(Path(config.training.init_joint).resolve()),
        "phase": "joint", "completed_updates": source["completed_updates"],
        "weights_sha256": source["weights_sha256"],
        "reference_id": metadata["reference_id"],
        "normalization_id": metadata["normalization_id"],
        "source_config_sha256": canonical_sha256(payload),
        "optimizer_update_offset": offset, "optimizer_restored": False,
    })
    if diagnostic is not None:
        report['processor_diagnostic'] = diagnostic
    if migration is not None:
        report['processor_migration'] = migration
    if objective_migration is not None:
        report['objective_migration'] = objective_migration
    if state_migration is not None:
        report["state_history_migration"] = state_migration
    if observation_migration is not None:
        report["observation_contract_migration"] = observation_migration
    if body_migration is not None:
        report["body_target_migration"] = body_migration
    if vlm_action_migration is not None:
        report["vlm_action_migration"] = vlm_action_migration
    if posttrain_migration is not None:
        report["posttrain_migration"] = posttrain_migration
    if raw_migration is not None:
        report["latent_prediction_migration"] = raw_migration
    parent = metadata.get("parent_world_model")
    if parent is None:
        # The strict bundle reader already validated this direct-joint ancestry.
        # Preserve it verbatim instead of inventing a world-model checkpoint.
        report["source_joint_provenance"] = {
            "phase": "joint", "completed_updates": source["completed_updates"],
            "weights_sha256": source["weights_sha256"],
            "resolved_config_sha256": metadata["resolved_config_sha256"],
            "metadata": {key: metadata[key] for key in
                         ("training_origin", "optimization_contract", "initialization") if key in metadata},
        }
    return report, parent
