"""Prepare a fresh N1.7 run from an audited Zeta joint data release.

Only the action horizon changes. Causal observations, future queries, task
sampling, train-only normalization and episode splits retain their identities.
Frozen features must subsequently be computed with the exported N1.7 backbone.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

from psi.config.predictive_world import PredictiveWorldDataConfig, predictive_world_config_from_payload
from psi.data.predictive_world.action_labels import COMPONENT_FIELDS
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.index import (
    predictive_world_index_binding, publish_predictive_world_index,
    temporal_abi_from_data_config, verify_predictive_world_index,
)
from psi.data.predictive_world.sonic_joint_training import JOINT_DATA_FORMAT, immutable_json
from psi.models.predictive_world.gr00t_assets import verify_backbone_export


def shorten_action_window(source, *, source_horizon, target_horizon=40):
    """Retain native action samples; never pad or resample a shorter source."""
    if not 0 < target_horizon <= source_horizon:
        raise ValueError("source action horizon is shorter than N1.7")
    row = copy.deepcopy(source)
    for field in ("action_indices", "action_valid", "action_actual_times"):
        if len(row[field]) != source_horizon:
            raise ValueError("source action window disagrees with its temporal ABI")
        row[field] = row[field][:target_horizon]
    for field in COMPONENT_FIELDS:
        if field in row:
            if any(len(value) != source_horizon for value in row[field].values()):
                raise ValueError("source component window disagrees with its temporal ABI")
            row[field] = {key: value[:target_horizon] for key, value in row[field].items()}
    row.pop("eligibility", None)  # The index publisher recomputes it from masks.
    return row


def prepare_training(source_data, template, *, base, backbone, output, run_output):
    """Publish immutable config/index/provenance, without building features or training."""
    from psi.data.predictive_world.provenance import load_qwen_processor
    from psi.data.lerobot.action_aligned_cache import build_action_aligned_vlm_processor_fingerprint
    from psi.data.predictor_pretrain.runtime import fingerprint_model_directory

    source_data, base, backbone, output, run_output = (
        Path(p).expanduser().resolve() for p in (source_data, base, backbone, output, run_output))
    if output.exists() or run_output.exists():
        raise FileExistsError("preparation and training output directories must be new")
    source = json.loads(source_data.read_text())
    if source.get("format") != JOINT_DATA_FORMAT:
        raise ValueError("expected an audited Zeta joint_data.json release")
    if source.get("identity") != canonical_sha256({key: value for key, value in source.items() if key != "identity"}):
        raise ValueError("source joint release identity mismatch")
    data = PredictiveWorldDataConfig(**source["data"])
    if (data.action_horizon < 40 or data.action_hz != 50 or data.state_history_count != 1
            or data.observed_visual_mode != "current_multiview" or data.source_sampling_exponent != 0):
        raise ValueError("N1.7 preparation requires current RGB/state and task-uniform SONIC at 50 Hz")
    if data.current_visual_pooling != "content_area_v1":
        raise ValueError("N1.7 preparation requires explicit content_area_v1 observations")
    meta = verify_predictive_world_index(data.index_path)
    if meta["temporal_abi"] != temporal_abi_from_data_config(data):
        raise ValueError("source data and index temporal ABI disagree")
    names = {"measurement_registry": "measurement_registry",
             "canonical_future_capabilities": "capability_registry",
             "action_protocol_registry": "action_protocol_registry", "state_schema_registry": "state_schema_registry"}
    binding = {name + "_sha256": file_sha256(Path(getattr(data, field))) for field, name in names.items()}
    if meta["provenance"].get("registry_binding") != binding:
        raise ValueError("source index registry identity mismatch")
    registries = {field: json.loads(Path(getattr(data, field)).read_text()) for field in names}
    capabilities = registries["canonical_future_capabilities"]
    protocol_ids = {value["action_protocol_id"] for value in capabilities.values()}
    measurement_ids = {value["measurement_schema_id"] for value in capabilities.values()}
    state_id = source.get("state_schema_id")
    if len(protocol_ids) != 1 or state_id not in registries["state_schema_registry"]:
        raise ValueError("N1.7 v1 requires one shared action protocol and an explicit state schema")
    registries["action_protocol_registry"] = {key: registries["action_protocol_registry"][key] for key in protocol_ids}
    registries["state_schema_registry"] = {state_id: registries["state_schema_registry"][state_id]}
    registries["measurement_registry"] = {key: registries["measurement_registry"][key] for key in measurement_ids}
    base_sha = file_sha256(base / "config.json")
    export = verify_backbone_export(backbone, base, expected_config_sha256=base_sha)
    processor = load_qwen_processor(backbone)
    fingerprint = build_action_aligned_vlm_processor_fingerprint(processor)
    provenance = dict(encoder_id="qwen3vl:" + fingerprint_model_directory(backbone),
        processor_id="qwen3vl-processor:" + canonical_sha256(fingerprint),
        image_preprocessing="aspect_v1", feature_layer="post_merger_content_area_overlap_4x4_v1",
        dtype="float16", feature_shape=[16, 2048], video_decoder="torchcodec", gr00t_source=export)
    output.mkdir(parents=True)
    registry_paths = {}
    for field, name in names.items():
        registry_paths[field] = str(output / (name + ".json"))
        immutable_json(registry_paths[field], registries[field])
    target_binding = {name + "_sha256": file_sha256(Path(registry_paths[field])) for field, name in names.items()}
    target = replace(data, **registry_paths, action_horizon=40,
        index_path=str(output / "predictive_world_index.jsonl"), feature_cache=str(output / "feature_cache"))
    source_binding = predictive_world_index_binding(data.index_path)

    def rows():
        with Path(data.index_path).open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    material = row["materialization"]
                    if (material["state_schema_id"] != state_id
                            or material["action_protocol_id"] not in protocol_ids
                            or material["measurement_schema_id"] not in measurement_ids):
                        raise ValueError("source row uses a different state/action/measurement schema")
                    yield shorten_action_window(row, source_horizon=data.action_horizon)

    publish_predictive_world_index(target.index_path, rows=rows(), presorted=True,
        temporal_abi=temporal_abi_from_data_config(target),
        provenance={**meta["provenance"], "registry_binding": target_binding, "gr00t_action_window": {
            "source_index": source_binding, "source_release_sha256": file_sha256(source_data),
            "operation": "retain_first_40_native_actions", "future_queries_unchanged": True}})
    payload = copy.deepcopy(template)
    payload["data"] = asdict(target)
    payload["model"].update(gr00t_model=str(base), qwen_model=str(backbone), gr00t_config_sha256=base_sha)
    payload["runtime"]["output_dir"] = str(run_output)
    config, _ = predictive_world_config_from_payload(payload)
    immutable_json(output / "feature_provenance.json", provenance)
    immutable_json(output / "train.json", asdict(config))
    report = dict(format="zeta0.gr00t_n1d7.preparation.v1", training_started=False,
        feature_cache_built=False, source_release_sha256=file_sha256(source_data),
        index_binding=predictive_world_index_binding(target.index_path),
        config=str(output / "train.json"), feature_provenance=str(output / "feature_provenance.json"))
    immutable_json(output / "preparation.json", report)
    return report
