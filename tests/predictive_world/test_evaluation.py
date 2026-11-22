from dataclasses import asdict
import json

import pytest
from torch import nn

from psi.config.predictive_world import (
    PredictiveWorldConfig,
    PredictiveWorldDataConfig,
    PredictiveWorldModelConfig,
    PredictiveWorldRuntimeConfig,
    PredictiveWorldTrainingConfig,
    predictive_world_architecture_contract,
)
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.evaluation import (
    EVALUATION_FORMAT,
    REQUIRED_VIEWS,
    aggregate_metrics,
    checkpoint_evaluation_binding,
    evaluate_gates,
    fixed_sample,
    validate_passed_evaluation_report,
    validate_record_bindings,
    validate_release_record_ids,
)
from psi.models.predictive_world.checkpoint import save_predictive_world_bundle


def test_fixed_evaluation_and_quality_gates_are_deterministic():
    rows = [
        {"sample_id": str(index), "view": "prior", "metrics": {"latent_mse": index / 10}}
        for index in range(10)
    ]
    first = fixed_sample(rows, 5, 101)
    assert first == fixed_sample(list(reversed(rows)), 5, 101)
    metrics = aggregate_metrics(first)
    result = evaluate_gates(metrics, [{"view": "prior", "metric": "latent_mse.count", "min": 5}])
    assert result[0]["passed"]


def test_evaluation_rejects_duplicates_and_missing_gate_metrics():
    row = {"sample_id": "a", "view": "prior", "metrics": {"mse": 1.0}}
    with pytest.raises(ValueError, match="duplicate"):
        fixed_sample([row, row], 2, 0)
    with pytest.raises(KeyError, match="missing"):
        evaluate_gates(aggregate_metrics([row]), [{"view": "world", "metric": "mse.mean", "max": 1}])


def test_release_sampling_requires_every_view_for_each_selected_sample():
    rows = [
        {"sample_id": sample, "view": view, "metrics": {"mse": float(index)}}
        for index, sample in enumerate(("a", "b", "c"))
        for view in REQUIRED_VIEWS
    ]
    selected = fixed_sample(rows, 2, 7, required_views=REQUIRED_VIEWS)
    assert len(selected) == 2 * len(REQUIRED_VIEWS)
    assert {row["view"] for row in selected} == REQUIRED_VIEWS
    incomplete = [row for row in rows if not (row["sample_id"] == "a" and row["view"] == "all")]
    with pytest.raises(ValueError, match="lack required views"):
        fixed_sample(incomplete, 2, 7, required_views=REQUIRED_VIEWS)
    with pytest.raises(ValueError, match="requires 4 complete samples"):
        fixed_sample(rows, 4, 7, required_views=REQUIRED_VIEWS)


def _passed_report(binding):
    metrics = {
        view: {"score.mean": 0.5, "score.count": 2.0}
        for view in REQUIRED_VIEWS
    }
    gates = [
        {"view": view, "metric": "score.mean", "max": 1.0, "value": 0.5, "passed": True}
        for view in sorted(REQUIRED_VIEWS)
    ]
    return {
        "format": EVALUATION_FORMAT,
        "evaluation_binding": dict(binding),
        "required_views": sorted(REQUIRED_VIEWS),
        "sample_count": 2,
        "record_count": 2 * len(REQUIRED_VIEWS),
        "metrics": metrics,
        "gates": gates,
        "passed": True,
    }


def test_records_and_export_report_are_bound_to_the_exact_subject():
    binding = {
        "checkpoint_identity": "checkpoint-a",
        "resolved_config_sha256": "config-a",
        "data_fingerprint": "data-a",
        "code_revision": "code-a",
        "code_fingerprint": "source-a",
        "evaluation_release": {
            "format": "zeta0.predictive_world.evaluation_release.v1",
            "index_sha256": "index-a",
            "split": "validation",
            "seed": 101,
            "sample_count": 2,
            "eligibility": ["policy", "world"],
            "sample_ids_sha256": "release-a",
        },
    }
    rows = [{"sample_id": "a", "view": "all", "evaluation_binding": binding}]
    validate_record_bindings(rows, binding)
    with pytest.raises(ValueError, match="unbound or stale"):
        validate_record_bindings(rows, {**binding, "checkpoint_identity": "checkpoint-b"})
    report = _passed_report(binding)
    validate_passed_evaluation_report(report, binding)
    stale = {**report, "evaluation_binding": {**binding, "code_revision": "code-b"}}
    with pytest.raises(ValueError, match="exact policy bundle"):
        validate_passed_evaluation_report(stale, binding)
    forged = {**report, "passed": False}
    with pytest.raises(ValueError, match="did not pass"):
        validate_passed_evaluation_report(forged, binding)
    missing_view_gate = {**report, "gates": report["gates"][:-1]}
    with pytest.raises(ValueError, match="cover every required view"):
        validate_passed_evaluation_report(missing_view_gate, binding)


def test_release_records_must_match_the_exact_sample_set():
    rows = [{"sample_id": sample} for sample in ("a", "b") for _ in range(4)]
    validate_release_record_ids(rows, ("a", "b"))
    with pytest.raises(ValueError, match="immutable release"):
        validate_release_record_ids(rows, ("a", "c"))


def test_checkpoint_binding_validates_resolved_config_data_and_code(tmp_path):
    files = {}
    for name in ("index", "capabilities", "actions", "measurements", "states"):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(f"{name}\n", encoding="utf-8")
    feature_manifest = tmp_path / "feature_manifest.json"
    feature_manifest.write_text(json.dumps({
        "cache_identity": "feature-cache-a",
        "provenance": {"encoder_id": "encoder-a", "processor_id": "processor-a"},
    }), encoding="utf-8")
    config = PredictiveWorldConfig(
        data=PredictiveWorldDataConfig(
            measurement_registry=str(files["measurements"]),
            canonical_future_capabilities=str(files["capabilities"]),
            action_protocol_registry=str(files["actions"]),
            state_schema_registry=str(files["states"]),
            index_path=str(files["index"]),
            feature_cache=str(feature_manifest),
        ),
        model=PredictiveWorldModelConfig(qwen_model="unused", measurement_dim=1),
        training=PredictiveWorldTrainingConfig(
            phase="policy", max_updates=1, world_model="unused",
            micro_batch=1, effective_global_batch=1,
        ),
        runtime=PredictiveWorldRuntimeConfig(
            nodes=1, gpus_per_node=1, output_dir=str(tmp_path / "run")
        ),
    )
    payload = asdict(config)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    data_fingerprint = canonical_sha256({
        "index": file_sha256(files["index"]),
        "feature_cache": "feature-cache-a",
        "capabilities": file_sha256(files["capabilities"]),
    })
    metadata = {
        "architecture": predictive_world_architecture_contract(config.model),
        "temporal_abi": {
            "history_visual_offsets_s": list(config.data.history_visual_offsets_s),
            "state_history_count": config.data.state_history_count,
            "state_query_hz": config.data.state_query_hz,
            "future_group_ends_s": list(config.data.future_group_ends_s),
            "action_horizon": config.data.action_horizon,
            "action_hz": config.data.action_hz,
        },
        "encoder_id": "encoder-a", "processor_id": "processor-a",
        "protocol_registry_sha256": file_sha256(files["actions"]),
        "measurement_registry_sha256": file_sha256(files["measurements"]),
        "normalization_id": "normalizer-a", "reference_id": "reference-a",
        "data_fingerprint": data_fingerprint,
        "resolved_config_sha256": canonical_sha256(payload),
        "code_revision": "code-a", "code_fingerprint": "source-a",
        "environment_fingerprint": "environment-a",
    }
    checkpoint = save_predictive_world_bundle(
        tmp_path / "policy", phase="policy", modules={"student": nn.Linear(1, 1)},
        completed_updates=3, metadata=metadata,
    ).parent
    binding = checkpoint_evaluation_binding(
        checkpoint, config_path, config, payload,
        code_revision="code-a", code_fingerprint="source-a",
    )
    assert binding["data_fingerprint"] == data_fingerprint
    assert binding["checkpoint_completed_updates"] == 3
    files["index"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config/data/code mismatch"):
        checkpoint_evaluation_binding(
            checkpoint, config_path, config, payload,
            code_revision="code-a", code_fingerprint="source-a",
        )
