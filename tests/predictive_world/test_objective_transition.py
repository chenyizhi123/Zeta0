import copy
import json
from dataclasses import asdict, replace

import pytest
import torch

from psi.config.predictive_world import (
    PredictiveWorldConfig,
    PredictiveWorldDataConfig,
    PredictiveWorldRuntimeConfig,
    PredictiveWorldTrainingConfig,
    predictive_world_loss_contract,
)
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.models.predictive_world.checkpoint import (
    read_predictive_world_bundle,
    save_predictive_world_bundle,
)
from psi.models.predictive_world.initialization import initialize_joint
from psi.models.predictive_world.objective_transition import (
    full_ert_training,
    make_objective_transition,
    verify_objective_transition,
)
from psi.models.predictive_world.representation import EMAFutureTrajectoryEncoder
from psi.models.predictive_world.training import module_state_sha256


def _source():
    training = PredictiveWorldTrainingConfig(phase="joint", world_model="world", max_updates=15000,
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0)
    payload = {"training": asdict(training)}
    source = {"phase": "joint", "weights_sha256": "a" * 64, "metadata": {
        "resolved_config_sha256": canonical_sha256(payload),
        "loss_contract": predictive_world_loss_contract(training),
    }}
    return training, payload, source


def test_dual_reader_transition_declares_exact_source_and_delta():
    training, payload, source = _source()
    target = full_ert_training(training)
    record = make_objective_transition(source, payload, target, "FULL+ERT")
    assert verify_objective_transition(record, source, payload, target) == record
    assert record["changed_fields"] == {"forecast_control_reader": {"source": "teacher", "target": "both"}}
    assert record["optimizer_restored"] is False
    assert record["target_loss_contract"]["version"] == 5
    assert source["metadata"]["loss_contract"] == predictive_world_loss_contract(training)


def test_pretrained_v6_transition_keeps_legacy_source_and_raw_target_identity():
    training, payload, source = _source()
    training = replace(full_ert_training(training), world_model=None, joint_from_scratch=True)
    values = asdict(training)
    values["from_scratch"] = values.pop("joint_from_scratch")
    payload["training"] = values
    source["metadata"].update(
        resolved_config_sha256=canonical_sha256(payload),
        loss_contract=predictive_world_loss_contract(training),
    )
    target = replace(training, objective_contract_version=6, latent_prediction_space="raw",
        preserve_control_sampling=True, joint_from_scratch=False,
        init_joint="source", init_source_config="source.json")
    record = make_objective_transition(source, payload, target, "pretrained")
    assert record["format"] == "zeta0.joint_training_objective_transition.v1"
    assert record["target_loss_contract"] == predictive_world_loss_contract(target)
    assert record["target_loss_contract"]["latent_prediction_space"] == "raw"
    assert verify_objective_transition(record, source, payload, target) == record
    with pytest.raises(ValueError, match="undeclared changes"):
        verify_objective_transition(record, source, payload, replace(target, scene_loss_weight=.5))


@pytest.mark.parametrize("damage", ["hash", "source_config", "field", "extra", "undeclared", "source_contract"])
def test_transition_rejects_tampering_and_undeclared_changes(damage):
    training, payload, source = _source()
    target = full_ert_training(training)
    record = make_objective_transition(source, payload, target, "FULL+ERT")
    if damage == "hash":
        record["source_weights_sha256"] = "b" * 64
    elif damage == "source_config":
        payload["training"]["seed"] += 1
    elif damage == "field":
        record["changed_fields"] = {}
    elif damage == "extra":
        record["allow_everything"] = True
    elif damage == "undeclared":
        target = replace(target, teacher_action_loss_weight=0)
    else:
        source["metadata"]["loss_contract"]["forecast_control_scope"] = "both"
    with pytest.raises(ValueError):
        verify_objective_transition(record, source, payload, target)


def _disk_source(tmp_path):
    from test_checkpoint import _joint_metadata
    from test_phases import _components
    encoder, readout, teacher, adapters, history, transition, normalizer = _components()
    encoder = EMAFutureTrajectoryEncoder(encoder, decay=0.99)
    encoder.normalizer_frozen.fill_(True)
    encoder.last_target_update.fill_(14999)
    modules = dict(encoder=encoder, readout=readout, teacher=teacher,
        history_state_adapters=adapters, history_encoder=history, latent_transition=transition,
        normalizer=normalizer, student=copy.deepcopy(teacher))
    data, hashes = {}, {}
    values = {
        "action_protocol_registry": {"toy": {"dim": 2, "mean": [1., -1.], "std": [2., 4.]}},
        "state_schema_registry": {"state": {"dim": 3}},
        "measurement_registry": {"body": {"dim": 4}},
    }
    for name, content in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content))
        data[name], hashes[name] = str(path), file_sha256(path)
    data["index_path"] = str(tmp_path / "index.jsonl")
    (tmp_path / "index.jsonl.meta.json").write_text(json.dumps({"provenance": {"registry_binding": {
        "state_schema_registry_sha256": hashes["state_schema_registry"]}}}))
    training, payload, _ = _source()
    payload["data"] = data
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps(payload))
    metadata = _joint_metadata()
    metadata.update(resolved_config_sha256=canonical_sha256(payload),
        protocol_registry_sha256=hashes["action_protocol_registry"],
        measurement_registry_sha256=hashes["measurement_registry"],
        loss_contract=predictive_world_loss_contract(training),
        optimization_contract={"target_encoder_ema_decay": 0.99},
        normalization_id=module_state_sha256(normalizer))
    path = tmp_path / "source"
    save_predictive_world_bundle(path, phase="joint", completed_updates=15000,
                                modules=modules, metadata=metadata)
    source = read_predictive_world_bundle(path)
    target_training = replace(full_ert_training(training), world_model=None,
        init_joint=str(path), init_source_config=str(config_path))
    target_training = replace(target_training, init_objective_migration=make_objective_transition(
        source, payload, target_training, "FULL+ERT"))
    config = PredictiveWorldConfig(training=target_training, data=PredictiveWorldDataConfig(**data),
                                   runtime=PredictiveWorldRuntimeConfig(output_dir=str(tmp_path / "new")))
    expected = {**metadata, "loss_contract": predictive_world_loss_contract(target_training)}
    return config, modules, expected


def test_real_disk_joint_initialization_allows_only_recorded_objective_delta(tmp_path):
    config, source, expected = _disk_source(tmp_path)
    destination = copy.deepcopy(source)
    for module in destination.values():
        for parameter in module.parameters():
            parameter.data.zero_()
    report, parent = initialize_joint(config, destination, expected)
    assert report["objective_migration"]["variant_id"] == "FULL+ERT"
    assert report["optimizer_restored"] is False
    assert parent["phase"] == "world_model"
    for name in source:
        assert module_state_sha256(source[name]) == module_state_sha256(destination[name])


@pytest.mark.parametrize("damage", ["undeclared", "architecture", "processor_id", "encoder_id", "temporal_abi"])
def test_objective_transition_does_not_relax_other_initialization_guards(tmp_path, damage):
    config, modules, expected = _disk_source(tmp_path)
    if damage == "undeclared":
        config = replace(config, training=replace(config.training, init_objective_migration=None))
    else:
        expected[damage] = {"changed": True} if damage.endswith("abi") or damage == "architecture" else "changed"
    before = {name: module_state_sha256(module) for name, module in modules.items()}
    with pytest.raises(ValueError, match="metadata mismatch"):
        initialize_joint(config, modules, expected)
    assert before == {name: module_state_sha256(module) for name, module in modules.items()}


@pytest.mark.parametrize("damage", ["teacher_weight", "student_weight", "variant", "body", "temporal", "target", "source"])
def test_dual_reader_transition_rejects_relabeling_or_unrelated_deltas(damage):
    training, payload, source = _source()
    target = full_ert_training(training)
    record = make_objective_transition(source, payload, target, "FULL+ERT")
    assert record["changed_fields"] == {"forecast_control_reader": {"source": "teacher", "target": "both"}}
    if damage == "teacher_weight":
        record["target_loss_contract"]["forecast_control_teacher_response_weight"] = 1.
    elif damage == "student_weight":
        record["target_loss_contract"]["forecast_control_student_response_weight"] = 0.
    elif damage == "variant":
        record["variant_id"] = "FULL"
    elif damage == "body":
        target = replace(target, body_loss_weight=.5)
    elif damage == "temporal":
        target = replace(target, temporal_loss_weight=.5)
    elif damage == "target":
        target = replace(target, forecast_control_target="demonstration_flow")
    else:
        source["weights_sha256"] = "b"*64
    with pytest.raises(ValueError):
        verify_objective_transition(record, source, payload, target)


@pytest.mark.parametrize("saved_reader", ["teacher", "student", "both"])
def test_exact_resume_rejects_dual_reader_contract_mismatches_before_mutating_weights(tmp_path, saved_reader):
    from test_checkpoint import _metadata

    from psi.models.predictive_world.checkpoint import load_predictive_world_bundle

    training, _, _ = _source()
    dual_reader = full_ert_training(training)
    expected_contract = predictive_world_loss_contract(dual_reader)
    saved = {
        "teacher": training,
        "student": replace(training, objective_contract_version=4, forecast_control_reader="student"),
        "both": dual_reader,
    }[saved_reader]
    metadata = dict(_metadata(), loss_contract=predictive_world_loss_contract(saved))
    path = save_predictive_world_bundle(tmp_path/"bundle", phase="policy", completed_updates=1,
        modules={"student": torch.nn.Linear(2, 3)}, metadata=metadata).parent
    target = {"student": torch.nn.Linear(2, 3)}
    before = module_state_sha256(target["student"])
    if saved_reader == "both":
        loaded = load_predictive_world_bundle(path, phase="policy", modules=target,
            expected_metadata={"loss_contract": expected_contract})
        assert loaded["metadata"]["loss_contract"] == expected_contract
    else:
        with pytest.raises(ValueError, match="loss_contract"):
            load_predictive_world_bundle(path, phase="policy", modules=target,
                expected_metadata={"loss_contract": expected_contract})
        assert module_state_sha256(target["student"]) == before
