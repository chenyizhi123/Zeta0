"""The new prediction coordinate system preserves a pretrained model's function."""
import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from psi.config.predictive_world import PredictiveWorldTrainingConfig, predictive_world_loss_contract
from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.models.predictive_world.checkpoint import read_predictive_world_bundle, save_predictive_world_bundle
from psi.models.predictive_world.initialization import initialize_joint
from psi.models.predictive_world.latent_transition import TemporalLatentTransition
from psi.models.predictive_world.posttrain_transition import (
    add_output_coordinate_tensors,
    make_raw_latent_transition,
    migrate_raw_latent_tensors,
    verify_raw_latent_transition,
)
from psi.models.predictive_world.representation import EMAFutureTrajectoryEncoder
from psi.models.predictive_world.training import module_state_sha256


def _affine_copy(transition):
    target = TemporalLatentTransition(
        {"toy": {"dim": 2, "mean": [1., -1.], "std": [2., 4.]}},
        context_dim=16, latent_shape=transition.latent_shape, hidden_dim=16,
        num_blocks=1, num_heads=4, ff_dim=32, output_affine=True,
    )
    tensors = {"latent_transition." + key: value for key, value in transition.state_dict().items()}
    if not transition.output_affine:
        tensors = add_output_coordinate_tensors(tensors, target)
    target.load_state_dict({key.removeprefix("latent_transition."): value for key, value in tensors.items()})
    return target


def _source_bundle(tmp_path, *, source_affine=False, training=None):
    from test_checkpoint import _joint_metadata
    from test_phases import _components

    encoder, readout, teacher, adapters, history, transition, normalizer = _components()
    nn.init.normal_(transition.output_projection[-1].weight, std=.1)
    for expert in (teacher,):
        for adapter in expert.adapters.values():
            nn.init.normal_(adapter.output[-1].weight, std=.1)
    if source_affine:
        transition = _affine_copy(transition)
        transition.output_scale.copy_(torch.rand_like(transition.output_scale) * 2 + .1)
        transition.output_shift.copy_(torch.randn_like(transition.output_shift))
        transition.output_calibrated.fill_(True)
    ema = EMAFutureTrajectoryEncoder(encoder)
    ema.last_target_update.fill_(17)
    ema.normalizer_frozen.fill_(True)
    modules = dict(encoder=ema, readout=readout, teacher=teacher, student=copy.deepcopy(teacher),
                   history_state_adapters=adapters, history_encoder=history,
                   latent_transition=transition, normalizer=normalizer)
    data = {}
    registry_values = dict(
        action_protocol_registry={"toy": {"dim": 2, "mean": [1., -1.], "std": [2., 4.]}},
        state_schema_registry={"state": {"dim": 3}},
        measurement_registry={"body": {"dim": 4}},
    )
    for name, value in registry_values.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        data[name] = str(path)
    data["index_path"] = str(tmp_path / "index.jsonl")
    (tmp_path / "index.jsonl.meta.json").write_text(json.dumps({"provenance": {"registry_binding": {
        "state_schema_registry_sha256": file_sha256(Path(data["state_schema_registry"]))
    }}}))
    training = training or PredictiveWorldTrainingConfig(phase="joint", objective_contract_version=5,
                                                        forecast_control_reader="both")
    architecture = {"latent_shape": [3, 8, 4], "latent_transition": {}}
    if source_affine:
        architecture["latent_transition"]["output_affine"] = True
    payload = dict(training=asdict(training), data=data, model=copy.deepcopy(architecture))
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps(payload))
    metadata = _joint_metadata()
    metadata.update(
        architecture=architecture, resolved_config_sha256=canonical_sha256(payload),
        protocol_registry_sha256=file_sha256(Path(data["action_protocol_registry"])),
        measurement_registry_sha256=file_sha256(Path(data["measurement_registry"])),
        normalization_id=module_state_sha256(normalizer),
        loss_contract=predictive_world_loss_contract(training),
        optimization_contract={"target_encoder_ema_decay": training.target_encoder_ema_decay},
    )
    checkpoint = tmp_path / "source"
    save_predictive_world_bundle(checkpoint, phase="joint", completed_updates=18,
                                modules=modules, metadata=metadata)
    source = read_predictive_world_bundle(checkpoint)
    return modules, source, payload, config_path, checkpoint, training


@pytest.mark.parametrize("source_affine", [False, True])
def test_joint_raw_migration_preserves_raw_futures_readouts_and_causal_actions(tmp_path, source_affine):
    from test_phases import _batch
    from test_policy import _ObservationEncoder, _observed
    from psi.models.predictive_world.policy import PredictiveWorldPolicy

    torch.manual_seed(73)
    modules, source, payload, config_path, checkpoint, training = _source_bundle(
        tmp_path, source_affine=source_affine)
    target = copy.deepcopy(modules)
    target["latent_transition"] = _affine_copy(modules["latent_transition"])
    record = make_raw_latent_transition(source, payload)
    target_training = replace(training, init_joint=str(checkpoint), init_source_config=str(config_path),
                              latent_prediction_space="raw", normalizer_calibration_updates=0,
                              init_latent_prediction_migration=record)
    expected = copy.deepcopy(source["metadata"])
    expected["architecture"]["latent_transition"]["output_affine"] = True
    expected["loss_contract"] = predictive_world_loss_contract(target_training)
    config = SimpleNamespace(training=target_training, data=SimpleNamespace(**payload["data"]),
                             runtime=SimpleNamespace(asset_path_map={}))
    report, _ = initialize_joint(config, target, expected)
    assert report["latent_prediction_migration"] == record
    assert report["optimizer_update_offset"] == 18 and not report["optimizer_restored"]
    assert module_state_sha256(target["normalizer"]) == module_state_sha256(modules["normalizer"])
    batch = _batch()
    context = torch.randn(2, 3, 16)
    for mode in ("prior", "world"):
        condition = batch["actions"] if mode == "world" else None
        original = modules["normalizer"].unwhiten(modules["latent_transition"](
            context, mode=mode, action_condition=condition))
        migrated = target["latent_transition"](context, mode=mode, action_condition=condition)
        torch.testing.assert_close(migrated, original, atol=2e-6, rtol=2e-6)
        args = dict(anchor_patches=batch["anchor_patches"], anchor_patch_valid=batch["anchor_patch_valid"],
                    group_valid=batch["group_valid"], scene_patch_count=5)
        for old, new in zip(modules["readout"](z_raw=original, **args), target["readout"](z_raw=migrated, **args)):
            torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-6)
    def policy(values, space):
        return PredictiveWorldPolicy(
            observation_encoder=_ObservationEncoder(), history_state_adapters=values["history_state_adapters"],
            history_encoder=values["history_encoder"], latent_transition=values["latent_transition"],
            latent_normalizer=values["normalizer"], student=values["student"], action_solver_steps=2,
            latent_prediction_space=space).eval()
    noise = torch.randn(1, 4, 2)
    old_actions = policy(modules, "whitened").sample(_observed(), action_noise=noise)
    torch.testing.assert_close(policy(target, "raw").sample(_observed(), action_noise=noise),
                               old_actions, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("damage", ["normalizer_unfitted", "nonfinite", "shape", "zero_std", "affine_missing", "affine_unset"])
def test_invalid_source_coordinates_fail_without_modifying_source(tmp_path, damage):
    from safetensors.torch import load_file

    modules, source, _, _, checkpoint, _ = _source_bundle(tmp_path, source_affine=True)
    tensors = load_file(str(checkpoint / source["weights_file"]))
    if damage == "normalizer_unfitted": tensors["normalizer.fitted"].fill_(False)
    elif damage == "nonfinite": tensors["normalizer.mean"][0, 0, 0] = float("nan")
    elif damage == "shape": tensors["normalizer.std"] = tensors["normalizer.std"][:1]
    elif damage == "zero_std": tensors["normalizer.std"][0, 0, 0] = 0
    elif damage == "affine_missing": tensors.pop("latent_transition.output_shift")
    else: tensors["latent_transition.output_calibrated"].fill_(False)
    before = {key: value.clone() for key, value in tensors.items()}
    with pytest.raises(ValueError):
        migrate_raw_latent_tensors(tensors, modules["latent_transition"], source_output_affine=True)
    for key, value in tensors.items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0, equal_nan=True)


def test_raw_migration_record_is_strict_and_only_rewrites_coordinate_identity(tmp_path):
    _, source, payload, _, _, training = _source_bundle(tmp_path)
    record = make_raw_latent_transition(source, payload)
    expected = copy.deepcopy(source["metadata"])
    expected["architecture"]["latent_transition"]["output_affine"] = True
    expected["loss_contract"] = predictive_world_loss_contract(replace(training, latent_prediction_space="raw"))
    assert verify_raw_latent_transition(record, source, payload, expected) == source["metadata"]
    for key in ("source_weights_sha256", "source_config_sha256", "source_normalization_id", "calibration_batches"):
        with pytest.raises(ValueError, match="exact source"):
            verify_raw_latent_transition({**record, key: "wrong"}, source, payload, expected)
    with pytest.raises(ValueError, match="raw prediction"):
        verify_raw_latent_transition(record, source, payload, source["metadata"])
    broken = copy.deepcopy(expected)
    broken["architecture"]["latent_transition"]["hidden_dim"] = 999
    previous = verify_raw_latent_transition(record, source, payload, broken)
    assert previous["architecture"]["latent_transition"]["hidden_dim"] == 999


def test_raw_coordinates_compose_with_pooling_only_and_dual_response_transfer(tmp_path):
    from psi.models.predictive_world.posttrain_transition import make_posttrain_transition
    from psi.models.predictive_world.objective_transition import full_ert_training, make_objective_transition

    modules, source, payload, config_path, checkpoint, training = _source_bundle(
        tmp_path, training=PredictiveWorldTrainingConfig(phase="joint"))
    raw_record = make_raw_latent_transition(source, payload)
    pooling_record = make_posttrain_transition(source, payload, current_visual_pooling="content_area_v1",
                                               calibration_batches=0)
    target_training = replace(full_ert_training(training),
                              init_joint=str(checkpoint), init_source_config=str(config_path),
                              latent_prediction_space="raw", normalizer_calibration_updates=0,
                              init_latent_prediction_migration=raw_record,
                              init_posttrain_migration=pooling_record)
    objective_record = make_objective_transition(source, payload, target_training, "FULL+ERT")
    target_training = replace(target_training, init_objective_migration=objective_record)
    expected = copy.deepcopy(source["metadata"])
    expected["architecture"]["latent_transition"]["output_affine"] = True
    expected["temporal_abi"]["current_visual_pooling"] = "content_area_v1"
    expected["loss_contract"] = predictive_world_loss_contract(target_training)
    target = copy.deepcopy(modules)
    target["latent_transition"] = _affine_copy(modules["latent_transition"])
    config = SimpleNamespace(training=target_training, data=SimpleNamespace(**payload["data"]),
                             runtime=SimpleNamespace(asset_path_map={}))
    report, _ = initialize_joint(config, target, expected)
    assert report["latent_prediction_migration"] == raw_record
    assert report["posttrain_migration"] == pooling_record
    assert report["objective_migration"] == objective_record
    # Removing the explicit coordinate migration cannot bypass the output and loss identity checks.
    config.training = replace(target_training, init_latent_prediction_migration=None,
                              init_posttrain_migration=None)
    with pytest.raises(ValueError, match="architecture|prediction space"):
        initialize_joint(config, target, expected)


def test_ordinary_objective_migration_cannot_reinterpret_whitened_outputs_as_raw(tmp_path):
    from psi.models.predictive_world.objective_transition import full_ert_training, make_objective_transition

    modules, source, payload, config_path, checkpoint, training = _source_bundle(
        tmp_path, training=PredictiveWorldTrainingConfig(phase="joint"))
    target_training = replace(full_ert_training(training),
        init_joint=str(checkpoint), init_source_config=str(config_path),
        latent_prediction_space="raw", normalizer_calibration_updates=0)
    target_training = replace(target_training, init_objective_migration=make_objective_transition(
        source, payload, target_training, "FULL+ERT"))
    expected = copy.deepcopy(source["metadata"])
    expected["loss_contract"] = predictive_world_loss_contract(target_training)
    config = SimpleNamespace(training=target_training, data=SimpleNamespace(**payload["data"]),
                             runtime=SimpleNamespace(asset_path_map={}))
    with pytest.raises(ValueError, match="explicit coordinate migration"):
        initialize_joint(config, copy.deepcopy(modules), expected)


def test_raw_source_can_initialize_raw_without_refitting_or_reconversion(tmp_path):
    training = PredictiveWorldTrainingConfig(phase="joint", latent_prediction_space="raw",
                                            normalizer_calibration_updates=0)
    modules, source, payload, config_path, checkpoint, _ = _source_bundle(tmp_path, training=training)
    # Unfitted statistics are legal and unused in a source that already predicts raw latents.
    modules["normalizer"].fitted.fill_(False)
    modules["encoder"].normalizer_frozen.fill_(False)
    source["metadata"]["normalization_id"] = module_state_sha256(modules["normalizer"])
    second_checkpoint = tmp_path / "raw_source"
    save_predictive_world_bundle(second_checkpoint, phase="joint", completed_updates=18,
                                modules=modules, metadata=source["metadata"])
    target_training = replace(training, init_joint=str(second_checkpoint), init_source_config=str(config_path))
    target = copy.deepcopy(modules)
    config = SimpleNamespace(training=target_training, data=SimpleNamespace(**payload["data"]),
                             runtime=SimpleNamespace(asset_path_map={}))
    report, _ = initialize_joint(config, target, source["metadata"])
    assert "latent_prediction_migration" not in report
    assert not bool(target["normalizer"].fitted) and not bool(target["encoder"].normalizer_frozen)
    with pytest.raises(ValueError, match="whitened source"):
        make_raw_latent_transition(source, payload)
