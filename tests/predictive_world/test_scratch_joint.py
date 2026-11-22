"""Random full-q29 joint training, calibration, gradients and strict recovery."""
import copy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from psi.config.predictive_world import (
    PredictiveWorldConfig, PredictiveWorldDataConfig, PredictiveWorldModelConfig,
    PredictiveWorldRuntimeConfig, PredictiveWorldTrainingConfig, predictive_world_loss_contract,
    LatentTransitionConfig,
)
from psi.models.predictive_world.scratch import random_joint_initialization, scratch_training_from_reference


def scratch_config(tmp_path):
    from test_body_target_transition import schema_fixture
    from psi.models.predictive_world.body_target_transition import SCHEMA, whole_body_schema

    protocols = tmp_path / "protocols.json"
    protocols.write_text(json.dumps({"toy": {"dim": 2, "mean": [1., -1.], "std": [2., 4.]}}))
    states = tmp_path / "states.json"
    states.write_text(json.dumps({"state": {"dim": 3, "mean": [0.]*3, "std": [1.]*3}}))
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"provenance": {"encoder_id": "fixture", "processor_id": "fixture"}}))
    old, names, mean, std = schema_fixture()
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({SCHEMA: whole_body_schema(old, names, mean, std)}))
    return PredictiveWorldConfig(
        data=PredictiveWorldDataConfig(action_protocol_registry=str(protocols), state_schema_registry=str(states),
            measurement_registry=str(measurements), canonical_future_capabilities="capabilities.json",
            index_path="index.jsonl", feature_cache=str(features), state_history_count=1,
            observed_visual_mode="current_multiview"),
        model=PredictiveWorldModelConfig(qwen_model="frozen_fixture", measurement_dim=29,
            latent_shape=(3, 8, 4), representation_hidden_dim=16, representation_heads=4,
            representation_ff_dim=32, representation_blocks=1, history_hidden_dim=16, history_heads=4,
            history_ff_dim=32, history_blocks=1, expert_hidden_dim=24, expert_heads=4,
            expert_ff_dim=48, expert_blocks=1, max_scene_patches=5, body_points_per_group=2,
            latent_transition=LatentTransitionConfig(
                hidden_dim=16, num_blocks=1, num_heads=4, ff_dim=32)),
        training=PredictiveWorldTrainingConfig(phase="joint", joint_from_scratch=True, seed=0,
            objective_contract_version=5, forecast_control_reader="both", max_updates=8,
            micro_batch=2, effective_global_batch=2, normalizer_calibration_updates=2,
            forecast_warmup_updates=2, forecast_ramp_updates=2, rtc_training_max_delay=0),
        runtime=PredictiveWorldRuntimeConfig(nodes=1, gpus_per_node=1, output_dir=str(tmp_path / "run")))


@pytest.mark.parametrize("damage", [
    {"joint_from_scratch": 1}, {"world_model": "world"}, {"init_joint": "joint", "init_source_config": "source"},
    {"init_processor_migration": {}}, {"init_objective_migration": {}}, {"init_state_history_migration": {}},
    {"init_body_target_migration": {}}, {"normalizer_calibration_updates": 0}, {"forecast_ramp_updates": 0},
    {"forecast_warmup_updates": 7}, {"phase": "policy"}, {"seed": None},
])
def test_scratch_cannot_silently_load_pretraining_or_skip_startup(tmp_path, damage):
    config = scratch_config(tmp_path)
    with pytest.raises(ValueError):
        replace(config, training=replace(config.training, **damage)).validate()


def test_reference_conversion_retains_full_objectives_and_budget(tmp_path):
    config = scratch_config(tmp_path)
    reference = replace(config.training, joint_from_scratch=False, max_updates=40000,
        init_joint="pretrained", init_source_config="source", init_body_target_migration={"fixture": "body29"},
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0)
    result = scratch_training_from_reference(reference, latent_prediction_space="whitened")
    assert predictive_world_loss_contract(result) == predictive_world_loss_contract(reference)
    assert result.max_updates == reference.max_updates == 40000
    assert result.seed == 0
    assert result.init_joint is None and result.init_source_config is None and result.init_body_target_migration is None
    assert result.normalizer_calibration_updates == 500 and result.forecast_warmup_updates == 500
    assert result.forecast_ramp_updates == 500


def build_small_scratch(monkeypatch, config):
    from test_phases import _batch
    from psi.models.predictive_world import factory
    import psi.data.predictive_world.provenance as provenance
    import transformers

    batch = _batch(action_steps=30)
    batch["observed"] = replace(batch["observed"], raw_states=batch["observed"].raw_states[:, -1:],
        state_valid=batch["observed"].state_valid[:, -1:], state_times=torch.zeros(2, 1))
    batch["actions"] = replace(batch["actions"], times=torch.arange(1, 31)[None].expand(2, -1).float()/50)
    batch["features"] = replace(batch["features"], current_visual=torch.randn(2, 16, 2048))
    batch["anchor_patches"] = batch["features"].current_visual
    batch["future_patches"] = torch.randn(2, 3, 5, 2048)
    batch["current_measurements"] = torch.randn(2, 29)
    batch["current_measurement_valid"] = torch.ones(2, 29, dtype=torch.bool)
    batch["future_measurements"] = torch.randn(2, 3, 2, 29)
    batch["future_measurement_valid"] = torch.ones(2, 3, 2, 29, dtype=torch.bool)
    batch["world_action_group_valid"] = torch.ones(2, 3, dtype=torch.bool)

    class FrozenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(hidden_size=6)

    class CachedObservation(nn.Module):
        # Only frozen VLM I/O is replaced; the factory and all task modules are real.
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, observed):
            return batch["features"]

    def forbidden(*args, **kwargs):
        pytest.fail("scratch loaded a task checkpoint")

    monkeypatch.setattr(transformers.Qwen3VLForConditionalGeneration, "from_pretrained", lambda *a, **k: FrozenBackbone())
    monkeypatch.setattr(provenance, "verify_qwen_provenance", lambda *a, **k: {})
    monkeypatch.setattr(factory, "FrozenQwenObservationEncoder", CachedObservation)
    monkeypatch.setattr(factory, "load_predictive_world_bundle", forbidden)
    monkeypatch.setattr(factory, "initialize_joint", forbidden)
    built = factory.build_phase(config, device=torch.device("cpu"))
    return built, batch


def test_factory_initializes_all_task_modules_and_new_ema_without_loading_weights(tmp_path, monkeypatch):
    config = scratch_config(tmp_path)
    config.validate()
    built, _ = build_small_scratch(monkeypatch, config)
    assert built.parent_world_model is None
    assert built.initialization == random_joint_initialization(config.training)
    assert not built.reference_modules["normalizer"].fitted
    assert all(not p.requires_grad for p in built.model.observation_encoder.parameters())
    encoder = built.reference_modules["encoder"]
    assert int(encoder.last_target_update) == -1
    assert not encoder.normalizer_frozen
    for key, value in encoder.online_encoder.state_dict().items():
        torch.testing.assert_close(value, encoder.target_encoder.state_dict()[key], rtol=0, atol=0)
    assert all(not p.requires_grad for p in encoder.target_encoder.parameters())
    teacher_stats = built.checkpoint_modules["teacher"].state_adapters.adapters["state"]
    history_stats = built.checkpoint_modules["history_state_adapters"].adapters["state"]
    assert teacher_stats.mean.data_ptr() != history_stats.mean.data_ptr()
    assert teacher_stats.std.data_ptr() != history_stats.std.data_ptr()
    for teacher, student in zip(built.checkpoint_modules["teacher"].parameters(), built.checkpoint_modules["student"].parameters()):
        assert teacher.data_ptr() != student.data_ptr()
        torch.testing.assert_close(teacher, student, rtol=0, atol=0)


def test_scratch_calibrates_only_training_batches_then_freezes_and_trains_both_responses(tmp_path, monkeypatch):
    built, batch = build_small_scratch(monkeypatch, scratch_config(tmp_path))
    model = built.model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    normalizer = built.reference_modules["normalizer"]
    encoder = built.reference_modules["encoder"]
    frozen = None
    for update, expected_weight in enumerate((0., 0., .5, 1., 1.)):
        optimizer.zero_grad(set_to_none=True)
        batch.update(_optimizer_update=update, forecast_mode="prior" if update % 2 == 0 else "world")
        result = model(batch)
        assert bool(normalizer.fitted)
        assert result.metrics["forecast_objective_weight"].item() == expected_weight
        result.per_sample.mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert built.reference_modules["readout"].body_output[1].weight.grad[14:].abs().sum() > 0
        optimizer.step()
        if update == 1:
            frozen = normalizer.mean.clone()
            assert encoder.normalizer_frozen
        elif update > 1:
            torch.testing.assert_close(normalizer.mean, frozen, rtol=0, atol=0)
            assert encoder.last_normalizer_update.item() == 1
        if update == 2:
            assert result.metrics["prior_control_teacher_response_weight"].item() == .5
            assert result.metrics["prior_control_student_response_weight"].item() == .5
    assert built.reference_modules["readout"].body_output[1].out_features == 29


def test_first_calibration_cannot_publish_unobserved_future_groups(tmp_path, monkeypatch):
    built, batch = build_small_scratch(monkeypatch, scratch_config(tmp_path))
    batch["group_valid"][:, -1] = False
    with pytest.raises(ValueError, match="every future group"):
        built.model(batch)
    assert not built.reference_modules["normalizer"].fitted


@pytest.mark.parametrize("completed", [1, 2])
def test_checkpoint_recovers_ema_and_calibration_on_both_sides_of_freeze(tmp_path, monkeypatch, completed):
    from test_checkpoint import _joint_metadata
    from psi.data.predictive_world.cache import canonical_sha256
    from psi.models.predictive_world.checkpoint import save_predictive_world_bundle, load_predictive_world_bundle
    from psi.models.predictive_world.engine import save_engine_state, load_engine_state

    config = scratch_config(tmp_path)
    built, batch = build_small_scratch(monkeypatch, config)
    optimizer = torch.optim.AdamW([p for p in built.model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
    for update in range(completed):
        batch.update(_optimizer_update=update, forecast_mode="prior" if update % 2 == 0 else "world")
        optimizer.zero_grad(set_to_none=True)
        built.model(batch).per_sample.mean().backward()
        optimizer.step()
        scheduler.step()
    metadata = {**_joint_metadata(), "initialization": built.initialization,
                "optimization_contract": asdict(config.training),
                "resolved_config_sha256": canonical_sha256(asdict(config)),
                "loss_contract": predictive_world_loss_contract(config.training)}
    metadata.pop("parent_world_model")
    checkpoint = tmp_path / "checkpoint"
    save_predictive_world_bundle(checkpoint, phase="joint", modules=built.checkpoint_modules,
                                  completed_updates=completed, metadata=metadata)
    save_engine_state(checkpoint, optimizer=optimizer, scheduler=scheduler, completed_updates=completed,
                      next_occurrence=completed * 2, config_identity=metadata["resolved_config_sha256"])
    restored, _ = build_small_scratch(monkeypatch, config)
    restored_optimizer = torch.optim.AdamW([p for p in restored.model.parameters() if p.requires_grad], lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: .9 ** step)
    load_predictive_world_bundle(checkpoint, phase="joint", modules=restored.checkpoint_modules,
                                  expected_metadata={"resolved_config_sha256": metadata["resolved_config_sha256"]})
    progress = load_engine_state(checkpoint, optimizer=restored_optimizer, scheduler=restored_scheduler,
        config_identity=metadata["resolved_config_sha256"], device=torch.device("cpu"))
    assert progress == (completed, completed * 2)
    # Reuse identical observations and flow noise; only scratch state is under test.
    restored.model.observation_encoder.forward = lambda observed: batch["features"]
    batch.update(_optimizer_update=completed, forecast_mode="prior" if completed % 2 == 0 else "world")
    rng = torch.get_rng_state()
    optimizer.zero_grad(set_to_none=True)
    wanted = built.model(batch)
    wanted.per_sample.mean().backward()
    optimizer.step()
    scheduler.step()
    torch.set_rng_state(rng)
    restored_optimizer.zero_grad(set_to_none=True)
    actual = restored.model(batch)
    actual.per_sample.mean().backward()
    restored_optimizer.step()
    restored_scheduler.step()
    torch.testing.assert_close(actual.per_sample, wanted.per_sample, rtol=0, atol=0)
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()
    for name in built.checkpoint_modules:
        for key, value in built.checkpoint_modules[name].state_dict().items():
            torch.testing.assert_close(restored.checkpoint_modules[name].state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["parent", "seed", "pretraining", "schedule"])
def test_scratch_checkpoint_rejects_false_initialization_provenance(tmp_path, damage):
    from test_checkpoint import _joint_metadata
    from psi.models.predictive_world.checkpoint import _validate_phase_metadata

    training = scratch_config(tmp_path).training
    metadata = copy.deepcopy(_joint_metadata())
    metadata.pop("parent_world_model")
    metadata.update(initialization=random_joint_initialization(training), optimization_contract=asdict(training))
    _validate_phase_metadata("joint", metadata)
    if damage == "parent":
        metadata["parent_world_model"] = _joint_metadata()["parent_world_model"]
    elif damage == "seed":
        metadata["initialization"]["seed"] = 2
    elif damage == "pretraining":
        metadata["optimization_contract"]["init_joint"] = "pretrained"
    else:
        metadata["optimization_contract"]["normalizer_calibration_updates"] = 0
    with pytest.raises(ValueError):
        _validate_phase_metadata("joint", metadata)
