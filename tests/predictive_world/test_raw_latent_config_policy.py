"""New-run configuration and causal policy agree on raw output coordinates."""
from dataclasses import asdict, replace
from pathlib import Path
import runpy

import pytest
import torch

from psi.config.predictive_world import predictive_world_loss_contract
from psi.models.predictive_world.checkpoint import load_predictive_world_bundle, save_predictive_world_bundle
from psi.models.predictive_world.scratch import random_joint_initialization, scratch_training_from_reference
from test_checkpoint import _metadata
from test_policy import _observed, _policy
from test_scratch_joint import scratch_config


def raw_config(tmp_path):
    reference = scratch_config(tmp_path)
    return replace(reference, training=replace(reference.training, latent_prediction_space="raw",
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0))


def test_raw_scratch_has_explicit_identity_and_no_calibration(tmp_path):
    config = raw_config(tmp_path)
    config.validate()
    legacy = replace(config.training, latent_prediction_space="whitened")
    assert "latent_prediction_space" not in predictive_world_loss_contract(legacy)
    assert predictive_world_loss_contract(config.training)["latent_prediction_space"] == "raw"
    assert random_joint_initialization(config.training)["normalizer"] == "unused_raw_latent"
    with pytest.raises(ValueError, match="calibration|startup"):
        legacy.validate(world_size=1)


@pytest.mark.parametrize("changes", [
    {"latent_prediction_space": "typo"}, {"normalizer_calibration_updates": 1},
    {"forecast_warmup_updates": -1}, {"forecast_ramp_updates": -1},
    {"forecast_warmup_updates": 8}, {"init_latent_prediction_migration": {}},
    {"init_posttrain_migration": {"calibration_batches": 16}},
])
def test_raw_config_rejects_invalid_or_implicit_migrations(tmp_path, changes):
    config = raw_config(tmp_path)
    with pytest.raises(ValueError):
        replace(config, training=replace(config.training, **changes)).validate()


def test_scratch_preparation_defaults_raw_keeps_optimizer_and_never_loads_checkpoint(tmp_path, monkeypatch):
    import psi.models.predictive_world.initialization as initialization
    reference = scratch_config(tmp_path)
    reference = replace(reference, training=replace(reference.training, joint_from_scratch=False,
        init_joint="unavailable-source", init_source_config="unavailable.json",
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0))
    def forbidden(*args, **kwargs):
        pytest.fail("scratch preparation read a task checkpoint")
    monkeypatch.setattr(initialization, "read_initialization_source", forbidden)
    script = Path(__file__).resolve().parents[2] / "scripts/train/predictive_world/prepare_raw_latent_training.py"
    prepare = runpy.run_path(str(script))["raw_latent_config"]
    actual = prepare(reference, initialization="scratch", output_dir=tmp_path / "new")
    actual.validate()
    assert actual.training.max_updates == 20_000
    assert actual.training == replace(scratch_training_from_reference(reference.training), max_updates=20_000)
    assert actual.training.learning_rate == reference.training.learning_rate
    assert actual.training.warmup_updates == reference.training.warmup_updates
    assert not actual.model.latent_transition.output_affine
    assert not (tmp_path / "new").exists()
    # An explicit future-loss schedule is independent of statistics fitting.
    delayed = prepare(reference, initialization="scratch", output_dir=tmp_path / "delayed",
        forecast_warmup_updates=10, forecast_ramp_updates=20)
    assert delayed.training.normalizer_calibration_updates == 0
    assert delayed.training.forecast_warmup_updates == 10


def test_pretrained_preparation_rebuilds_objective_identity_and_removes_stat_fit(tmp_path):
    from test_raw_latent_migration import _source_bundle
    from psi.config.predictive_world import PredictiveWorldTrainingConfig
    from psi.models.predictive_world.objective_transition import (
        full_ert_training, make_objective_transition, verify_objective_transition,
    )
    from psi.models.predictive_world.posttrain_transition import make_posttrain_transition
    source_dir = tmp_path / "pretrained"
    source_dir.mkdir()
    _, source, payload, source_config, checkpoint, training = _source_bundle(
        source_dir, training=PredictiveWorldTrainingConfig(phase="joint"))
    target_training = replace(full_ert_training(training),
        init_joint=str(checkpoint), init_source_config=str(source_config),
        micro_batch=2, effective_global_batch=2, max_updates=40_000,
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0,
        init_posttrain_migration=make_posttrain_transition(source, payload,
            current_visual_pooling="content_area_v1", calibration_batches=16))
    target_training = replace(target_training, init_objective_migration=make_objective_transition(
        source, payload, target_training, "FULL+ERT"))
    reference = scratch_config(tmp_path)
    reference = replace(reference, training=target_training,
        data=replace(reference.data, current_visual_pooling="content_area_v1"))
    script = Path(__file__).resolve().parents[2] / "scripts/train/predictive_world/prepare_raw_latent_training.py"
    prepare = runpy.run_path(str(script))["raw_latent_config"]
    actual = prepare(reference, initialization="pretrained", output_dir=tmp_path / "raw-run")
    record = actual.training.init_objective_migration
    assert record == verify_objective_transition(record, source, payload, actual.training)
    assert record["target_loss_contract"]["latent_prediction_space"] == "raw"
    assert actual.training.init_posttrain_migration["calibration_batches"] == 0
    assert actual.training.init_latent_prediction_migration["target_prediction_space"] == "raw"
    assert actual.model.latent_transition.output_affine


def test_raw_policy_uses_unfitted_latent_and_survives_strict_checkpoint_roundtrip(tmp_path, monkeypatch):
    source, target = _policy(latent_prediction_space="raw"), _policy(latent_prediction_space="raw")
    normalizer = source.latent_normalizer
    normalizer.fitted.fill_(False)
    normalizer.mean.fill_(float("nan"))
    normalizer.std.fill_(float("nan"))
    def forbidden(*args, **kwargs):
        pytest.fail("raw policy read legacy latent statistics")
    monkeypatch.setattr(normalizer, "unwhiten", forbidden)
    noise = torch.randn(1, 4, 2)
    expected = source.sample(_observed(), action_noise=noise)
    assert torch.isfinite(expected).all()
    names = ("history_state_adapters", "history_encoder", "latent_transition", "latent_normalizer", "student")
    metadata = _metadata()
    metadata["loss_contract"] = predictive_world_loss_contract(raw_config(tmp_path).training)
    bundle = tmp_path / "raw-deploy"
    save_predictive_world_bundle(bundle, phase="deploy", completed_updates=20_000,
        modules={name: getattr(source, name) for name in names}, metadata=metadata)
    load_predictive_world_bundle(bundle, phase="deploy",
        modules={name: getattr(target, name) for name in names}, expected_metadata={"loss_contract": metadata["loss_contract"]})
    torch.testing.assert_close(target.sample(_observed(), action_noise=noise), expected, rtol=0, atol=0)
    legacy_contract = dict(metadata["loss_contract"])
    legacy_contract.pop("latent_prediction_space")
    with pytest.raises(ValueError, match="loss_contract"):
        load_predictive_world_bundle(bundle, phase="deploy",
            modules={name: getattr(target, name) for name in names}, expected_metadata={"loss_contract": legacy_contract})


def test_raw_joint_bundle_records_zero_calibration_and_can_be_recovered(tmp_path, monkeypatch):
    from test_scratch_joint import build_small_scratch
    from test_checkpoint import _joint_metadata
    config = raw_config(tmp_path)
    built, batch = build_small_scratch(monkeypatch, config)
    batch.update(_optimizer_update=0, forecast_mode="prior")
    loss = built.model(batch)
    loss.per_sample.mean().backward()
    metadata = _joint_metadata()
    metadata.pop("parent_world_model")
    metadata["initialization"] = built.initialization
    metadata["optimization_contract"] = asdict(config.training)
    metadata["loss_contract"] = predictive_world_loss_contract(config.training)
    bundle = tmp_path / "raw-joint"
    save_predictive_world_bundle(bundle, phase="joint", modules=built.checkpoint_modules,
        completed_updates=1, metadata=metadata)
    target, _ = build_small_scratch(monkeypatch, config)
    load_predictive_world_bundle(bundle, phase="joint", modules=target.checkpoint_modules,
        expected_metadata={"loss_contract": metadata["loss_contract"]})
    assert not target.reference_modules["normalizer"].fitted
    assert int(target.reference_modules["encoder"].last_target_update) == 0


def test_raw_exact_resume_preserves_optimizer_ema_and_next_update(tmp_path, monkeypatch):
    # Reuse the full factory + optimizer recovery boundary with raw configuration.
    import test_scratch_joint
    config = raw_config(tmp_path)
    monkeypatch.setattr(test_scratch_joint, "scratch_config", lambda path: config)
    test_scratch_joint.test_checkpoint_recovers_ema_and_calibration_on_both_sides_of_freeze(
        tmp_path, monkeypatch, completed=2)
