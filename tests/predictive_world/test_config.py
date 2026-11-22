import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from psi.config.predictive_world import (
    PredictiveWorldConfig,
    PredictiveWorldDataConfig,
    PredictiveWorldModelConfig,
    PredictiveWorldRuntimeConfig,
    PredictiveWorldTrainingConfig,
    load_predictive_world_config,
    predictive_world_architecture_contract,
    predictive_world_loss_contract,
)


def _config(tmp_path: Path, **training):
    return PredictiveWorldConfig(
        data=PredictiveWorldDataConfig(
            measurement_registry="measurement.json",
            canonical_future_capabilities="capabilities.json",
            action_protocol_registry="actions.json",
            state_schema_registry="states.json",
            index_path="index.jsonl",
            feature_cache="features",
        ),
        model=PredictiveWorldModelConfig(qwen_model="qwen", measurement_dim=4),
        training=PredictiveWorldTrainingConfig(max_updates=10, **training),
        runtime=PredictiveWorldRuntimeConfig(output_dir=str(tmp_path / "new-run")),
    )


def test_architecture_contract_serializes_legacy_model():
    model = PredictiveWorldModelConfig(qwen_model="qwen", measurement_dim=4)
    expected = json.loads(json.dumps(asdict(model)))
    expected.pop("goal_intent_enabled")
    for name in ("host_architecture", "gr00t_model", "gr00t_config_sha256"):
        expected.pop(name)
    expected["latent_transition"].pop("output_affine")
    assert predictive_world_architecture_contract(model) == expected


def test_ddp_bucket_views_are_an_explicit_boolean_runtime_setting(tmp_path):
    config = _config(tmp_path)
    assert config.runtime.ddp_gradient_as_bucket_view is False
    optimized = replace(config, runtime=replace(config.runtime, ddp_gradient_as_bucket_view=True))
    optimized.validate()
    assert predictive_world_architecture_contract(config.model) == predictive_world_architecture_contract(optimized.model)
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="ddp_gradient_as_bucket_view must be boolean"):
            replace(config, runtime=replace(config.runtime, ddp_gradient_as_bucket_view=invalid)).validate()


def test_gr00t_identity_and_initialization_cannot_run_as_legacy(tmp_path):
    config = _config(tmp_path, phase="joint", joint_from_gr00t=True,
        objective_contract_version=5, forecast_control_reader="both", latent_prediction_space="raw",
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0)
    with pytest.raises(ValueError, match="requires the N1.7"):
        config.validate()
    config = replace(config, model=replace(config.model, host_architecture="gr00t_n1d7",
        gr00t_model="native-base", gr00t_config_sha256="a" * 64, action_solver_steps=4),
        data=replace(config.data, state_history_count=1, observed_visual_mode="current_multiview", action_horizon=40))
    config.validate()
    contract = predictive_world_architecture_contract(config.model)
    assert contract["host_architecture"] == "gr00t_n1d7" and "gr00t_adapter" in contract
    with pytest.raises(ValueError, match="distinct"):
        replace(config, training=replace(config.training, joint_from_scratch=True)).validate()
    with pytest.raises(ValueError, match="horizon40"):
        replace(config, data=replace(config.data, action_horizon=30)).validate()


@pytest.mark.parametrize("field", ["model", "data", "training", "migration"])
def test_retired_goal_experiments_cannot_silently_run_as_mainline(tmp_path, field):
    config = _config(tmp_path)
    if field == "model":
        config = replace(config, model=replace(config.model, goal_intent_enabled=True))
    elif field == "data":
        config = replace(config, data=replace(config.data, goal_supervision_path="goals.json"))
    else:
        change = {"objective_contract_version": 3} if field == "training" else {"init_goal_intent": {}}
        config = replace(config, training=replace(config.training, **change))
    with pytest.raises(ValueError, match="retired"):
        config.validate()


@pytest.mark.parametrize("setting", [
    {"student_latent_mode": "zero_raw"}, {"student_latent_gradient": "stop"},
    {"body_auxiliary_enabled": False}, {"future_body_input_enabled": False},
    {"preserve_control_sampling": True},
    {"forecast_control_target": "demonstration_flow"},
])
def test_training_rejects_unsupported_objective_overrides(tmp_path, setting):
    for version in (1, 2):
        with pytest.raises(ValueError):
            _config(tmp_path, objective_contract_version=version, **setting).validate()


def test_executor_reader_has_own_versioned_loss_identity(tmp_path):
    with pytest.raises(ValueError, match="objective_contract_version=4"):
        _config(tmp_path, phase="joint", forecast_control_reader="student").validate()
    config = _config(tmp_path, phase="joint", objective_contract_version=4,
                     forecast_control_reader="student",
                     world_model="world", normalizer_calibration_updates=0,
                     forecast_warmup_updates=0, forecast_ramp_updates=0)
    config.validate()
    contract = predictive_world_loss_contract(config.training)
    assert contract["version"] == 4
    assert contract["forecast_control_reader"] == "student"
    assert contract["forecast_control_target"] == "teacher_response"
    legacy = _config(tmp_path, objective_contract_version=2).training
    assert "forecast_control_reader" not in predictive_world_loss_contract(legacy)


def test_dual_reader_has_fixed_independent_versioned_loss_identity(tmp_path):
    config = _config(tmp_path, phase="joint", objective_contract_version=5,
        forecast_control_reader="both", world_model="world", normalizer_calibration_updates=0,
        forecast_warmup_updates=0, forecast_ramp_updates=0)
    config.validate()
    contract = predictive_world_loss_contract(config.training)
    assert contract["version"] == 5
    assert contract["forecast_control_reader"] == "both"
    assert contract["forecast_control_target"] == "teacher_response"
    assert contract["forecast_control_loss_weight"] == 1.
    assert contract["forecast_control_teacher_response_weight"] == .5
    assert contract["forecast_control_student_response_weight"] == .5
    assert contract["body_auxiliary_enabled"] is True
    assert contract["body_loss_weight"] == 1.
    assert contract["temporal_loss_weight"] == .25


@pytest.mark.parametrize("damage", [
    {"objective_contract_version": 4}, {"forecast_control_reader": "teacher"},
    {"forecast_control_target": "demonstration_flow"}, {"forecast_control_scope": "both"},
    {"forecast_control_loss_weight": 2.}, {"forecast_control_loss_weight": 0.},
    {"body_auxiliary_enabled": False}, {"body_loss_weight": 0.}, {"temporal_loss_weight": 0.},
    {"student_latent_gradient": "stop"}, {"student_latent_mode": "zero_raw"},
])
def test_dual_reader_cannot_silently_change_full_plus_ert_semantics(tmp_path, damage):
    settings = dict(phase="joint", objective_contract_version=5, forecast_control_reader="both",
        world_model="world", normalizer_calibration_updates=0, forecast_warmup_updates=0,
        forecast_ramp_updates=0)
    settings.update(damage)
    with pytest.raises(ValueError):
        _config(tmp_path, **settings).validate()


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_legacy_loss_contracts_do_not_acquire_dual_reader_keys(version):
    training = PredictiveWorldTrainingConfig(objective_contract_version=version,
        forecast_control_reader="student" if version == 4 else "teacher")
    contract = predictive_world_loss_contract(training)
    assert "forecast_control_teacher_response_weight" not in contract
    assert "forecast_control_student_response_weight" not in contract
    if version < 4:
        assert "forecast_control_reader" not in contract
    else:
        assert contract["forecast_control_reader"] == "student"


@pytest.mark.parametrize("setting", [
    {"forecast_control_reader": "other"},
    {"objective_contract_version": 4},
    {"objective_contract_version": 4, "forecast_control_reader": "student"},
])
def test_invalid_executor_reader_config_rejected(tmp_path, setting):
    with pytest.raises(ValueError):
        _config(tmp_path, **setting).validate()


@pytest.mark.parametrize("setting", [
    {"objective_contract_version": True}, {"objective_contract_version": 3},
    {"student_latent_mode": "oracle"}, {"student_latent_gradient": "freeze_all"},
    {"forecast_control_target": "actions"}, {"body_auxiliary_enabled": 0},
    {"preserve_control_sampling": 1}, {"init_objective_migration": {}},
])
def test_invalid_objective_config_rejected(tmp_path, setting):
    with pytest.raises(ValueError):
        _config(tmp_path, **setting).validate()


@pytest.mark.parametrize('resume', [False, True])
def test_asset_map_is_validated_even_on_resume(tmp_path, resume):
    from dataclasses import replace
    config = _config(tmp_path)
    bad = replace(config, runtime=replace(config.runtime, asset_path_map={'relative': '/local'}))
    with pytest.raises(ValueError, match='asset_path_map'):
        bad.validate(allow_existing_output_override=resume)
    good = replace(config, runtime=replace(config.runtime, asset_path_map={'/old': '/local'}))
    good.validate(allow_existing_output_override=resume)


def test_default_5n40g_training_contract_is_explicit(tmp_path):
    config = _config(tmp_path)
    config.validate()
    assert config.runtime.world_size == 40
    assert config.training.micro_batch == 256
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.effective_global_batch == 10_240
    assert config.training.joint_world_learning_rate == 3e-5
    assert config.training.joint_world_loss_weight == 1.0
    assert config.training.joint_action_loss_weight == 1.0
    assert config.training.target_encoder_ema_decay == 0.99
    assert config.training.normalizer_calibration_updates == 500
    assert config.training.forecast_warmup_updates == 500
    assert config.training.forecast_ramp_updates == 500
    assert config.training.latent_diagnostic_every == 100
    assert config.training.policy_latent_origin == "predicted"
    assert config.training.rtc_training_max_delay is None
    assert config.model.latent_transition.semantics == "parallel_process"
    assert config.training.forecast_control_loss_weight == 1.0
    assert config.training.forecast_control_scope == "prior"
    assert config.training.forecast_readout_loss_weight == 0.25
    assert config.runtime.process_group_timeout_s == 600
    assert config.runtime.dataloader_timeout_s == 600
    assert (
        config.model.history_hidden_dim,
        config.model.history_blocks,
        config.model.history_heads,
        config.model.history_ff_dim,
    ) == (1024, 16, 16, 4096)
    assert (
        config.model.latent_transition.hidden_dim,
        config.model.latent_transition.num_blocks,
        config.model.latent_transition.num_heads,
        config.model.latent_transition.ff_dim,
    ) == (768, 8, 12, 3072)


@pytest.mark.parametrize('maximum', [-1, 30, 31, 6.5, True])
def test_rtc_training_delay_rejects_invalid_or_full_horizon_bounds(tmp_path, maximum):
    with pytest.raises(ValueError, match='rtc_training_max_delay'):
        _config(tmp_path, rtc_training_max_delay=maximum).validate()


@pytest.mark.parametrize('maximum', [None, 0, 6, 29])
def test_rtc_training_delay_roundtrips_without_relabeling_legacy_configs(tmp_path, maximum):
    config = _config(tmp_path, rtc_training_max_delay=maximum)
    payload = json.loads(json.dumps(asdict(config)))
    if maximum is None:
        payload['training'].pop('rtc_training_max_delay')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(payload))
    loaded, actual = load_predictive_world_config(path)
    assert loaded.training.rtc_training_max_delay == maximum
    assert actual == payload


def test_batch_override_recomputes_global_batch_and_is_recorded(tmp_path):
    original = _config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(original)))
    loaded, payload = load_predictive_world_config(
        path,
        training_overrides={"micro_batch": 128, "gradient_accumulation_steps": 1},
    )
    assert loaded.training.micro_batch == 128
    assert loaded.training.effective_global_batch == 5_120
    assert payload["training"]["effective_global_batch"] == 5_120


def test_legacy_flow_config_cannot_silently_become_a_transition(tmp_path):
    payload = asdict(_config(tmp_path))
    payload["model"]["latent_flow"] = payload["model"].pop("latent_transition")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="legacy latent-flow config"):
        load_predictive_world_config(path)


def test_recurrent_config_requires_explicit_new_run(tmp_path):
    payload = asdict(_config(tmp_path))
    payload["model"]["latent_transition"]["semantics"] = "direct_recurrent_process"
    path = tmp_path / "recurrent.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported transition semantics"):
        load_predictive_world_config(path)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_control_loss_weight_rejects_nonfinite_or_negative(tmp_path, value):
    with pytest.raises(ValueError, match="loss weights"):
        _config(tmp_path, forecast_control_loss_weight=value).validate()


def test_response_scope_and_weights_are_explicit_and_overridable(tmp_path):
    with pytest.raises(ValueError, match="forecast_control_scope"):
        _config(tmp_path, forecast_control_scope="world").validate()
    alternate = _config(tmp_path, forecast_control_scope="both", forecast_control_loss_weight=0)
    alternate.validate()
    contract = predictive_world_loss_contract(alternate.training)
    assert contract["forecast_control_scope"] == "both"
    assert contract["forecast_control_loss_weight"] == 0
    assert contract["forecast_readout_parameters"] == "detached"
    assert contract["measurement_slots"] == "fixed_nominal_queries"


def test_phase_dependencies_and_global_batch_are_rejected_early(tmp_path):
    with pytest.raises(ValueError, match="world-model bundle"):
        _config(tmp_path, phase="policy").validate()
    with pytest.raises(ValueError, match="world-model bundle"):
        _config(
            tmp_path,
            phase="joint",
            normalizer_calibration_updates=0,
            forecast_warmup_updates=0,
            forecast_ramp_updates=0,
        ).validate()
    config = _config(tmp_path, effective_global_batch=320)
    with pytest.raises(ValueError, match="does not equal"):
        config.validate()


def test_observed_times_and_future_times_cannot_cross(tmp_path):
    config = _config(tmp_path)
    bad = PredictiveWorldConfig(
        data=PredictiveWorldDataConfig(
            history_visual_offsets_s=(-0.2, 0.01),
            measurement_registry="measurement.json",
            canonical_future_capabilities="capabilities.json",
            action_protocol_registry="actions.json",
            state_schema_registry="states.json",
            index_path="index.jsonl",
            feature_cache="features",
        ),
        model=config.model,
        training=config.training,
        runtime=config.runtime,
    )
    with pytest.raises(ValueError, match="end at the decision"):
        bad.validate()


def test_resume_override_allows_only_the_output_occupancy(tmp_path):
    config = _config(tmp_path)
    output = Path(config.runtime.output_dir)
    output.mkdir()
    (output / "checkpoint").write_text("owned")
    with pytest.raises(ValueError, match="new or empty"):
        config.validate()
    config.validate(allow_existing_output_override=True)


def test_alternating_forecast_requires_both_flow_objectives(tmp_path):
    config = _config(
        tmp_path, phase="world_model", prior_prediction_loss_weight=0
    )
    with pytest.raises(ValueError, match="positive prior and world"):
        config.validate()


def test_two_stage_policy_accepts_one_world_model_bundle(tmp_path):
    config = _config(tmp_path, phase="policy", world_model="world-model-final")
    config.validate()


def test_joint_continuation_requires_active_losses_and_no_recalibration(tmp_path):
    config = _config(
        tmp_path,
        phase="joint",
        world_model="world-model-final",
        normalizer_calibration_updates=0,
        forecast_warmup_updates=0,
        forecast_ramp_updates=0,
    )
    config.validate()
    with pytest.raises(ValueError, match="reuse the fitted"):
        _config(
            tmp_path,
            phase="joint",
            world_model="world-model-final",
        ).validate()
    with pytest.raises(ValueError, match="positive world and action"):
        _config(
            tmp_path,
            phase="joint",
            world_model="world-model-final",
            normalizer_calibration_updates=0,
            forecast_warmup_updates=0,
            forecast_ramp_updates=0,
            joint_action_loss_weight=0,
        ).validate()


def test_world_target_schedule_and_policy_mixture_are_validated(tmp_path):
    with pytest.raises(ValueError, match="finish before forecast warmup"):
        _config(
            tmp_path,
            normalizer_calibration_updates=501,
            forecast_warmup_updates=500,
        ).validate()
    with pytest.raises(ValueError, match="predicted latents"):
        _config(
            tmp_path,
            policy_latent_origin="mixed",
        ).validate()


def test_runtime_timeout_contract_is_positive_and_pg_covers_loader(tmp_path):
    config = _config(tmp_path)
    bad_runtime = PredictiveWorldRuntimeConfig(
        output_dir=str(tmp_path / "bad-timeout"),
        process_group_timeout_s=599,
        dataloader_timeout_s=600,
    )
    with pytest.raises(ValueError, match=">= DataLoader"):
        bad_runtime.validate()
    zero_runtime = PredictiveWorldRuntimeConfig(
        output_dir=str(tmp_path / "zero-timeout"),
        process_group_timeout_s=0,
    )
    with pytest.raises(ValueError, match="must be positive"):
        zero_runtime.validate()
    config.runtime.validate()


def test_monitoring_template_is_opt_in_and_preserves_joint_objective():
    root = Path(__file__).resolve().parents[2] / 'configs/predictive_world'
    old = json.loads((root / 'joint.template.json').read_text())
    new = json.loads((root / 'joint.monitoring-v1.template.json').read_text())
    old_training = PredictiveWorldTrainingConfig(**old['training'])
    new_training = PredictiveWorldTrainingConfig(**new['training'])
    new_training.validate(world_size=40)
    assert old_training.validation_every == 0 and not old_training.log_both_modes
    assert new_training.validation_every == 200 and new_training.log_both_modes
    assert predictive_world_loss_contract(old_training) == predictive_world_loss_contract(new_training)
    assert new['runtime']['output_dir'] != old['runtime']['output_dir']


@pytest.mark.parametrize('settings', [
    {'validation_every': -1}, {'validation_batches': 0}, {'validation_micro_batch': 0},
    {'validation_every': 200},
])
def test_invalid_or_unsupported_validation_config_is_rejected(tmp_path, settings):
    with pytest.raises(ValueError, match='validation'):
        _config(tmp_path, **settings).validate()


def test_joint_initialization_is_distinct_from_phase_init_and_resume(tmp_path):
    config = _config(tmp_path, phase='joint', init_joint='joint-12000', init_source_config='source.json',
                     normalizer_calibration_updates=0, forecast_warmup_updates=0,
                     forecast_ramp_updates=0, adam_betas=(.95,.999), lr_schedule='cosine', checkpoint_retention=3)
    config.validate()
    for changes in ({'init_source_config': None}, {'world_model': 'world-15000'}, {'phase': 'policy'}):
        from dataclasses import replace
        with pytest.raises(ValueError, match='joint initialization'):
            replace(config, training=replace(config.training, **changes)).validate()
    for changes, error in (({'adam_betas': (1.,.999)}, 'Adam betas'),
                            ({'lr_schedule': 'unknown'}, 'learning-rate schedule'),
                            ({'checkpoint_retention': 0}, 'checkpoint_retention')):
        from dataclasses import replace
        with pytest.raises(ValueError, match=error):
            replace(config, training=replace(config.training, **changes)).validate()
