"""Explicit initialization identity for the no-Zeta-0-pretraining control."""
from __future__ import annotations

from psi.config.predictive_world import PredictiveWorldTrainingConfig

RANDOM_JOINT_FORMAT = "oraclevla.predictive_world.random_joint_initialization.v1"


def random_joint_initialization(training: PredictiveWorldTrainingConfig) -> dict:
    """Describe random task modules without inspecting/loading any bundle."""
    if not training.joint_from_scratch:
        raise ValueError("random initialization needs an explicit scratch configuration")
    denominator = training.micro_batch * training.gradient_accumulation_steps
    if denominator <= 0 or training.effective_global_batch % denominator:
        raise ValueError("scratch batch identity is invalid")
    training.validate(world_size=training.effective_global_batch // denominator)
    return {
        "format": RANDOM_JOINT_FORMAT,
        "kind": "random_task_modules",
        "seed": training.seed,
        "pretrained_task_weights_loaded": False,
        "optimizer_restored": False,
        "optimizer_update_offset": 0,
        "ema_initialization": "copy_new_online_encoder",
        "student_initialization": "independent_copy_of_random_teacher",
        "normalizer": "fresh_train_only_ema",
        "normalizer_calibration_updates": training.normalizer_calibration_updates,
        "forecast_warmup_updates": training.forecast_warmup_updates,
        "forecast_ramp_updates": training.forecast_ramp_updates,
    }


def scratch_training_from_reference(training: PredictiveWorldTrainingConfig) -> PredictiveWorldTrainingConfig:
    """Keep the final task recipe, changing only initialization and its startup."""
    from dataclasses import replace

    if (training.phase != "joint" or training.objective_contract_version != 5
            or training.forecast_control_reader != "both" or training.joint_from_scratch):
        raise ValueError("scratch control requires the pretrained FULL+ERT task recipe")
    result = replace(training, world_model=None, init_joint=None, init_source_config=None,
        init_processor_diagnostic=None, init_processor_migration=None, init_objective_migration=None,
        init_goal_intent=None, init_state_history_migration=None, init_observation_contract_migration=None,
        init_body_target_migration=None, joint_from_scratch=True,
        normalizer_calibration_updates=500, forecast_warmup_updates=500, forecast_ramp_updates=500)
    random_joint_initialization(result)
    return result
