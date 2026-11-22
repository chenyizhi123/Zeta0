"""Source-bound post-training objective changes, separate from exact resume."""

from __future__ import annotations

from dataclasses import fields, replace

from psi.config.predictive_world import (
    PredictiveWorldTrainingConfig,
    effective_objective_fields,
    predictive_world_loss_contract,
)
from psi.data.predictive_world.cache import canonical_sha256

OBJECTIVE_TRANSITION_FORMAT = "zeta0.post_training_objective_transition.v1"


def full_ert_training(training: PredictiveWorldTrainingConfig) -> PredictiveWorldTrainingConfig:
    """Derive the v5 dual-response recipe from FULL."""
    defaults = PredictiveWorldTrainingConfig()
    for name in (
        "forecast_control_target", "preserve_control_sampling", "body_auxiliary_enabled",
        "student_latent_gradient", "student_latent_mode", "forecast_control_reader",
    ):
        if getattr(training, name) != getattr(defaults, name):
            raise ValueError("variant preparation requires a FULL objective")
    if training.forecast_control_loss_weight <= 0 or training.temporal_loss_weight <= 0:
        raise ValueError("FULL recipe must contain response and temporal supervision")
    if training.objective_contract_version == 3:
        raise ValueError("executor response transfer requires the original latent architecture")
    if training.forecast_control_loss_weight != 1.0:
        raise ValueError("FULL+ERT requires outer response loss weight 1 with fixed half reader weights")
    return replace(training, objective_contract_version=5, forecast_control_reader="both",
                   init_objective_migration=None)


def make_objective_transition(source: dict, source_payload: dict,
                              target: PredictiveWorldTrainingConfig, variant_id: str) -> dict:
    """Bind the permitted objective delta to one verified source/config identity.

    Bundle reading and registry/architecture validation are owned by initialization;
    this function never grants permission to change those other boundaries.
    """
    if source.get("phase") != "joint" or target.phase != "joint":
        raise ValueError("objective transition requires joint source and target")
    if target.objective_contract_version == 6:
        return make_joint_objective_transition(source, source_payload, target, variant_id)
    if target.objective_contract_version != 5 or variant_id != "FULL+ERT":
        raise ValueError("objective transition requires the version-5 FULL+ERT recipe")
    metadata = source["metadata"]
    if canonical_sha256(source_payload) != metadata["resolved_config_sha256"]:
        raise ValueError("objective transition source config identity mismatch")
    source_training = PredictiveWorldTrainingConfig(**source_payload["training"])
    source_contract = predictive_world_loss_contract(source_training)
    if source_contract != metadata.get("loss_contract"):
        raise ValueError("objective transition source loss contract disagrees with config")
    expected = full_ert_training(source_training)
    source_fields = effective_objective_fields(source_training)
    target_fields = effective_objective_fields(target)
    if target_fields != effective_objective_fields(expected):
        raise ValueError("objective transition contains undeclared changes")
    delta_source_fields = {
        "forecast_control_reader": source_training.forecast_control_reader,
        **source_fields,
    }
    return {
        "format": OBJECTIVE_TRANSITION_FORMAT,
        "variant_id": variant_id,
        "source_weights_sha256": source["weights_sha256"],
        "source_config_sha256": metadata["resolved_config_sha256"],
        "source_loss_contract": source_contract,
        "source_effective_objective": source_fields,
        "target_loss_contract": predictive_world_loss_contract(target),
        "changed_fields": {
            name: {"source": delta_source_fields[name], "target": value}
            for name, value in target_fields.items()
            if delta_source_fields[name] != value
        },
        "optimizer_restored": False,
    }


def make_joint_objective_transition(source, source_payload, target, variant_id="pretrained"):
    """Declare v6 separately from the raw output-coordinate migration."""
    if (variant_id != "pretrained" or target.objective_contract_version != 6
            or source.get("phase") != "joint" or target.phase != "joint"
            or canonical_sha256(source_payload) != source["metadata"]["resolved_config_sha256"]):
        raise ValueError("v6 objective migration requires the exact pretrained joint source")
    values = dict(source_payload["training"])
    # The imported 29k source records this original spelling. This is a weight transfer.
    if "from_scratch" in values:
        if "joint_from_scratch" in values or type(values["from_scratch"]) is not bool:
            raise ValueError("ambiguous source scratch identity")
        values["joint_from_scratch"] = values.pop("from_scratch")
    if values.keys() - {item.name for item in fields(PredictiveWorldTrainingConfig)}:
        raise ValueError("unsupported source training fields in v6 objective migration")
    source_training = PredictiveWorldTrainingConfig(**values)
    if source_training.phase != "joint":
        raise ValueError("source configuration must describe joint training")
    source_contract = predictive_world_loss_contract(source_training)
    if source_contract != source["metadata"].get("loss_contract"):
        raise ValueError("source loss contract disagrees with its config")
    if not target.init_joint or not target.init_source_config or source_training.world_model:
        raise ValueError("v6 objective migration requires joint checkpoint initialization")
    expected = replace(source_training,
        objective_contract_version=6, latent_prediction_space="raw",
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0,
        forecast_control_reader="both", forecast_control_loss_weight=1.0,
        forecast_control_scope="prior", forecast_control_target="teacher_response",
        preserve_control_sampling=True, body_auxiliary_enabled=True, future_body_input_enabled=True,
        body_loss_weight=1.0, temporal_loss_weight=.25, student_latent_gradient="joint",
        student_latent_mode="predicted", prior_prediction_loss_weight=1.0,
        world_prediction_loss_weight=1.0, joint_from_scratch=False, joint_from_gr00t=False,
        init_joint=target.init_joint, init_source_config=target.init_source_config)
    denominator = expected.micro_batch * expected.gradient_accumulation_steps
    if denominator <= 0 or expected.effective_global_batch % denominator:
        raise ValueError("v6 objective migration source batch cannot be partitioned across ranks")
    expected.validate(world_size=expected.effective_global_batch // denominator)
    if effective_objective_fields(target) != effective_objective_fields(expected):
        raise ValueError("v6 objective transition contains undeclared changes")
    return dict(format="zeta0.joint_training_objective_transition.v1", variant_id=variant_id,
        source_weights_sha256=source["weights_sha256"],
        source_config_sha256=source["metadata"]["resolved_config_sha256"],
        source_loss_contract=source_contract, target_loss_contract=predictive_world_loss_contract(target),
        source_effective_objective=effective_objective_fields(source_training),
        target_effective_objective=effective_objective_fields(target), optimizer_restored=False)


def verify_objective_transition(record: dict, source: dict, source_payload: dict,
                                target: PredictiveWorldTrainingConfig) -> dict:
    if not isinstance(record, dict):
        raise ValueError("objective transition must be a mapping")
    expected = make_objective_transition(source, source_payload, target, record.get("variant_id"))
    if record != expected:
        raise ValueError("objective transition identity or allowed change mismatch")
    return expected
