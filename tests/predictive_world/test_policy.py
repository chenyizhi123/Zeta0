import pytest
import torch
from dataclasses import replace
from torch import nn

from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.data.predictive_world.contracts import NativeActionBatch, ObservedHistory
from psi.models.predictive_world.action_expert import TemporalNativeActionExpert
from psi.models.predictive_world.history import PredictiveHistoryEncoder
from psi.models.predictive_world.latent_transition import LatentNormalizer, TemporalLatentTransition
from psi.models.predictive_world.policy import ObservationFeatures, PredictiveWorldPolicy


class _ObservationEncoder(nn.Module):
    def forward(self, observed):
        batch = observed.raw_states.shape[0]
        return ObservationFeatures(
            qwen_context=torch.ones(batch, 5, 6),
            context_valid=torch.ones(batch, 5, dtype=torch.bool),
            current_visual=torch.ones(batch, 3, 6),
        )


def _policy(**kwargs):
    protocols = {"toy": {"dim": 2, "mean": [0.0, 0.0], "std": [1.0, 1.0]}}
    history_adapters = StateAdapterBank({"state": 3})
    history = PredictiveHistoryEncoder(
        qwen_dim=6, visual_dim=6, state_dim=64, state_history_count=5,
        output_tokens=3, context_tokens=4, hidden_dim=16, num_blocks=1,
        num_heads=4, ff_dim=32,
    )
    flow = TemporalLatentTransition(
        protocols, context_dim=16, latent_shape=(3, 2, 4), hidden_dim=16,
        num_blocks=1, num_heads=4, ff_dim=32,
    )
    normalizer = LatentNormalizer((3, 2, 4))
    normalizer.fit(torch.randn(4, 3, 2, 4), torch.ones(4, 3, dtype=torch.bool))
    student = TemporalNativeActionExpert(
        protocols, {"state": 3}, qwen_dim=6, context_tokens=3,
        latent_dim=4, latent_groups=3, latent_slots=2, state_history_count=5,
        action_steps=4, hidden_dim=24, num_blocks=1, num_heads=4, ff_dim=48,
    )
    return PredictiveWorldPolicy(
        observation_encoder=_ObservationEncoder(),
        history_state_adapters=history_adapters,
        history_encoder=history,
        latent_transition=flow,
        latent_normalizer=normalizer,
        student=student,
        action_solver_steps=1,
        **kwargs,
    ).eval()


def test_joint_initialization_export_has_no_world_model_dependency():
    import runpy
    from pathlib import Path
    from psi.config.predictive_world import PredictiveWorldConfig, PredictiveWorldTrainingConfig
    export = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/export_predictive_world_policy.py"))
    config = PredictiveWorldConfig(training=PredictiveWorldTrainingConfig(
        phase="joint", init_joint="joint-source", init_source_config="source.json",
        objective_contract_version=2))
    expected = export["export_expected_metadata"](config)
    assert expected["loss_contract"]["student_latent_mode"] == "predicted"
    assert "reference_id" not in expected


def _observed():
    return ObservedHistory(
        qwen_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
        raw_states=torch.zeros(1, 5, 3),
        state_valid=torch.ones(1, 5, 3, dtype=torch.bool),
        state_times=torch.linspace(-0.4, 0, 5)[None],
        state_schema_ids=("state",),
        action_protocol_ids=("toy",),
        action_native_dims=torch.tensor([2]),
        action_normalization_spaces=("native",),
    )


def test_deploy_policy_runs_only_from_observed_history_and_explicit_noise():
    policy = _policy()
    result = policy.sample(
        _observed(),
        action_noise=torch.randn(1, 4, 2),
    )
    assert result.shape == (1, 4, 2)
    assert torch.isfinite(result).all()


def test_deploy_policy_can_emit_opt_in_stage_timings():
    policy = _policy()
    timings: dict[str, float] = {}
    result = policy.sample(
        _observed(),
        action_noise=torch.randn(1, 4, 2),
        stage_timings=timings,
    )
    assert result.shape == (1, 4, 2)
    assert {
        "validate_s",
        "observation_encoder_s",
        "history_encoder_s",
        "latent_transition_s",
        "action_expert_s",
        "total_s",
    } <= set(timings)
    assert all(value >= 0.0 for value in timings.values())


def test_deploy_policy_rejects_non_observed_objects():
    policy = _policy()
    with pytest.raises(TypeError, match="ObservedHistory only"):
        policy.sample(
            object(),
            action_noise=torch.randn(1, 4, 2),
        )


def test_empty_committed_prefix_is_equivalent_to_no_prefix():
    policy = _policy()
    observed = _observed()
    empty = NativeActionBatch(
        values=torch.zeros(1, 4, 2),
        valid=torch.zeros(1, 4, 2, dtype=torch.bool),
        times=torch.zeros(1, 4),
        protocol_ids=("toy",),
        native_dims=torch.tensor([2]),
        normalization_spaces=("native",),
    )
    action_noise = torch.randn(1, 4, 2)
    without = policy.sample(
        observed, action_noise=action_noise.clone()
    )
    with_empty = policy.sample(
        replace(observed, committed_actions=empty),
        action_noise=action_noise.clone(),
    )
    torch.testing.assert_close(without, with_empty)
