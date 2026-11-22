import pytest
import torch

from psi.models.predictive_world.history import PredictiveHistoryEncoder


def _model():
    torch.manual_seed(19)
    return PredictiveHistoryEncoder(
        qwen_dim=6,
        visual_dim=6,
        state_dim=4,
        state_history_count=5,
        output_tokens=3,
        context_tokens=4,
        hidden_dim=16,
        num_blocks=2,
        num_heads=4,
        ff_dim=32,
    ).eval()


def _inputs():
    return {
        "qwen_context": torch.randn(2, 7, 6),
        "context_valid": torch.ones(2, 7, dtype=torch.bool),
        "current_visual": torch.randn(2, 3, 6),
        "adapted_states": torch.randn(2, 5, 4),
        "state_valid": torch.ones(2, 5, dtype=torch.bool),
        "state_times": torch.linspace(-0.4, 0, 5)[None].expand(2, -1),
    }


def test_history_encoder_produces_reusable_condition_tokens():
    model = _model()
    with torch.no_grad():
        condition = model(**_inputs())
    assert condition.shape == (2, 3, 16)
    assert torch.isfinite(condition).all()


def test_masked_qwen_nan_is_sanitized_before_projection():
    model = _model()
    inputs = _inputs()
    inputs["context_valid"][:, -1] = False
    inputs["qwen_context"][:, -1] = float("nan")
    with torch.no_grad():
        condition = model(**inputs)
    assert torch.isfinite(condition).all()


def test_history_encoder_rejects_future_state_time():
    inputs = _inputs()
    inputs["state_times"] = inputs["state_times"].clone()
    inputs["state_times"][0, -1] = 0.01
    with pytest.raises(ValueError, match="causal"):
        _model()(**inputs)


def test_legacy_state_positions_map_by_physical_tail_not_interpolation():
    model = PredictiveHistoryEncoder(
        qwen_dim=6,
        visual_dim=6,
        state_dim=4,
        state_history_count=21,
        output_tokens=3,
        context_tokens=4,
        hidden_dim=16,
        num_blocks=1,
        num_heads=4,
        ff_dim=32,
    )
    legacy = torch.randn(8, 16)
    model.load_legacy_state_positions(legacy)
    assert torch.equal(model.state_history_position[-8:], legacy)
    assert torch.equal(model.state_history_position[:-8], torch.zeros(13, 16))
