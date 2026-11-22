from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from psi.data.predictive_world.contracts import ObservedHistory
from psi.deploy.predictive_world_history import (
    CommittedActionPrefix,
    PredictiveWorldHistoryBuffer,
    PredictiveWorldRequest,
    TimedImage,
    TimedState,
)
from psi.deploy.predictive_world_service import PredictiveWorldService, QwenRequestAdapter


class _Policy:
    latent_transition = SimpleNamespace(latent_shape=(3, 2, 4))
    student = SimpleNamespace(
        action_steps=4,
        adapters={"toy": SimpleNamespace(dim=2)},
    )

    def sample(self, _observed, *, action_noise):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled()
        return torch.zeros_like(action_noise)


class _Backend:
    def __init__(self):
        self.revisions = []

    def validate_and_publish(self, actions, *, action_grid_start_time, action_frequency,
                             expected_queue_revision):
        assert actions.shape == (4, 2) and action_frequency == 50
        self.revisions.append(expected_queue_revision)
        return {"published": True}


def _request(revision=0, sequence=0):
    anchor = 10.0
    return PredictiveWorldRequest(
        session_id="session", episode_id="episode", request_seq=sequence,
        observation_anchor_time=anchor, clock_domain="robot", clock_mapping_id="map",
        task_text="move", task_revision="one",
        images=tuple(
            TimedImage("primary", anchor + offset, str(index), b"unused")
            for index, offset in enumerate((-0.4, -0.2, 0.0))
        ),
        states=tuple(
            TimedState(anchor + offset, index, "state", (1.0, 2.0), (True, True))
            for index, offset in enumerate(np.linspace(-0.4, 0, 21))
        ),
        action_protocol_id="toy", action_frequency=50,
        action_grid_start_time=anchor, native_action_dim=2,
        committed_prefix=CommittedActionPrefix(
            revision, (), (), (), anchor
        ),
    )


def _adapt(_request, _window):
    return ObservedHistory(
        qwen_inputs={}, raw_states=torch.zeros(1, 21, 2),
        state_valid=torch.ones(1, 21, 2, dtype=torch.bool),
        state_times=torch.linspace(-0.4, 0, 21).unsqueeze(0),
        state_schema_ids=("state",), action_protocol_ids=("toy",),
        action_native_dims=torch.tensor([2]), action_normalization_spaces=("native",),
    )


def test_service_publishes_only_through_backend_and_advances_revision():
    backend = _Backend()
    service = PredictiveWorldService(
        _Policy(), _adapt, PredictiveWorldHistoryBuffer(),
        device=torch.device("cpu"), backend=backend,
    )
    response, receipt = service.infer_and_publish(
        _request(), receive_time=10.01, seed=1
    )
    assert receipt == {"published": True} and response.queue_revision == 0
    assert backend.revisions == [0]
    with pytest.raises(RuntimeError, match="stale"):
        service.infer(_request(revision=0, sequence=1), receive_time=10.01, seed=1)


def test_service_refuses_publish_without_backend():
    service = PredictiveWorldService(
        _Policy(), _adapt, PredictiveWorldHistoryBuffer(),
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="backend"):
        service.infer_and_publish(_request(), receive_time=10.01, seed=1)


class _Processor:
    """Small image/text fixture; the tested action boundary is not mocked."""

    def __init__(self):
        self.image_processor = self.encode_images

    def apply_chat_template(self, *_args, **_kwargs):
        return "move"

    def encode_images(self, *, images, **_kwargs):
        return {
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
            "pixel_values": torch.zeros(len(images) * 4, 3),
        }

    def __call__(self, *, images, **_kwargs):
        return {
            **self.encode_images(images=images),
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }


def _sonic_request_and_window(*, prefix_count=6, anchor=0.88):
    request = _request()
    prefix = CommittedActionPrefix(
        0,
        tuple(anchor + index / 50 for index in range(prefix_count)),
        tuple((float(index),) * 40 for index in range(prefix_count)),
        tuple((True,) * 40 for _ in range(prefix_count)),
        anchor,
    )
    request = replace(
        request,
        observation_anchor_time=anchor,
        action_grid_start_time=anchor,
        action_protocol_id="sonic_semantic40_v31",
        native_action_dim=40,
        images=(TimedImage("primary", anchor, "current", b"", raw_rgb=np.zeros((32, 32, 3), dtype=np.uint8)),),
        states=tuple(
            TimedState(anchor + index / 50, index + 20, "sonic_state64_v31", (0.0,) * 64, (True,) * 64)
            for index in range(-20, 1)
        ),
        committed_prefix=prefix,
    )
    window = PredictiveWorldHistoryBuffer(observed_visual_mode="current_multiview").build(
        request, receive_time=anchor
    )
    return request, window


def _sonic_adapter(action_horizon=30, **kwargs):
    return QwenRequestAdapter(
        _Processor(), image_preprocessing="aspect_v1", action_horizon=action_horizon,
        normalization_spaces={"sonic_semantic40_v31": "native"},
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["legacy_command_time_v1", "training_action_grid_v1"])
@pytest.mark.parametrize("prefix_count", [0, 1, 6])
@pytest.mark.parametrize("horizon", [30, 40, 50])
def test_rtc_adapter_versioned_model_times_preserve_physical_queue(mode, prefix_count, horizon):
    request, window = _sonic_request_and_window(prefix_count=prefix_count)
    original_prefix = request.committed_prefix

    observed = _sonic_adapter(action_horizon=horizon, rtc_model_time_mode=mode)(request, window)

    prefix = observed.committed_actions
    assert prefix.values.shape == prefix.valid.shape == (1, horizon, 40)
    assert prefix.times.shape == (1, horizon)
    expected_times = (
        torch.tensor(original_prefix.command_times) - request.observation_anchor_time
        if mode == "legacy_command_time_v1"
        else torch.tensor([index / 50 for index in range(1, prefix_count + 1)])
    )
    torch.testing.assert_close(prefix.times[0, :prefix_count], expected_times, rtol=0, atol=0)
    assert prefix.times[0, prefix_count:].eq(0).all()
    assert prefix.valid[0, :prefix_count].all()
    assert not prefix.valid[0, prefix_count:].any()
    assert prefix.values[0, prefix_count:].eq(0).all()
    assert request.committed_prefix is original_prefix
    assert window.committed_prefix is original_prefix
    assert request.action_grid_start_time == request.observation_anchor_time


@pytest.mark.parametrize("prefix_count", [0, 6])
@pytest.mark.parametrize("horizon", [20, 29, 31, 39, 41, 49, 51, 60])
def test_training_grid_rejects_unsupported_model_horizons_even_on_cold_start(horizon, prefix_count):
    request, window = _sonic_request_and_window(prefix_count=prefix_count)
    with pytest.raises(ValueError, match="30-, 40- or 50-step"):
        _sonic_adapter(action_horizon=horizon, rtc_model_time_mode="training_action_grid_v1")(request, window)


@pytest.mark.parametrize("prefix_count", [0, 6])
@pytest.mark.parametrize("change", [
    {"action_frequency": 25.0}, {"action_frequency": 51.0},
    {"native_action_dim": 39}, {"native_action_dim": 41},
    {"action_protocol_id": "toy"}, {"action_protocol_id": "sonic_semantic76"},
])
def test_native40_training_grid_preserves_frequency_width_and_protocol_guards(prefix_count, change):
    request, window = _sonic_request_and_window(prefix_count=prefix_count)
    with pytest.raises(ValueError, match="SONIC semantic40 50 Hz"):
        _sonic_adapter(action_horizon=40, rtc_model_time_mode="training_action_grid_v1")(
            replace(request, **change), window)


def test_default_rtc_adapter_retains_historical_coordinates():
    request, window = _sonic_request_and_window()
    actual = _sonic_adapter()(request, window).committed_actions
    expected = _sonic_adapter(rtc_model_time_mode="legacy_command_time_v1")(request, window).committed_actions
    torch.testing.assert_close(actual.times, expected.times, rtol=0, atol=0)
    assert actual.times[0, 0] == 0


def test_training_grid_rtc_coordinates_avoid_large_clock_cancellation():
    request, window = _sonic_request_and_window(anchor=1_000_000.0)
    actual = _sonic_adapter(rtc_model_time_mode="training_action_grid_v1")(request, window)
    torch.testing.assert_close(
        actual.committed_actions.times[0, :6],
        torch.tensor([index / 50 for index in range(1, 7)]), rtol=0, atol=0,
    )


def test_training_grid_adapter_rejects_malformed_or_unsupported_queue():
    request, window = _sonic_request_and_window()
    adapter = _sonic_adapter(rtc_model_time_mode="training_action_grid_v1")
    prefix = replace(window.committed_prefix, command_times=(0.881,) + window.committed_prefix.command_times[1:])
    with pytest.raises(ValueError, match="grid"):
        adapter(request, replace(window, committed_prefix=prefix))
    with pytest.raises(ValueError, match="50 Hz"):
        adapter(replace(request, action_frequency=25.0), window)
    with pytest.raises(ValueError, match="command grid"):
        adapter(replace(request, action_grid_start_time=0.90), window)
    with pytest.raises(ValueError, match="time mode"):
        _sonic_adapter(rtc_model_time_mode="unknown")
