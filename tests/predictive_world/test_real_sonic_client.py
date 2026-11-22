"""Deterministic control-loop checks using the real feedback and action queue."""

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from psi.deploy import real_sonic_client
from psi.deploy.real_sonic_io import SonicZmqIO
from psi.deploy.real_sonic_protocol import (
    CAMERAS,
    FORMAT,
    PERIOD,
    PROTOCOL,
    STATE_SCHEMA,
    MeasuredHistory,
)


class Clock:
    def __init__(self):
        self.now = 1.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ImmediateWorker:
    """Remove scheduling nondeterminism; network responses keep the real envelope."""
    def __init__(self, **kwargs):
        pass

    def submit(self, function, request):
        future = Future()
        try:
            future.set_result(function(request))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


class Sensors(SonicZmqIO):
    """Supply synthetic sensor receipts through the real observation boundary."""
    def __init__(self, clock, *, vision=lambda now: True, feedback_failure=None, missed_period=None):
        self.clock = clock
        self.vision = vision
        self.feedback_failure = feedback_failure
        self.missed_period = missed_period
        self.body_sample = None
        self.hand_samples = [None, None]
        self.history = MeasuredHistory(state_count=1)
        self.images = {name: deque(maxlen=8) for name in CAMERAS}
        self.rgb = np.zeros((480,640,3), np.uint8)
        self.commands = []
        self.feedback_checks = []
        self.observation_queries = []
        self.stopped = self.closed = False

    def poll(self, now):
        self.body_sample = (now-.002, np.zeros(29, np.float32))
        self.hand_samples = [(now-.002, np.zeros(6, np.float32), None)]*2
        self.history.append(now-.002, np.zeros(41, np.float32), component_times=[now-.002]*3)
        if self.vision(now):
            for samples in self.images.values():
                samples.append((now-.002, self.rgb))
        if self.feedback_failure and now >= 1.14:
            if self.feedback_failure == "body":
                self.body_sample = (now-.05, self.body_sample[1])
            else:
                side = 0 if self.feedback_failure == "left_hand" else 1
                self.hand_samples[side] = (now-.05, np.zeros(6, np.float32), None)
        if self.missed_period and now >= 1.14:
            self.clock.now += self.missed_period
            self.missed_period = None

    def measured_state(self, anchor):
        self.feedback_checks.append(anchor)
        return super().measured_state(anchor)

    def observation(self, anchor):
        self.observation_queries.append(anchor)
        return super().observation(anchor)

    def execute(self, action):
        self.commands.append((self.clock.monotonic(), action.copy()))

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class Transport:
    def __init__(self):
        self.requests = []

    def call(self, route, payload=None):
        if route == "/health":
            return dict(format=FORMAT, action_protocol_id=PROTOCOL, state_schema_id=STATE_SCHEMA,
                        action_hz=50, action_horizon=30, native_dim=76,
                        state_history_count=1, rtc_max_prefix_steps=15)
        assert route == "/reset"
        return payload

    def infer(self, request):
        self.requests.append(request)
        actions = np.full((30,76), .25, np.float32)
        actions[:,64:] = (np.arange(30, dtype=np.float32)[:,None]+1)/32
        prefix = request["committed_prefix"]
        if prefix:
            actions[:len(prefix)] = prefix
        reply = {name: request[name] for name in
                 ("format", "session_id", "request_seq", "queue_revision", "anchor_time", "action_protocol_id")}
        return dict(reply, actions=actions.tolist(), inference_seconds=0.)


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(real_sonic_client, "time", value)
    monkeypatch.setattr(real_sonic_client, "ThreadPoolExecutor", ImmediateWorker)
    return value


def test_visual_pause_consumes_existing_actions_and_recovers_with_full_rtc_prefix(clock):
    io = Sensors(clock, vision=lambda now: not 1.08 <= now < 1.26)
    transport = Transport()
    real_sonic_client.run(io, transport, task="doll", rtc_prefix_steps=15,
                          low_watermark=20, duration=.7)
    assert io.stopped and io.closed
    assert len(transport.requests) >= 2
    assert len(transport.requests[1]["committed_prefix"]) == 15
    assert transport.requests[1]["anchor_time"] >= 1.26
    # Image age exceeds 80ms during this interval, yet fresh feedback lets the
    # already committed commands execute at every 20ms slot.
    paused = [when for when, action in io.commands if 1.2 <= when < 1.26]
    assert len(paused) == 3
    assert np.allclose(np.diff(paused), PERIOD)
    assert len(io.feedback_checks) > len(io.observation_queries)


def test_camera_loss_stops_when_committed_queue_exhausts_without_repeating_actions(clock):
    io = Sensors(clock, vision=lambda now: now < 1.08)
    transport = Transport()
    with pytest.raises(RuntimeError, match="Action queue exhausted"):
        real_sonic_client.run(io, transport, task="doll", rtc_prefix_steps=15, low_watermark=20)
    assert io.stopped and io.closed
    assert len(transport.requests) == 1
    assert len(io.commands) == 29  # The first returned slot expired before arrival.
    assert np.allclose(np.diff([when for when, _ in io.commands]), PERIOD)
    assert np.array_equal([action[64] for _, action in io.commands], np.arange(2,31)/32)


@pytest.mark.parametrize("component", ["body", "left_hand", "right_hand"])
def test_stale_feedback_stops_before_next_command_even_with_a_full_queue(clock, component):
    io = Sensors(clock, vision=lambda now: now < 1.08, feedback_failure=component)
    transport = Transport()
    with pytest.raises(RuntimeError, match="feedback older than 40ms"):
        real_sonic_client.run(io, transport, task="doll", rtc_prefix_steps=15, low_watermark=20)
    assert io.stopped and io.closed
    assert io.commands and io.commands[-1][0] < 1.14
    assert len(io.commands) < 29


@pytest.mark.parametrize("component", ["body", "left_hand", "right_hand", "causal_history"])
def test_feedback_boundary_rejects_missing_components_and_noncausal_current_state(clock, component):
    io = Sensors(clock)
    io.poll(1.)
    if component == "body":
        io.body_sample = None
    elif component in ("left_hand", "right_hand"):
        io.hand_samples[0 if component == "left_hand" else 1] = None
    else:
        io.history = MeasuredHistory(state_count=1)
        io.history.append(1.001, np.zeros(41, np.float32))
    with pytest.raises(ValueError, match="Waiting for measured|No causal current"):
        io.measured_state(1.)


@pytest.mark.parametrize("old_component", [0, 1, 2, None])
def test_selected_causal_snapshot_checks_components_even_when_latest_feedback_is_fresh(clock, old_component):
    io = Sensors(clock)
    io.body_sample = (1.005, np.ones(29, np.float32))
    io.hand_samples = [(1.005, np.ones(6, np.float32), None)]*2
    components = [.965,.97,.969]
    if old_component is not None:
        components[old_component] = .935
    # Keep the combined max at .97 independently of which component is stale.
    components[(old_component+1)%3 if old_component is not None else 1] = .97
    io.history.append(.97, np.zeros(41, np.float32), component_times=components)
    io.history.append(1.005, np.ones(41, np.float32), component_times=[1.005]*3)
    if old_component is not None:
        with pytest.raises(ValueError, match="No causal current measured state"):
            io.measured_state(1.)
    else:
        state = io.measured_state(1.)
        assert state["values"] == [[0.]*41]
        assert state["valid"] == [[True]*41]
        assert state["times"] == pytest.approx([-.03])


def test_missed_control_period_still_stops_instead_of_catching_up(clock):
    io = Sensors(clock, missed_period=.021)
    with pytest.raises(RuntimeError, match="missed a full 20ms period"):
        real_sonic_client.run(io, Transport(), task="doll", rtc_prefix_steps=15, low_watermark=20)
    assert io.stopped and io.closed
    assert io.commands[-1][0] < 1.14


def test_feedback_processing_delay_cannot_send_an_expired_legacy_action(clock):
    class DelayedFeedback(Sensors):
        def check_feedback(self, anchor):
            super().check_feedback(anchor)
            if anchor >= 1.14:
                clock.now += .041

    io = DelayedFeedback(clock)
    with pytest.raises(RuntimeError, match="missed a full 20ms period"):
        real_sonic_client.run(io, Transport(), task="doll", duration=.3)
    assert io.commands[-1][0] < 1.14
    assert io.stopped and io.closed


@pytest.mark.parametrize("failure", ["health", "configure", "reset", "duration"])
def test_early_setup_failure_closes_io_without_starting_control(clock, failure):
    class BrokenSetup(Transport):
        def call(self, route, payload=None):
            result = super().call(route, payload)
            if failure == "health" and route == "/health":
                result["action_hz"] = 10
            if failure == "reset" and route == "/reset":
                return {}
            return result

    io = Sensors(clock)
    if failure == "configure":
        io.configure_observation_contract = lambda _: (_ for _ in ()).throw(ValueError("bad configuration"))
    with pytest.raises(ValueError):
        real_sonic_client.run(io, BrokenSetup(), task="doll", duration=float("nan") if failure == "duration" else .1)
    assert io.closed and not io.commands


def test_legacy_adapter_retains_combined_observation_checks_every_tick(clock):
    sensors = Sensors(clock, vision=lambda now: now < 1.08)
    legacy = SimpleNamespace(history=sensors.history, poll=sensors.poll,
                             observation=sensors.observation, execute=sensors.execute,
                             stop=sensors.stop, close=sensors.close)
    with pytest.raises(RuntimeError, match="Sensor input interrupted: Missing or stale trained camera"):
        real_sonic_client.run(legacy, Transport(), task="doll", rtc_prefix_steps=15, low_watermark=20)
    assert sensors.stopped and sensors.closed
    assert 0 < len(sensors.commands) < 29
