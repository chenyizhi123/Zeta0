"""Collector-mode asynchronous snapshots and bounded control-loop recovery."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_real_sonic_client import Clock, ImmediateWorker, Sensors, Transport
from test_real_sonic_deployment import request, response

from psi.deploy import real_sonic_client
from psi.deploy.real_sonic_io import SonicIOConfig, SonicZmqIO
from psi.deploy.real_sonic_protocol import (
    CAMERAS,
    PERIOD,
    AbsoluteActionQueue,
    validate_request,
)
from psi.real_observation_contract import (
    build_recorded_action_projection,
    make_deployment_observation_contract,
)


def contract(*, projection=False):
    data = dict(real_observation_mode="collector_snapshot_v1",
        real_observation_contract=dict(version=1, camera_max_skew_ms=25., camera_wait_ms=20., max_age_ms=200.,
                                       body_max_offset_ms=40., hand_max_offset_ms=50., history_size=128,
                                       image_shape=[480,640,3]),
        state_history_count=1, state_max_age_s=.2, visual_max_age_s=.2)
    if projection:
        data["real_action_projection"] = build_recorded_action_projection(-.75,.6875,"a"*64)
    return make_deployment_observation_contract(data)


def bare_io():
    io = object.__new__(SonicZmqIO)
    io.config = SonicIOConfig(state_history_count=1)
    io.history = SimpleNamespace(state_count=1)
    io.body_history = deque(maxlen=128)
    io.hand_histories = [deque(maxlen=128), deque(maxlen=128)]
    io.camera_bundles = deque(maxlen=128)
    io.body_sample = None
    io.hand_samples = [None,None]
    io.configure_observation_contract(contract())
    return io


def bundle_packet():
    return dict(images={"ego_view": b"ego", "video6": b"left", "video8": b"right"},
        timing=dict(receipt_clock="client_monotonic", cameras={
            "ego_view": dict(received_monotonic=.88), "video6": dict(received_monotonic=.892),
            "video8": dict(received_monotonic=.875)}))


def paired_io():
    io = bare_io()
    frames, times = io.decode_camera_bundle(bundle_packet(), .94)
    io.camera_bundles.append((.94, frames, times, "bundle"))
    # Fresh latest feedback differs from the feedback paired with the ego image.
    for stamp in (.872,.95):
        io.body_history.append((stamp, np.full(29,stamp, np.float32)))
        for side in range(2):
            io.hand_histories[side].append((stamp+.005, np.full(6,side+.1, np.float32), stamp+.006))
    io.body_sample = io.body_history[-1]
    io.hand_samples = [(*history[-1][:2], None) for history in io.hand_histories]
    return io


def test_collector_snapshot_pairs_to_ego_instead_of_latest_decision_feedback():
    io = paired_io()
    io.check_feedback(1.)  # Latest feedback may be 50ms old without killing control.
    observed = io.observation(1.)
    assert np.allclose(observed["state"]["values"][0][:29], .872)
    assert observed["snapshot_timing"] == pytest.approx(dict(camera_anchor_time=-.12, body_time=-.128,
                                                            hand_times=[-.123,-.123]))
    assert observed["state"]["times"] == pytest.approx([-.123])
    assert [observed["images"][name]["_source"] for name in CAMERAS] == [b"ego",b"left",b"right"]
    req = request()
    req.update(observed)
    req["images"] = {name: dict(time=image["time"], encoded="image") for name,image in observed["images"].items()}
    validate_request(req, state_count=1, rtc_max_prefix_steps=15, observation_contract=contract())
    with pytest.raises(ValueError, match="40ms"):
        validate_request(req, state_count=1)


@pytest.mark.parametrize("failure", ["camera_skew", "body_offset", "hand_offset", "future", "state_time", "missing_timing", "stale"])
def test_collector_wire_enforces_source_bound_pairing_and_real_component_times(failure):
    observed = paired_io().observation(1.)
    req = request()
    req.update(observed)
    req["images"] = {name: dict(time=image["time"], encoded="image") for name,image in observed["images"].items()}
    if failure == "camera_skew":
        req["images"][CAMERAS[-1]]["time"] = -.17
    elif failure == "body_offset":
        req["snapshot_timing"]["body_time"] = -.18
    elif failure == "hand_offset":
        req["snapshot_timing"]["hand_times"][0] = -.18
    elif failure == "future":
        req["snapshot_timing"]["body_time"] = .001
    elif failure == "state_time":
        req["state"]["times"][0] = -.01
    elif failure == "missing_timing":
        del req["snapshot_timing"]
    else:
        req["images"][CAMERAS[0]]["time"] = -.201
    with pytest.raises(ValueError):
        validate_request(req, state_count=1, rtc_max_prefix_steps=15, observation_contract=contract())


@pytest.mark.parametrize("failure", ["undeclared_clock", "robot_clock", "different_boot", "no_per_camera_time", "future_time"])
def test_collector_camera_adapter_never_guesses_a_robot_or_missing_receipt_clock(failure):
    io = bare_io()
    packet = bundle_packet()
    if failure == "undeclared_clock":
        del packet["timing"]["receipt_clock"]
    elif failure == "robot_clock":
        packet["timing"]["receipt_clock"] = "robot_ros_header_stamp"
    elif failure == "different_boot":
        packet["timing"].update(receipt_clock="workstation_monotonic", workstation_boot_id="different-host")
    elif failure == "no_per_camera_time":
        del packet["timing"]["cameras"]["video6"]["received_monotonic"]
    else:
        packet["timing"]["cameras"]["ego_view"]["received_monotonic"] = .95
    with pytest.raises(ValueError):
        io.decode_camera_bundle(packet, .94)


def test_collector_keeps_entire_bundles_and_masks_temporarily_unpairable_vision():
    io = paired_io()
    # The latest bundle's ego and wrist times are too far apart. The old valid
    # bundle remains usable; there is no cross-bundle camera substitution.
    io.camera_bundles.append((.99, {name:b"new" for name in CAMERAS},
                             dict(zip(CAMERAS,(.97,.92,.97))), "bad-bundle"))
    observed = io.observation(1.)
    assert observed["images"][CAMERAS[0]]["_source"] == b"ego"
    io.body_history.clear()
    io.body_history.append((.98,np.zeros(29, np.float32)))
    with pytest.raises(ValueError, match="paired body"):
        io.observation(1.)
    io.check_feedback(1.)  # A pairing miss alone does not mean feedback failed.


def test_collector_history_and_limits_are_checkpoint_bound():
    io = bare_io()
    assert io.body_history.maxlen == 128 and io.camera_bundles.maxlen == 128
    io.config.real_observation_mode = "collector_snapshot_v1"
    io.config.real_observation_contract = deepcopy(contract()["collector"])
    io.config.real_observation_contract["max_age_ms"] = 1000.
    with pytest.raises(ValueError, match="differ"):
        io.configure_observation_contract(contract())
    with pytest.raises(ValueError, match="matching collector checkpoint"):
        io.configure_observation_contract(None)


def test_skipped_absolute_slots_invalidate_inflight_prefix_and_keep_future_actions():
    queue = AbsoluteActionQueue(rtc_max_prefix_steps=15)
    queue.reset("test")
    req = request()
    old = np.full((30,76), .25, np.float32)
    queue.install(req,response(req,old),now=1.001)
    revision,prefix = queue.snapshot(1.,15)
    pending = request(revision=revision,prefix=prefix)
    reply = np.full((30,76), .5, np.float32)
    reply[:15] = prefix
    assert queue.skip_to(1.06) == 2
    assert queue.revision == revision+1
    with pytest.raises(ValueError, match="revision"):
        queue.install(pending,response(pending,reply),now=1.061)
    assert queue.pop(1.04) is None
    assert np.all(queue.pop(1.06) == .25)
    revision,prefix = queue.snapshot(1.06,15)
    fresh = request(1.06,revision,prefix)
    reply[:15] = prefix
    queue.install(fresh,response(fresh,reply),now=1.081)
    assert np.all(queue.pop(1.36) == .25)
    assert np.all(queue.pop(1.38) == .5)


class CollectorSensors(Sensors):
    def __init__(self, clock, *, vision=lambda now: True, supports_hold=False, hold_ack=True, **kwargs):
        super().__init__(clock, vision=vision, **kwargs)
        self.config = SonicIOConfig(state_history_count=1)
        self.body_history = deque(maxlen=128)
        self.hand_histories = [deque(maxlen=128),deque(maxlen=128)]
        self.camera_bundles = deque(maxlen=128)
        self.camera_bundle_error = None
        self.supports_hold = supports_hold
        self.hold_ack = hold_ack
        self.holds = []
        self.last_bundle_time = 0.
        self.missed_at = 1.14

    def poll(self, now):
        body = now-.035
        self.body_sample = (body, np.full(29,body,np.float32))
        self.body_history.append(self.body_sample)
        self.hand_samples = [(now-.047,np.zeros(6,np.float32),None)]*2
        for side in range(2):
            self.hand_histories[side].append((now-.047,np.zeros(6,np.float32),now))
        if self.vision(now) and now-self.last_bundle_time >= 1/30:
            times = dict(zip(CAMERAS,(now-.07,now-.058,now-.075)))
            self.camera_bundles.append((now-.002,{name:self.rgb for name in CAMERAS},times,now))
            self.last_bundle_time = now
        if self.feedback_failure and now >= 1.14:
            self.body_sample = (now-.201,self.body_sample[1])
        if self.missed_period and now >= self.missed_at:
            self.clock.now += self.missed_period
            self.missed_period = None

    def check_feedback(self, anchor):
        self.feedback_checks.append(anchor)
        return super().check_feedback(anchor)

    def hold(self):
        self.holds.append(self.clock.now)
        return self.hold_ack


class CollectorTransport(Transport):
    def __init__(self, *, projection=False):
        super().__init__()
        self.projection = projection
    def call(self, route, payload=None):
        result = super().call(route,payload)
        if route == "/health":
            result["observation_contract"] = contract(projection=self.projection)
        return result


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(real_sonic_client,"time",value)
    monkeypatch.setattr(real_sonic_client,"ThreadPoolExecutor",ImmediateWorker)
    return value


def test_collector_normal_asynchronous_state_ages_and_a_missed_period_continue(clock):
    io = CollectorSensors(clock, missed_period=.041)
    transport = CollectorTransport()
    real_sonic_client.run(io,transport,task="doll",rtc_prefix_steps=15,low_watermark=20,duration=.5)
    assert io.stopped and io.closed and len(io.commands)>10
    sent = [when for when,_ in io.commands]
    assert any(delta >= .04 for delta in np.diff(sent))
    slots = [int(np.floor(when/PERIOD+1e-9)) for when in sent]
    assert len(slots) == len(set(slots))  # At most one send per absolute slot, no catch-up burst.
    assert all(len(req["committed_prefix"]) in (0,15) for req in transport.requests)
    for req in transport.requests:
        observed = deepcopy(req)
        observed["images"] = {name:dict(time=image["time"],encoded="image") for name,image in req["images"].items()}
        validate_request(observed,state_count=1,rtc_max_prefix_steps=15,observation_contract=contract())


def test_collector_real_feedback_stream_loss_still_faults(clock):
    io = CollectorSensors(clock,feedback_failure="body")
    with pytest.raises(RuntimeError,match="feedback stream older than 200ms"):
        real_sonic_client.run(io,CollectorTransport(),task="doll",rtc_prefix_steps=15,low_watermark=20)
    assert io.commands[-1][0] < 1.14


def test_native_sonic_exhaustion_explains_missing_acknowledged_hold(clock):
    io = CollectorSensors(clock,vision=lambda now:now<1.08)
    with pytest.raises(RuntimeError,match="backend has no acknowledged hold capability"):
        real_sonic_client.run(io,CollectorTransport(),task="doll",rtc_prefix_steps=15,low_watermark=20)
    assert not io.holds


def test_supported_hold_waits_and_recovers_from_fresh_zero_prefix_inference(clock):
    io = CollectorSensors(clock,vision=lambda now:now<1.08 or now>=1.72,supports_hold=True)
    transport = CollectorTransport()
    real_sonic_client.run(io,transport,task="doll",rtc_prefix_steps=15,low_watermark=20,duration=1.0)
    assert io.holds and len(transport.requests)>=2
    recovered = [req for req in transport.requests if req["anchor_time"]>io.holds[0]]
    assert recovered[0]["committed_prefix"] == []
    assert any(when>1.76 for when,_ in io.commands)
    assert not any(io.holds[0] <= when <= io.holds[-1] for when,_ in io.commands)


@pytest.mark.parametrize("ack", [False,None])
def test_hold_capability_requires_an_explicit_success_ack(clock,ack):
    io = CollectorSensors(clock,vision=lambda now:now<1.08,supports_hold=True,hold_ack=ack)
    with pytest.raises(RuntimeError,match="hold was not acknowledged"):
        real_sonic_client.run(io,CollectorTransport(),task="doll",rtc_prefix_steps=15,low_watermark=20)


def test_supported_hold_has_a_fixed_nonrenewing_recovery_deadline(clock):
    io = CollectorSensors(clock,vision=lambda now:now<1.08,supports_hold=True)
    with pytest.raises(RuntimeError,match="grace expired"):
        real_sonic_client.run(io,CollectorTransport(),task="doll",rtc_prefix_steps=15,low_watermark=20,
                              starvation_grace_s=.1)
    assert io.holds and clock.now-io.holds[0] <= .12


def test_delayed_inflight_rtc_reply_is_revoked_after_missed_committed_slots(clock,monkeypatch,caplog):
    class DeferredResult:
        def __init__(self, function, req):
            self.value = function(req)
            self.ready_at = clock.now+(.04 if req["request_seq"] == 0 else .18)
        def done(self):
            return clock.now >= self.ready_at-1e-9
        def result(self):
            assert self.done()
            return self.value
    class DelayedWorker(ImmediateWorker):
        def submit(self, function, req):
            return DeferredResult(function,req)
    monkeypatch.setattr(real_sonic_client,"ThreadPoolExecutor",DelayedWorker)
    io = CollectorSensors(clock,missed_period=.041)
    io.missed_at = 1.3
    transport = CollectorTransport()
    real_sonic_client.run(io,transport,task="doll",rtc_prefix_steps=15,low_watermark=20,duration=.6)
    assert len(transport.requests) == 2
    assert len(transport.requests[1]["committed_prefix"]) == 15
    assert "Skipped 2 expired action slots" in caplog.text
    assert "stale session or queue revision" in caplog.text
    sent = [when for when,_ in io.commands]
    assert any(delta > .04 for delta in np.diff(sent))
    assert len({int(np.floor(when/PERIOD+1e-9)) for when in sent}) == len(sent)


def test_feedback_work_delay_skips_popped_action_and_revokes_its_rtc_reply(clock, monkeypatch, caplog):
    class DeferredResult:
        def __init__(self, function, req):
            self.value = function(req)
            self.ready_at = clock.now+(.04 if req["request_seq"] == 0 else .18)
        def done(self):
            return clock.now >= self.ready_at-1e-9
        def result(self):
            return self.value
    class DelayedWorker(ImmediateWorker):
        def submit(self, function, req):
            return DeferredResult(function, req)
    class DelayedFeedback(CollectorSensors):
        delayed = False
        def check_feedback(self, anchor):
            super().check_feedback(anchor)
            if not self.delayed and anchor >= 1.3-1e-9:
                self.delayed = True
                clock.now += .041

    monkeypatch.setattr(real_sonic_client, "ThreadPoolExecutor", DelayedWorker)
    io, transport = DelayedFeedback(clock), CollectorTransport()
    real_sonic_client.run(io, transport, task="doll", rtc_prefix_steps=15,
                          low_watermark=20, duration=.6)
    assert "Skipped 2 expired action slots" in caplog.text
    assert "stale session or queue revision" in caplog.text
    assert len(transport.requests[1]["committed_prefix"]) == 15
    # A command for 1.30 cannot be sent at 1.341 after feedback processing.
    anchor = transport.requests[0]["anchor_time"]
    for sent, action in io.commands:
        slot = np.floor(sent/PERIOD+1e-9)*PERIOD
        assert action[64] == pytest.approx(round((slot-anchor)/PERIOD)/32)
    slots = [int(np.floor(sent/PERIOD+1e-9)) for sent,_ in io.commands]
    assert len(slots) == len(set(slots))


def test_skipped_popped_commit_revokes_reply_even_without_other_expired_queue_slots():
    queue = AbsoluteActionQueue(rtc_max_prefix_steps=15)
    queue.reset("test")
    req = request()
    queue.install(req, response(req), now=1.001)
    revision, prefix = queue.snapshot(1., 15)
    pending = request(revision=revision, prefix=prefix)
    assert queue.pop(1.02) is not None
    assert queue.skip_to(1.04, missed_current=True) == 1
    with pytest.raises(ValueError, match="stale session or queue revision"):
        queue.install(pending, response(pending), now=1.041)
    assert queue.pop(1.04) is not None  # The next original slot remains usable.


def test_result_handling_delay_discards_expired_reply_and_recaptures_at_actual_time(clock, monkeypatch, caplog):
    class SlowResult:
        def __init__(self, function, req):
            self.value = function(req)
            self.delay = .64 if req["request_seq"] == 0 else 0.
        def done(self):
            return True
        def result(self):
            clock.now += self.delay
            return self.value
    class SlowWorker(ImmediateWorker):
        def submit(self, function, req):
            return SlowResult(function, req)
    class DistinctReplies(CollectorTransport):
        def infer(self, req):
            reply = super().infer(req)
            reply["actions"] = np.full((30,76), .25 if req["request_seq"] == 0 else .5).tolist()
            return reply

    monkeypatch.setattr(real_sonic_client, "ThreadPoolExecutor", SlowWorker)
    io, transport = CollectorSensors(clock), DistinctReplies()
    real_sonic_client.run(io, transport, task="doll", duration=.8)
    assert "Discarded reply after control-thread delay" in caplog.text
    assert transport.requests[1]["anchor_time"] >= 1.68-1e-9
    assert io.commands and all(np.all(action == .5) for _,action in io.commands)


@pytest.mark.parametrize("configuration", [dict(camera_bundle_format="unknown"),
    dict(camera_bundle_format="chenyz_composer_v1", camera_sources={"ego_view":"tcp://127.0.0.1:1"})])
def test_collector_camera_route_is_explicit_and_unambiguous(configuration):
    io = bare_io()
    for key,value in configuration.items():
        setattr(io.config, key, value)
    with pytest.raises(ValueError, match="camera_bundle_format"):
        io.configure_observation_contract(contract())


@pytest.mark.parametrize("profile", [None, build_recorded_action_projection(-.75,.6875,"a"*64)])
def test_action_projection_binds_queue_rtc_and_preserves_recorded_extrema(profile):
    queue = AbsoluteActionQueue(rtc_max_prefix_steps=15,action_projection=profile)
    queue.reset("test")
    req = request()
    raw = np.full((30,76),.25,np.float32)
    raw[:,:4] = [-.75,.6875,-1.,1.]
    queue.install(req,response(req,raw),now=1.001)
    revision,prefix = queue.snapshot(1.,15)
    expected = [-.625,.625,-.625,.625] if profile is None else [-.75,.6875,-.75,.6875]
    assert prefix[0][:4] == expected
    followup = request(revision=revision,prefix=prefix)
    raw[:15] = prefix
    changed = raw.copy()
    changed[0,0] -= .01  # The same clamped boundary cannot hide a raw prefix edit.
    with pytest.raises(ValueError,match="changed committed actions"):
        queue.install(followup,response(followup,changed),now=1.002)
    queue.install(followup,response(followup,raw),now=1.002)
    assert queue.pop(1.02)[:4].tolist() == expected


def test_io_and_custom_adapter_projection_must_match_source_bound_health(clock):
    io = bare_io()
    bound = contract(projection=True)
    io.configure_observation_contract(bound)
    assert io.action_projection == bound["action_projection"]
    io.config.real_action_projection = build_recorded_action_projection(-.625,.625,"a"*64)
    with pytest.raises(ValueError,match="projection differs"):
        io.configure_observation_contract(bound)
    io = CollectorSensors(clock)
    def ignore_profile(advertised):
        io.observation_contract = advertised
        io.action_projection = None
    io.configure_observation_contract = ignore_profile
    with pytest.raises(ValueError,match="adapter action projection differs"):
        real_sonic_client.run(io,CollectorTransport(projection=True),task="doll",rtc_prefix_steps=15,
                              low_watermark=20,duration=.1)


@pytest.mark.parametrize("change", ["missing_identity", "offgrid_bound", "wrong_hand_space"])
def test_queue_rejects_a_tampered_projection_profile(change):
    profile = build_recorded_action_projection(-.75,.6875,"a"*64)
    if change == "missing_identity":
        del profile["source_sha256"]
    elif change == "offgrid_bound":
        profile["motion_token_max"] = .7
    else:
        profile["hand_max"] = 1000.
    with pytest.raises(ValueError):
        AbsoluteActionQueue(action_projection=profile)


def test_live_collector_client_uses_the_advertised_recorded_action_projection(clock):
    class RecordedActions(CollectorTransport):
        def infer(self, req):
            result = super().infer(req)
            actions = np.asarray(result["actions"],np.float32)
            actions[len(req["committed_prefix"]):,:2] = [-.75,.6875]
            result["actions"] = actions.tolist()
            return result
    io = CollectorSensors(clock)
    transport = RecordedActions(projection=True)
    real_sonic_client.run(io,transport,task="doll",rtc_prefix_steps=15,low_watermark=20,duration=.4)
    assert io.commands and all(action[:2].tolist() == [-.75,.6875] for _,action in io.commands)
    assert io.action_projection == contract(projection=True)["action_projection"]
    prefixes = [req["committed_prefix"] for req in transport.requests if req["committed_prefix"]]
    assert prefixes and prefixes[0][0][:2] == [-.75,.6875]


def test_recorded_projection_wire_rejects_unprojected_prefix_without_rewriting_it():
    req = request()
    req.update(paired_io().observation(1.))
    req["images"] = {name:dict(time=image["time"],encoded="image") for name,image in req["images"].items()}
    prefix = np.full((15,76),.25,np.float32)
    prefix[:,:2] = [-.75,.6875]
    req["committed_prefix"] = prefix.tolist()
    validate_request(req,state_count=1,rtc_max_prefix_steps=15,observation_contract=contract(projection=True))
    req["committed_prefix"][0][0] = -.8
    with pytest.raises(ValueError,match="projected controller commands"):
        validate_request(req,state_count=1,rtc_max_prefix_steps=15,observation_contract=contract(projection=True))
    assert req["committed_prefix"][0][0] == -.8
