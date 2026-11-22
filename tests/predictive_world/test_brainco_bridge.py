"""Native DDS boundary tests; no robot, serial SDK or DDS participant is started."""
import asyncio
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from psi.deploy.brainco_bridge import (
    ExistingDdsBridge, MeasuredHoldAcknowledgement, _NativeDdsTransport,
    serve_existing_dds, write_measured_hold,
)


def feedback(q):
    return SimpleNamespace(states=[SimpleNamespace(q=value) for value in q])


def command(q):
    return dict(format="zeta0.brainco12.command.v1", positions=q)


def test_measured_order_ages_sequence_and_native_normalized_units():
    bridge = ExistingDdsBridge()
    left = np.array([0., .1, .2, .3, .4, .5])
    right = np.array([1., .9, .8, .7, .6, .5])
    bridge.feedback(0, feedback(left), 10.)
    assert bridge.state(10.01) is None
    bridge.feedback(1, feedback(right), 10.02)
    state = bridge.state(10.05)
    assert state["positions"] == pytest.approx(np.concatenate([left, right]))
    assert state["age_s"] == pytest.approx([.05, .03])
    assert state["sequence"] == [1, 1]
    assert state["hold_capability"] is None
    assert state["hold_ack"]["acknowledged"] == [False, False]
    bridge.command(command([1.]*12), 10.05)
    bridge.step(10.05, lambda *args: pytest.fail("Preview must not publish DDS commands"))
    assert bridge.state(10.06)["positions"] == pytest.approx(np.concatenate([left, right]))


@pytest.mark.parametrize("q", [[1000.]*6, [-.01]*6, [.5]*5, [float("nan")]*6])
def test_invalid_dds_values_do_not_replace_measured_feedback(q):
    bridge = ExistingDdsBridge()
    bridge.feedback(0, feedback([.1]*6), 1.)
    with pytest.raises(ValueError):
        bridge.feedback(0, feedback(q), 1.1)
    assert bridge.samples[0][0] == 1.
    assert bridge.sequence[0] == 1


def test_execute_starts_at_feedback_slews_and_expiry_requests_real_position_once():
    bridge = ExistingDdsBridge(execute=True)
    writes = []
    write = lambda side, q, dq: writes.append((side, q.copy(), dq.copy()))
    for side in (0, 1):
        bridge.feedback(side, feedback([.2]*6), 1.)
    bridge.command(command([1.]*12), 1.)
    bridge.step(1., write)
    assert all(np.allclose(q, .2) and np.allclose(dq, .8) for _, q, dq in writes)
    bridge.step(1.02, write)
    assert all(np.allclose(q, .26) for _, q, _ in writes[-2:])
    for side in (0, 1):
        bridge.feedback(side, feedback([.23]*6), 1.11)
    bridge.step(1.11, write)
    assert all(np.allclose(q, .23) for _, q, _ in writes[-2:])
    count = len(writes)
    bridge.step(1.12, write)
    assert len(writes) == count


def test_feedback_gap_resets_slew_and_hold_never_claims_native_ack():
    bridge = ExistingDdsBridge(execute=True)
    writes = []
    write = lambda side, q, dq: writes.append((side, q.copy()))
    bridge.feedback(0, feedback([.1]*6), 1.)
    bridge.command(command([1.]*12), 1.)
    bridge.step(1., write)
    bridge.command(command([1.]*12), 1.3)
    bridge.step(1.3, write)
    assert len(writes) == 1  # No measured-position request from stale telemetry.
    bridge.feedback(0, feedback([.7]*6), 1.31)
    bridge.step(1.31, write)
    assert np.allclose(writes[-1][1], .7)  # Reconnect starts at actual pose.
    bridge.command(dict(format="zeta0.brainco12.command.v1", hold=True, hold_request_id="h1"), 1.32)
    bridge.step(1.32, write)
    assert bridge.hold_ack.state() == dict(request_id="h1", acknowledged=[False, False])


def test_rejected_command_does_not_partially_change_target():
    bridge = ExistingDdsBridge()
    bridge.command(command([.4]*12), 1.)
    for q, stamp in (([.5]*12, float("nan")), ([1.1]*12, 1.1)):
        with pytest.raises(ValueError):
            bridge.command(command(q), stamp)
        assert np.allclose(bridge.target, .4)
        assert bridge.target_time == 1.


def test_sdk_write_ack_requires_success_and_does_not_claim_pose_reached():
    ack = MeasuredHoldAcknowledgement()
    ack.request("h")
    calls = []
    async def write(slave, q, dq):
        calls.append((slave, q, dq))
        return False
    device = SimpleNamespace(ctx=SimpleNamespace(set_finger_positions_and_speeds=write), slave_id=126)
    with pytest.raises(RuntimeError, match="rejected"):
        asyncio.run(write_measured_hold(device, np.full(6, .2), ack, 0, "h"))
    assert ack.state()["acknowledged"] == [False, False]
    async def success(*args):
        await write(*args)
        return True
    device.ctx.set_finger_positions_and_speeds = success
    asyncio.run(write_measured_hold(device, np.full(6, .2), ack, 0, "h"))
    assert ack.state()["acknowledged"] == [True, False]
    assert calls[0] == (126, [200]*6, [800]*6)


def test_native_transport_uses_actual_topics_idl_units_and_lazy_imports(monkeypatch):
    events = []
    def participant(domain):
        events.append(("participant", domain))
        return "participant"
    def topic(participant, name, dtype):
        events.append(("topic", name, dtype))
        return name
    class Writer:
        def __init__(self, participant, topic, qos):
            self.topic = topic
        def write(self, message):
            events.append(("write", self.topic, message))
    class Reader:
        def __init__(self, *args, **kwargs):
            pass
        def take(self, count):
            return []
    modules = {
        "cyclonedds.domain": dict(Domain=lambda *args: object(), DomainParticipant=participant),
        "cyclonedds.topic": dict(Topic=topic), "cyclonedds.pub": dict(DataWriter=Writer),
        "cyclonedds.sub": dict(DataReader=Reader),
        "cyclonedds.core": dict(Qos=lambda *args: args, Policy=SimpleNamespace(
            Reliability=SimpleNamespace(BestEffort="best_effort"), History=SimpleNamespace(KeepLast=lambda n:n))),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": dict(
            MotorCmd_=lambda *args: args, MotorCmds_=lambda motors: motors, MotorStates_=object),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    transport = _NativeDdsTransport({}, True)
    transport.write(1, np.arange(6)/5, np.full(6, .8))
    event = events[-1]
    assert event[1] == "rt/brainco/right/cmd"
    assert [motor[1] for motor in event[2]] == pytest.approx(np.arange(6)/5)
    assert all(motor[0] == 0 and motor[2] == .8 and motor[3:] == (0,0,0,[0,0,0]) for motor in event[2])
    assert [event[1] for event in events if event[0] == "topic"] == [
        "rt/brainco/left/state", "rt/brainco/left/cmd", "rt/brainco/right/state", "rt/brainco/right/cmd"]
    transport.close()
    assert transport.readers == transport.writers == transport.entities == []


def test_dds_disposal_notifications_and_cleanup_failure_do_not_leak_zmq_resources(monkeypatch):
    import psi.deploy.brainco_bridge as module
    event = asyncio.Event()
    class Transport:
        def read(self, side):
            if side == 1:
                event.set()
            return [object(), feedback([.2]*6)]
        def write(self, *args):
            pytest.fail("Read-only bridge must not write")
        def close(self):
            raise RuntimeError("cleanup failure")
    import zmq
    context = zmq.Context()
    monkeypatch.setattr(module.zmq, "Context", lambda: context)
    asyncio.run(serve_existing_dds(dict(state_bind="inproc://state", command_bind="inproc://cmd"),
                                   transport=Transport(), stop_event=event))
    assert context.closed
