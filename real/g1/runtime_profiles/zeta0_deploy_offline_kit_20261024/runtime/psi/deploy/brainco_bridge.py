"""BrainCo Revo2 bridge using the existing native DDS service or Modbus SDK.

The two SDK devices retain their native six-coordinate order. Feedback always
comes from motor feedback, never from the requested target. No tactile/current
mode is used. DDS uses the already running hand service; direct Modbus requires
exclusive ownership of the serial ports. DDS publication is not an SDK ACK.
"""
import argparse
import asyncio
import importlib
import json
import logging
from pathlib import Path
import sys
import time
import uuid

import msgpack
import numpy as np
import zmq

from psi.deploy.real_sonic_protocol import finite_array

LOG = logging.getLogger(__name__)
HOLD_CAPABILITY = "oraclevla.brainco12.measured_hold.v1"


class MeasuredHoldAcknowledgement:
    """A hold is acknowledged only after both SDK writes complete successfully."""

    def __init__(self):
        self.request_id = None
        self.completed = [None, None]

    def request(self, request_id):
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("BrainCo hold requires a nonempty request_id")
        if request_id != self.request_id:
            self.request_id = request_id
            self.completed = [None, None]

    def resume(self):
        self.request_id = None
        self.completed = [None, None]

    def needed(self, side):
        return self.request_id is not None and self.completed[side] != self.request_id

    def acknowledge(self, side, request_id):
        if request_id == self.request_id:
            self.completed[side] = request_id

    def state(self):
        return dict(request_id=self.request_id,
                    acknowledged=[self.request_id is not None and done == self.request_id for done in self.completed])


async def write_measured_hold(device, positions, acknowledgement, side, request_id):
    """Record SDK write completion, not physical arrival at the requested pose."""
    result = await asyncio.wait_for(device.ctx.set_finger_positions_and_speeds(device.slave_id,
        np.rint(positions*1000).astype(int).tolist(), [800]*6), timeout=.1)
    if result is False:
        raise RuntimeError("BrainCo SDK rejected the measured-position write")
    acknowledgement.acknowledge(side, request_id)


def normalized_motor_positions(status):
    values = finite_array(list(status.positions), (6,), "BrainCo SDK measured motor positions")
    if (values < 0).any() or (values > 1000).any():
        raise ValueError("SDK feedback is outside normalized per-thousand 0..1000")
    return values / 1000.


def limited_target(target, previous, elapsed, rate=3.):
    target = finite_array(target, (6,), "BrainCo native hand target")
    previous = finite_array(previous, (6,), "BrainCo preceding hand target")
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Hand rate limiter received invalid elapsed time")
    if not np.isfinite(rate) or rate <= 0 or (previous < 0).any() or (previous > 1).any():
        raise ValueError("Hand rate limiter requires a positive rate and normalized previous position")
    bound = rate * min(elapsed, .1)
    return np.clip(previous + np.clip(np.clip(target, 0., 1.)-previous, -bound, bound), 0., 1.)


class ExistingDdsBridge:
    """Pure feedback/control boundary for the supplied six-coordinate service.

    Order is Thumb, Thumb_aux, Index, Middle, Ring, Pinky for each hand, left
    then right. Native DDS q and dq are normalized 0..1, not raw SDK 0..1000.
    Readiness/age uses actual DDS receives. No publisher success claims a
    hardware-consumption or measured-position-hold acknowledgement.
    """

    def __init__(self, *, execute=False):
        self.execute = execute
        self.session = uuid.uuid4().hex
        self.samples = [None, None]
        self.sequence = [0, 0]
        self.previous = [None, None]
        self.previous_time = [None, None]
        self.writing = [False, False]
        self.target = None
        self.target_time = None
        self.hold_ack = MeasuredHoldAcknowledgement()

    def feedback(self, side, message, received):
        stamp = float(received)
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError("DDS feedback receipt must be finite monotonic time")
        q = finite_array([motor.q for motor in message.states], (6,), "BrainCo DDS measured q")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("BrainCo DDS feedback q must be normalized 0..1")
        if self.samples[side] is not None and stamp < self.samples[side][0]:
            raise ValueError("BrainCo DDS receive clock regressed")
        self.samples[side] = stamp, q.copy()
        self.sequence[side] += 1

    def command(self, packet, received):
        if not isinstance(packet, dict) or packet.get("format") != "oraclevla.brainco12.command.v1":
            raise ValueError("Unsupported hand command format")
        if packet.get("hold"):
            self.hold_ack.request(packet.get("hold_request_id"))
            self.target = None
        elif packet.get("stop"):
            self.target = None
            self.hold_ack.resume()
        else:
            target = finite_array(packet["positions"], (12,), "BrainCo DDS command").copy()
            stamp = float(received)
            if (target < 0).any() or (target > 1).any():
                raise ValueError("BrainCo DDS command q must be normalized 0..1")
            if not np.isfinite(stamp) or stamp < 0:
                raise ValueError("BrainCo DDS command receipt must be finite monotonic time")
            self.target, self.target_time = target, stamp
            self.hold_ack.resume()

    def step(self, now, write):
        if not self.execute:
            return
        now = float(now)
        if not np.isfinite(now) or now < 0:
            raise ValueError("BrainCo DDS decision must be finite monotonic time")
        for side, sample in enumerate(self.samples):
            if sample is None or not 0 <= now-sample[0] <= .2:
                self.writing[side] = False
                self.previous[side] = self.previous_time[side] = None
                continue  # Never synthesize a measured hold from stale feedback.
            _, q = sample
            if self.target is not None and 0 <= now-self.target_time <= .1:
                if self.previous[side] is None or not self.writing[side]:
                    value = q.copy()  # First active target starts at measured position.
                else:
                    value = limited_target(self.target[side*6:side*6+6], self.previous[side],
                                           now-self.previous_time[side])
                write(side, value, np.full(6, .8))
                self.previous[side], self.previous_time[side], self.writing[side] = value, now, True
            elif self.writing[side]:
                write(side, q, np.full(6, .8))
                self.previous[side], self.previous_time[side], self.writing[side] = q.copy(), now, False

    def state(self, now):
        if any(sample is None for sample in self.samples):
            return None
        return dict(format="oraclevla.brainco12.state.v1", session_id=self.session,
                    sequence=list(self.sequence), positions=np.concatenate([sample[1] for sample in self.samples]).tolist(),
                    age_s=[now-sample[0] for sample in self.samples], transport="existing_dds",
                    hold_capability=None, hold_ack=self.hold_ack.state())


class _NativeDdsTransport:
    """Lazy imports: importing this module never constructs DDS participants."""

    def __init__(self, config, execute):
        from cyclonedds.domain import Domain, DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.core import Qos, Policy
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_

        self.readers, self.writers, self.entities = [], [], []
        self.message_type, self.motor_type = MotorCmds_, MotorCmd_
        try:
            domain_id = int(config.get("dds_domain_id", 0))
            if config.get("dds_config"):
                self.entities.append(Domain(domain_id, config["dds_config"]))
            participant = DomainParticipant(domain_id)
            self.entities.append(participant)
            qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
            for side in ("left", "right"):
                topic = Topic(participant, f"rt/brainco/{side}/state", MotorStates_)
                self.entities.append(topic)
                self.readers.append(DataReader(participant, topic, qos=qos))
                if execute:
                    topic = Topic(participant, f"rt/brainco/{side}/cmd", MotorCmds_)
                    self.entities.append(topic)
                    self.writers.append(DataWriter(participant, topic, qos=qos))
        except Exception:
            self.close()
            raise

    def read(self, side):
        return self.readers[side].take(20)

    def write(self, side, positions, speeds):
        # Matches the supplied collection bridge's native Unitree IDL constructor.
        self.writers[side].write(self.message_type([
            self.motor_type(0, float(q), float(dq), 0, 0, 0, [0, 0, 0])
            for q, dq in zip(positions, speeds, strict=True)]))

    def close(self):
        self.writers.clear()
        self.readers.clear()
        # CycloneDDS entities own their C resources and release them on deletion.
        self.entities.clear()


async def serve_existing_dds(config, *, execute=False, transport=None, stop_event=None):
    bridge = ExistingDdsBridge(execute=execute)
    context = zmq.Context()
    pub, sub = context.socket(zmq.PUB), context.socket(zmq.SUB)
    try:
        pub.setsockopt(zmq.SNDHWM, 2)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        pub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt(zmq.LINGER, 0)
        pub.bind(config.get("state_bind", "tcp://*:5561"))
        sub.bind(config.get("command_bind", "tcp://*:5562"))
        if transport is None:
            transport = _NativeDdsTransport(config, execute)
        LOG.info("BrainCo existing DDS bridge ready; execute=%s; SDK hold ACK unavailable", execute)
        while stop_event is None or not stop_event.is_set():
            for side in (0, 1):
                for message in transport.read(side):
                    # DDS disposal notifications contain no telemetry. Invalid
                    # samples never refresh the preceding real feedback age.
                    if not hasattr(message, "states"):
                        continue
                    try:
                        bridge.feedback(side, message, time.monotonic())
                    except (AttributeError, TypeError, ValueError):
                        LOG.warning("Ignoring invalid BrainCo DDS feedback for hand %s", side)
            try:
                raw = sub.recv(zmq.NOBLOCK)
            except zmq.Again:
                raw = None
            if raw is not None:
                bridge.command(msgpack.unpackb(raw, raw=False), time.monotonic())
            now = time.monotonic()
            bridge.step(now, transport.write)
            packet = bridge.state(time.monotonic())
            if packet is not None:
                pub.send(msgpack.packb(packet, use_bin_type=True))
            await asyncio.sleep(.02)
    finally:
        if transport is not None:
            if execute:
                bridge.target = None
                try:
                    bridge.step(time.monotonic(), transport.write)
                except Exception:
                    LOG.exception("BrainCo final DDS measured-position request failed")
            try:
                transport.close()
            except Exception:
                LOG.exception("BrainCo DDS transport cleanup failed")
        pub.close(linger=0)
        sub.close(linger=0)
        context.term()


async def serve(args):
    with open(args.config) as stream:
        config = json.load(stream)
    transport = config.get("transport", "sdk_modbus")
    if transport == "existing_dds":
        return await serve_existing_dds(config, execute=args.execute)
    if transport != "sdk_modbus":
        raise ValueError("BrainCo transport must be existing_dds or sdk_modbus")
    return await _serve_sdk(args)


async def _serve_sdk(args):
    if not args.sdk_python:
        raise ValueError("sdk_modbus requires --sdk-python; existing_dds reuses the running native service")
    sdk_path = Path(args.sdk_python).resolve()
    if not (sdk_path/"common_init.py").is_file():
        raise ValueError("--sdk-python must point to the installed official BrainCo SDK python directory")
    sys.path.insert(0, str(sdk_path))
    helpers = importlib.import_module("common_init")
    sdk = importlib.import_module("common_imports").sdk
    with open(args.config) as stream:
        config = json.load(stream)
    devices = []
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    sub = context.socket(zmq.SUB)
    pub.setsockopt(zmq.SNDHWM, 2)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    pub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.LINGER, 0)
    session = uuid.uuid4().hex
    measured = [None, None]
    seq = [0, 0]
    target = None
    target_time = 0.
    workers = []
    writing_flags = [False, False]
    fault = asyncio.Event()
    error = []
    hold_ack = MeasuredHoldAcknowledgement()

    async def hand_worker(side, device):
        nonlocal target, target_time
        previous, previous_time = None, time.monotonic()
        writing = False
        try:
            while not fault.is_set():
                begun = time.monotonic()
                status = await asyncio.wait_for(device.ctx.get_motor_status(device.slave_id), timeout=.1)
                q = normalized_motor_positions(status)
                measured[side] = (time.monotonic(), q.copy())
                seq[side] += 1
                if previous is None:
                    previous = q.copy()
                now = time.monotonic()
                if args.execute and hold_ack.needed(side):
                    request_id = hold_ack.request_id
                    await write_measured_hold(device, q, hold_ack, side, request_id)
                    previous = q.copy()
                    writing = False
                    writing_flags[side] = False
                elif args.execute and hold_ack.request_id is not None:
                    previous = q.copy()
                elif args.execute and target is not None and now-target_time <= .1:
                    command = limited_target(target[side*6:side*6+6], previous, now-previous_time)
                    await asyncio.wait_for(device.ctx.set_finger_positions_and_speeds(device.slave_id,
                        np.rint(command*1000).astype(int).tolist(), [800]*6), timeout=.1)
                    previous = command
                    writing = True
                    writing_flags[side] = True
                elif args.execute and writing:
                    # On stop/expiry request the freshly measured position, so
                    # an old closing target is not left active in the SDK.
                    await asyncio.wait_for(device.ctx.set_finger_positions_and_speeds(device.slave_id,
                        np.rint(q*1000).astype(int).tolist(), [800]*6), timeout=.1)
                    previous = q.copy()
                    writing = False
                    writing_flags[side] = False
                elif not writing:
                    previous = q.copy()
                previous_time = now
                await asyncio.sleep(max(0., .02-(time.monotonic()-begun)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error.append(exc)
            fault.set()

    try:
        pub.bind(config.get("state_bind", "tcp://*:5561"))
        sub.bind(config.get("command_bind", "tcp://*:5562"))
        left_config, right_config = config["left"], config["right"]
        if left_config["port"] == right_config["port"]:
            raise ValueError("This bridge requires one serial transport per hand; use the existing adapter for a shared bus")
        for side, spec in enumerate((left_config, right_config)):
            slave = spec["slave_id"]
            slave = int(slave, 0) if isinstance(slave, str) else int(slave)
            device = await helpers.init_modbus(spec["port"], int(spec["baudrate"]), slave)
            if device is None:
                raise RuntimeError(f"Failed to initialize {'left' if side == 0 else 'right'} hand")
            devices.append(device)
            if args.execute:
                await device.ctx.set_finger_unit_mode(device.slave_id, sdk.FingerUnitMode.Normalized)
            if await device.ctx.get_finger_unit_mode(device.slave_id) != sdk.FingerUnitMode.Normalized:
                raise ValueError("BrainCo unit mode must already be Normalized for read-only use")
            workers.append(asyncio.create_task(hand_worker(side, device)))
        LOG.info("BrainCo bridge ready; execute=%s native normalized order retained", args.execute)
        while not fault.is_set():
            try:
                raw = sub.recv(zmq.NOBLOCK)
            except zmq.Again:
                raw = None
            if raw is not None:
                data = msgpack.unpackb(raw, raw=False)
                if data.get("format") != "oraclevla.brainco12.command.v1":
                    raise ValueError("Unsupported hand command format")
                if data.get("hold"):
                    hold_ack.request(data.get("hold_request_id"))
                    target = None
                elif data.get("stop"):
                    target = None
                    hold_ack.resume()
                else:
                    target = finite_array(data["positions"], (12,), "hand command").copy()
                    target_time = time.monotonic()
                    hold_ack.resume()
            if all(value is not None for value in measured):
                now = time.monotonic()
                pub.send(msgpack.packb(dict(format="oraclevla.brainco12.state.v1", session_id=session,
                    sequence=list(seq), positions=np.concatenate([v[1] for v in measured]).tolist(),
                    age_s=[now-v[0] for v in measured],
                    hold_capability=HOLD_CAPABILITY if args.execute else None,
                    hold_ack=hold_ack.state()), use_bin_type=True))
            await asyncio.sleep(.005)
        if error:
            raise RuntimeError("BrainCo SDK feedback/control interrupted") from error[0]
    finally:
        target = None
        # Allow successful workers one feedback cycle to stop an active target.
        await asyncio.sleep(.03)
        fault.set()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        if args.execute:
            for side, device in enumerate(devices):
                if writing_flags[side]:
                    try:
                        status = await asyncio.wait_for(device.ctx.get_motor_status(device.slave_id), timeout=.1)
                        q = normalized_motor_positions(status)
                        await asyncio.wait_for(device.ctx.set_finger_positions_and_speeds(device.slave_id,
                            np.rint(q*1000).astype(int).tolist(), [800]*6), timeout=.1)
                    except Exception:
                        LOG.exception("Hand %s did not acknowledge final measured-position hold", side)
        for device in devices:
            try:
                await helpers.cleanup_context(device)
            except Exception:
                LOG.exception("BrainCo SDK context cleanup failed")
        pub.close(linger=0)
        sub.close(linger=0)
        context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-python", help="Official SDK python directory; required only for sdk_modbus")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        LOG.info("Bridge stopped")


if __name__ == "__main__":
    main()
