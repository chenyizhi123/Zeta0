"""Existing SONIC ZMQ camera/body interfaces plus a native BrainCo12 bridge.

No 7-joint hand conversion is performed. ZMQ sockets belong to the client control
thread. Inference never blocks this thread. Imports do not open hardware.
"""
import base64
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import zmq

from psi.deploy.real_sonic_protocol import CAMERAS, MeasuredHistory, finite_array
from psi.deploy.sonic_wire import build_command_message, pack_pose_message
from psi.real_observation_contract import (
    validate_action_projection,
    validate_deployment_observation_contract,
    validate_observation_contract,
    validate_snapshot_timing,
)
from psi.sonic_actions import controller_command_values


@dataclass
class SonicIOConfig:
    """Same-host connection defaults; pass remote endpoints in the robot config."""

    body_endpoint: str = "tcp://127.0.0.1:5557"
    camera_endpoint: str = "tcp://127.0.0.1:5555"
    hand_state_endpoint: str = "tcp://127.0.0.1:5561"
    hand_command_endpoint: str = "tcp://127.0.0.1:5562"
    action_bind: str = "tcp://*:5556"
    execute: bool = False
    state_history_count: int = 21
    real_observation_mode: str | None = None
    real_observation_contract: dict | None = None
    real_action_projection: dict | None = None
    camera_sources: dict | None = None
    camera_bundle_format: str | None = None
    velocity_fallback: bool = False


def decode_camera_image(encoded):
    # Preserve the existing SONIC ImageMessageSchema semantics: string JPEGs
    # are legacy arrays decoded without channel swapping; binary JPEGs are RGB.
    if isinstance(encoded, str):
        image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(encoded, (bytes, bytearray)):
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            image = image[..., ::-1]
    else:
        image = np.asarray(encoded)
    if image is None or image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("Camera must supply recorded RGB uint8 [480,640,3]")
    return image


class SonicZmqIO:
    # Token retention in SONIC is not an acknowledged position/balance hold.
    supports_hold = False

    def __init__(self, config):
        if type(config.velocity_fallback) is not bool:
            raise ValueError("velocity_fallback must be an explicit boolean")
        if config.velocity_fallback and not config.execute:
            raise ValueError("Velocity fallback control requires --execute and an ACK-capable controller")
        self.config = config
        self.context = zmq.Context()
        self.sockets = []
        self.body = self.subscribe(config.body_endpoint, b"g1_debug")
        self.camera = None if config.camera_sources is not None else self.subscribe(config.camera_endpoint, b"")
        self.camera_receiver = None
        self.hands = self.subscribe(config.hand_state_endpoint, b"")
        self.pose = self.hand_commands = None
        if config.execute:
            self.pose = self.context.socket(zmq.PUB)
            self.pose.setsockopt(zmq.LINGER, 0)
            self.pose.setsockopt(zmq.SNDHWM, 2)
            self.pose.bind(config.action_bind)
            self.sockets.append(self.pose)
            self.hand_commands = self.context.socket(zmq.PUB)
            self.hand_commands.setsockopt(zmq.LINGER, 0)
            self.hand_commands.setsockopt(zmq.SNDHWM, 2)
            self.hand_commands.connect(config.hand_command_endpoint)
            self.sockets.append(self.hand_commands)
        self.body_sample = None
        self.hand_samples = [None, None]
        self.images = {}
        self.camera_identities = {}
        self.history = MeasuredHistory(state_count=config.state_history_count)
        self.frame_index = 0
        self.active = False
        self.controller_status = None
        self.stream_selected_at = None
        self.stream_mode_acknowledged = False
        self.error = None
        self.observation_contract = None
        self.action_projection = None
        self.body_history = deque(maxlen=128)
        self.hand_histories = [deque(maxlen=128), deque(maxlen=128)]
        self.camera_bundles = deque(maxlen=128)
        self.camera_bundle_error = None
        self.velocity_handshake = None
        self.asynchronous_hold = False
        self.hand_feedback = None
        self.hand_received = None
        self.control_mode = None
        self.hold_request_id = None
        self.hold_started = None
        self.hold_session = None
        self.hold_sequence = None
        if config.velocity_fallback:
            from psi.deploy.sonic_velocity_fallback import VelocityFallbackHandshake

            self.velocity_handshake = VelocityFallbackHandshake()
            self.asynchronous_hold = True

    def configure_observation_contract(self, contract):
        """Bind the optional collector route to the checkpoint's advertised inputs."""
        config = getattr(self, "config", None)
        mode = getattr(config, "real_observation_mode", None)
        limits = getattr(config, "real_observation_contract", None)
        projection = getattr(config, "real_action_projection", None)
        sources = getattr(config, "camera_sources", None)
        bundle_format = getattr(config, "camera_bundle_format", None)
        if bundle_format is not None:
            from psi.deploy.real_sonic_camera import COMPOSER_FORMAT

            if bundle_format != COMPOSER_FORMAT:
                raise ValueError("Unsupported camera_bundle_format")
            if sources is not None:
                raise ValueError("Choose independent camera_sources or camera_bundle_format")
        if contract is None:
            if mode is not None or limits is not None or projection is not None or sources is not None or bundle_format is not None or getattr(config, "velocity_fallback", False):
                raise ValueError("Collector IO configuration requires a matching collector checkpoint")
            self.observation_contract = None
            self.action_projection = None
            return
        bound = validate_deployment_observation_contract(contract)
        if self.history.state_count != bound["state_history_count"]:
            raise ValueError("Collector IO requires --current-state-only")
        if mode is not None and mode != bound["mode"]:
            raise ValueError("Configured observation mode differs from checkpoint")
        if limits is not None and validate_observation_contract(limits) != bound["collector"]:
            raise ValueError("Configured collector limits differ from checkpoint")
        if projection is not None and validate_action_projection(projection) != bound.get("action_projection"):
            raise ValueError("Configured action projection differs from checkpoint")
        self.observation_contract = bound
        self.action_projection = bound.get("action_projection")
        size = bound["collector"]["history_size"]
        self.body_history = deque(getattr(self, "body_history", ()), maxlen=size)
        self.hand_histories = [deque(history, maxlen=size) for history in getattr(self, "hand_histories", ((),()))]
        self.camera_bundles = deque(getattr(self, "camera_bundles", ()), maxlen=size)
        self.camera_bundle_error = None
        if sources is not None:
            from psi.deploy.real_sonic_camera import IndependentCameraReceiver

            if getattr(self, "camera_receiver", None) is not None:
                self.camera_receiver.close()
            self.camera_receiver = IndependentCameraReceiver(sources, bound["collector"], context=self.context)

    def decode_camera_bundle(self, packet, received):
        """Adapter boundary for an intact bundle with genuine client-clock receipts.

        A collector should publish ``timing.cameras.<name>.received_monotonic``
        and a declared receipt clock. Robot wall/ROS/capture timestamps are never
        interpreted as client monotonic times. Custom acquisition adapters may
        override this method after verifying their actual producer schema.
        """
        if getattr(getattr(self, "config", None), "camera_bundle_format", None) is not None:
            from psi.deploy.real_sonic_camera import normalize_collector_camera_bundle

            packet = normalize_collector_camera_bundle(packet, received,
                self.observation_contract["collector"], endpoint=self.config.camera_endpoint)
        if not isinstance(packet, Mapping):
            raise ValueError("Collector camera bundle must be a mapping")
        timing = packet.get("timing", packet)
        if not isinstance(timing, Mapping):
            raise ValueError("Collector camera timing must be a mapping")
        clock = timing.get("receipt_clock", timing.get("decision_clock"))
        if clock == "workstation_monotonic":
            boot_path = Path("/proc/sys/kernel/random/boot_id")
            if not boot_path.exists() or timing.get("workstation_boot_id") != boot_path.read_text().strip():
                raise ValueError("Collector camera workstation clock must match this client's boot_id")
        elif clock != "client_monotonic":
            raise ValueError("Collector camera bundle needs declared client_monotonic receipt timing; configure its acquisition adapter")
        images, cameras = packet.get("images"), timing.get("cameras")
        if not isinstance(images, Mapping) or not isinstance(cameras, Mapping):
            raise ValueError("Collector camera bundle needs images and per-camera timing mappings")
        aliases = {"video6": "left_wrist", "video8": "right_wrist"}
        frames, times = {}, {}
        for source, frame in images.items():
            if not isinstance(source, str):
                raise ValueError("Collector camera source name must be a string")
            short = source.removeprefix("observation.images.")
            name = "observation.images." + aliases.get(short, short)
            if name not in CAMERAS:
                continue
            if name in frames:
                raise ValueError("Camera bundle contains duplicate canonical camera names")
            info = cameras.get(source, cameras.get(short))
            if not isinstance(info, dict) or "received_monotonic" not in info:
                raise ValueError("Collector camera bundle needs each camera's real received_monotonic time")
            stamp = float(info["received_monotonic"])
            if not np.isfinite(stamp) or stamp > received+1e-9:
                raise ValueError("Collector camera receipt time is nonfinite or future on the client clock")
            frames[name], times[name] = frame, stamp
        if set(frames) != set(CAMERAS):
            raise ValueError("Collector camera bundle requires ego_view and both wrists together")
        return frames, times

    def subscribe(self, endpoint, topic):
        socket = self.context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, topic)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(endpoint)
        self.sockets.append(socket)
        return socket

    @staticmethod
    def receive(socket, prefix=b""):
        try:
            raw = socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None
        if not raw.startswith(prefix):
            raise ValueError("Unexpected sensor topic")
        return msgpack.unpackb(raw[len(prefix):], raw=False, object_hook=msgpack_numpy.decode)

    def poll(self, now):
        body = self.receive(self.body, b"g1_debug")
        if body is not None:
            # The audited SONIC producer publishes body_q in MuJoCo native order
            # and physical radians including its default-angle offset.
            q = finite_array(body["body_q"], (29,), "SONIC measured body_q")
            self.body_sample = (time.monotonic(), q.copy())
            self.controller_status = body
            self.body_history.append(self.body_sample)
            handshake = getattr(self, "velocity_handshake", None)
            if handshake is not None:
                try:
                    handshake.observe(body, self.body_sample[0])
                except ValueError as exc:
                    raise RuntimeError("Velocity fallback needs native g1_debug ACK diagnostics; apply the controller ACK patch and rebuild") from exc
        hand = self.receive(self.hands)
        if hand is not None:
            received = time.monotonic()
            if hand.get("format") != "zeta0.brainco12.state.v1":
                raise ValueError("Requires measured BrainCo bridge, not commanded/7-joint hands")
            positions = finite_array(hand["positions"], (12,), "BrainCo measured positions")
            ages = finite_array(hand["age_s"], (2,), "BrainCo sample ages")
            if (ages < 0).any() or (positions < 0).any() or (positions > 1).any():
                raise ValueError("BrainCo measured positions must be normalized 0..1")
            for side in range(2):
                identity = (hand["session_id"], hand["sequence"][side])
                old = self.hand_samples[side]
                if old is None or identity != old[2]:
                    self.hand_samples[side] = (received-float(ages[side]), positions[side*6:side*6+6].copy(), identity)
                    self.hand_histories[side].append((*self.hand_samples[side][:2], received))
            self.hand_feedback, self.hand_received = hand, received
        receiver = getattr(self, "camera_receiver", None)
        if receiver is not None:
            try:
                cameras = receiver.poll()
            except (ValueError, msgpack.UnpackException) as exc:
                self.camera_bundle_error = str(exc) or type(exc).__name__
                cameras = None
        else:
            try:
                cameras = self.receive(self.camera)
            except (ValueError, msgpack.UnpackException) as exc:
                if self.observation_contract is None:
                    raise
                self.camera_bundle_error = str(exc) or type(exc).__name__
                cameras = None
        if cameras is not None:
            received = time.monotonic()
            if self.observation_contract is not None:
                try:
                    frames, times = self.decode_camera_bundle(cameras, received)
                    identity = tuple(times[name] for name in CAMERAS)
                    if not self.camera_bundles or identity != self.camera_bundles[-1][3]:
                        self.camera_bundles.append((received, frames, times, identity))
                    self.camera_bundle_error = None
                except (ValueError, TypeError, KeyError) as exc:
                    # An unusable visual bundle blocks new inference; feedback and
                    # the existing action queue retain their independent checks.
                    self.camera_bundle_error = str(exc) or type(exc).__name__
            for name in (() if self.observation_contract is not None else CAMERAS):
                short = name.removeprefix("observation.images.")
                if short not in cameras.get("images", {}):
                    continue
                identity = cameras.get("frame_ids", {}).get(short, cameras.get("timestamps", {}).get(short))
                if identity is None:
                    raise ValueError("Camera requires a per-camera frame identity or source timestamp")
                if identity != self.camera_identities.get(name):
                    self.camera_identities[name] = identity
                    self.images.setdefault(name, deque(maxlen=8)).append((received, cameras["images"][short]))
        now = time.monotonic()
        handshake = getattr(self, "velocity_handshake", None)
        if handshake is not None:
            from psi.deploy.brainco_bridge import HOLD_CAPABILITY

            self.supports_hold = handshake.strict_capable(now) and (
                getattr(self, "hand_feedback", None) is not None and
                self.hand_feedback.get("hold_capability") == HOLD_CAPABILITY)
        if self.body_sample is not None and all(x is not None for x in self.hand_samples):
            body_time, q = self.body_sample
            left, right = self.hand_samples
            if now - min(body_time, left[0], right[0]) <= .04:
                values = np.concatenate((q, left[1], right[1]))
                self.history.append(max(body_time, left[0], right[0]), values,
                                    component_times=(body_time, left[0], right[0]))

    def check_feedback(self, decision):
        """Separate model pairing from genuine body/hand stream availability."""
        self._check_stream_mode(time.monotonic())
        if getattr(self, "observation_contract", None) is None:
            return self.measured_state(decision)
        if self.body_sample is None or any(sample is None for sample in self.hand_samples):
            raise ValueError("Waiting for measured body and both BrainCo hands")
        age = self.observation_contract["state_max_age_s"]
        if decision-min(self.body_sample[0], *(sample[0] for sample in self.hand_samples)) > age+1e-9:
            raise ValueError(f"Measured body/hand feedback stream older than {age*1000:g}ms")
        handshake = getattr(self, "velocity_handshake", None)
        if handshake is not None and self.control_mode == "stream":
            now = time.monotonic()
            if not handshake.stream_accepting(now) or (handshake.phase == "stream" and not handshake.stream_alive(now)):
                self._request_idle(now)
                raise ValueError("Native SONIC did not acknowledge a fresh decoder token within its 100ms stream window")

    def _check_stream_mode(self, now):
        """Legacy mode ACK is a routing check, never a decoder-consumption ACK."""
        selected = getattr(self, "stream_selected_at", None)
        status = getattr(self, "controller_status", None)
        if (getattr(self, "velocity_handshake", None) is not None or selected is None
                or not isinstance(status, Mapping) or "input_control_mode" not in status):
            return
        mode = finite_array(status["input_control_mode"], (1,), "SONIC input_control_mode")[0]
        if mode not in (0, 1):
            raise ValueError("Invalid SONIC input_control_mode")
        if self.body_sample[0] > selected and mode == 1:
            self.stream_mode_acknowledged = True
        elif getattr(self, "stream_mode_acknowledged", False):
            raise ValueError("Native SONIC left token stream mode")
        elif now-selected > .5:
            raise ValueError("Native SONIC did not acknowledge token stream mode within 500ms; initialize the controller before retrying")

    def _request_idle(self, now):
        from psi.deploy.sonic_velocity_fallback import idle_messages

        handshake = self.velocity_handshake
        if self.control_mode != "idle":
            self.control_mode = "idle"
            self.hold_request_id = uuid.uuid4().hex
            self.hold_started = now
            hand = self.hand_feedback or {}
            self.hold_session = hand.get("session_id")
            self.hold_sequence = hand.get("sequence")
            messages = handshake.request_idle(now)
        elif handshake.phase == "idle":
            messages = idle_messages()
        else:
            messages = handshake.request_idle(now)
        for message in messages:
            self.pose.send(message)
        self.hand_commands.send(msgpack.packb(dict(format="zeta0.brainco12.command.v1",
            hold=True, hold_request_id=self.hold_request_id), use_bin_type=True))
        self.active = True

    def _hands_held(self, now):
        hand = self.hand_feedback or {}
        acknowledgement = hand.get("hold_ack", {})
        sequence = hand.get("sequence")
        if (self.hand_received is None or not self.hold_started <= self.hand_received <= now or
                acknowledgement.get("request_id") != self.hold_request_id or
                not isinstance(acknowledgement.get("acknowledged"), list) or
                len(acknowledgement["acknowledged"]) != 2 or
                not all(value is True for value in acknowledgement["acknowledged"]) or
                not isinstance(sequence, list) or len(sequence) != 2):
            return False
        if self.hold_session is not None and hand.get("session_id") != self.hold_session:
            raise RuntimeError("BrainCo bridge session changed during hold acknowledgement")
        return self.hold_sequence is None or all(sequence[side] > self.hold_sequence[side] for side in range(2))

    def prepare_control(self, now):
        """Nonblocking startup: real planner CONTROL ACK and both SDK write ACKs."""
        handshake = getattr(self, "velocity_handshake", None)
        if handshake is None:
            return True
        self._request_idle(now)
        if handshake.status is not None:
            if handshake.status.ack_version != 1 or not handshake.status.fallback_enabled:
                raise RuntimeError("Velocity fallback requires the rebuilt ACK-v1 controller with --velocity-fallback enabled")
        if self.hand_feedback is not None:
            from psi.deploy.brainco_bridge import HOLD_CAPABILITY

            if self.hand_feedback.get("hold_capability") != HOLD_CAPABILITY:
                raise RuntimeError("Velocity fallback requires the executing BrainCo bridge with measured-position hold ACK support")
        if now-self.hold_started > 5.:
            raise RuntimeError("Timed out waiting for native planner and BrainCo hold acknowledgements")
        return handshake.idle_confirmed(now) and self._hands_held(now) and handshake.strict_capable(now)

    def hold(self):
        """Request zero-velocity balance plus SDK measured-position holds; never block."""
        handshake = getattr(self, "velocity_handshake", None)
        if handshake is None:
            return False
        now = time.monotonic()
        self._request_idle(now)
        return handshake.idle_confirmed(now) and self._hands_held(now) and handshake.strict_capable(now)

    def control_stable(self, now):
        handshake = getattr(self, "velocity_handshake", None)
        return handshake is None or handshake.stream_confirmed(now)

    def collector_snapshot(self, decision):
        """Pair cached feedback to one intact ego-anchored camera snapshot."""
        limits = self.observation_contract["collector"]
        age = self.observation_contract["state_max_age_s"]
        def nearest(history, anchor):
            eligible = [sample for sample in history if decision-age-1e-9 <= sample[0] <= decision+1e-9
                        and (len(sample) < 3 or sample[2] <= decision+1e-9)]
            if limits["version"] in (2, 3):
                return max(eligible, key=lambda sample: sample[0]) if eligible else None
            return min(eligible, key=lambda sample: (abs(sample[0]-anchor), sample[0])) if eligible else None
        for received, frames, times, _ in reversed(self.camera_bundles):
            if received > decision+1e-9:
                continue
            anchor = times[CAMERAS[0]]
            body = nearest(self.body_history, anchor)
            hands = [nearest(history, anchor) for history in self.hand_histories]
            if body is None or any(sample is None for sample in hands):
                continue
            try:
                validate_snapshot_timing(decision_time=decision, camera_anchor_time=anchor,
                    camera_times=times, body_time=body[0], hand_times=[sample[0] for sample in hands], contract=limits,
                    snapshot_max_age_s=self.observation_contract.get("real_snapshot_max_age_s"))
            except ValueError:
                continue
            values = np.concatenate((body[1], *(sample[1] for sample in hands)))
            state = dict(values=[values.tolist()], valid=[[True]*41],
                         times=[max(body[0], *(sample[0] for sample in hands))-decision])
            return dict(state=state, images={name: dict(_source=frames[name], time=times[name]-decision) for name in CAMERAS},
                        snapshot_timing=dict(camera_anchor_time=anchor-decision, body_time=body[0]-decision,
                                             hand_times=[sample[0]-decision for sample in hands]))
        detail = self.camera_bundle_error or "No causal camera bundle with paired body/left/right feedback"
        raise ValueError(detail)

    def measured_state(self, anchor):
        """Check fresh, causal feedback independently of camera availability."""
        if self.body_sample is None or any(x is None for x in self.hand_samples):
            raise ValueError("Waiting for measured body and both BrainCo hands")
        if anchor - min(self.body_sample[0], *(x[0] for x in self.hand_samples)) > .04:
            raise ValueError("Measured body/hand feedback older than 40ms")
        state = self.history.aligned(anchor)
        if not all(state["valid"][-1]):
            raise ValueError("No causal current measured state")
        return state

    def observation(self, anchor):
        if getattr(self, "observation_contract", None) is not None:
            return self.collector_snapshot(anchor)
        state = self.measured_state(anchor)
        images = {}
        for name in CAMERAS:
            sample = next((s for s in reversed(self.images.get(name, ())) if s[0] <= anchor), None)
            if sample is None or not -.08 <= sample[0] - anchor <= 1e-6:
                raise ValueError(f"Missing or stale trained camera: {name}")
            # Codec work belongs to the network worker, outside the 50Hz loop.
            images[name] = dict(_source=sample[1], time=sample[0]-anchor)
        return dict(state=state, images=images)

    def execute(self, action):
        action = finite_array(action, (76,), "native executable action")
        if not self.config.execute:
            return
        action = controller_command_values(action, projection=getattr(self, "action_projection", None))
        handshake = getattr(self, "velocity_handshake", None)
        now = time.monotonic()
        self._check_stream_mode(now)
        if handshake is not None:
            if self.control_mode == "idle":
                self.pose.send(handshake.request_stream(now))
                self.control_mode = "stream"
            if not handshake.stream_accepting(now):
                self._request_idle(now)
                raise RuntimeError("Native SONIC stream-entry ACK deadline expired; returned to zero velocity")
        elif getattr(self, "stream_selected_at", None) is None:
            # Native mode changes clear any previously staged token. Select the
            # stream first, then publish only this slot's fresh token. The first
            # packet may still race the native reset; later slots are new tokens.
            self.pose.send(build_command_message(start=False, stop=False, planner=False))
            self.stream_selected_at = now
        token = action[:64]
        self.pose.send(pack_pose_message(dict(token_state=token.reshape(1,64).astype(np.float32),
            frame_index=np.array([self.frame_index], dtype=np.int64)), topic="pose", version=4))
        if handshake is not None:
            handshake.register_sent_frame(self.frame_index, now)
        self.hand_commands.send(msgpack.packb(dict(format="zeta0.brainco12.command.v1",
            positions=action[64:].tolist(), sequence=self.frame_index), use_bin_type=True))
        if handshake is None and not getattr(self, "stream_mode_acknowledged", False):
            # A reset can discard the first token and reject its start request.
            # Retry with each new absolute slot until native mode feedback ACKs.
            self.pose.send(build_command_message(start=True, stop=False, planner=False))
        self.active = True
        self.frame_index += 1

    def stop(self):
        if self.config.execute and self.active:
            for _ in range(3):
                self.pose.send(build_command_message(start=False, stop=True, planner=False))
                self.hand_commands.send(msgpack.packb(dict(format="zeta0.brainco12.command.v1", stop=True), use_bin_type=True))
                time.sleep(.01)
        self.active = False
        self.stream_selected_at = None
        self.stream_mode_acknowledged = False

    def close(self):
        try:
            self.stop()
        finally:
            if getattr(self, "camera_receiver", None) is not None:
                self.camera_receiver.close()
            for socket in self.sockets:
                socket.close(linger=0)
            self.context.term()
