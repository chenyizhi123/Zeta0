"""LeRobot plugin for recording an independently running SONIC/PICO controller.

Only SUB sockets are opened. Robot.send_action acknowledges already observed
commands; SONIC and the BrainCo bridge remain the only actuator publishers.
LeRobot owns the recording loop, camera workers, dataset and episode lifecycle.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
import time

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import zmq

from lerobot.cameras.configs import CameraConfig, ColorMode
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig
from lerobot.robots import Robot, RobotConfig
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceNotConnectedError


DEFAULT_LOCAL_CONFIG = str(Path(__file__).resolve().parents[3] / 'config/local.json')
CHANNELS = ('thumb', 'thumb_aux', 'index', 'middle', 'ring', 'pinky')
BODY_NAMES = tuple(f'body.q_{i:02d}' for i in range(29))
HAND_NAMES = tuple(f'{side}.{finger}.q' for side in ('left', 'right') for finger in CHANNELS)
TOKEN_NAMES = tuple(f'sonic.motion_token_{i:02d}' for i in range(64))
STATE_NAMES = BODY_NAMES + HAND_NAMES
ACTION_NAMES = TOKEN_NAMES + HAND_NAMES
logger = logging.getLogger(__name__)


def local_config(path):
    return json.loads(Path(path).read_text())


def finite_vector(value, shape, label, *, normalized=False):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{label}: expected finite {shape} values')
    if normalized and np.any((array < 0) | (array > 1)):
        raise ValueError(f'{label}: expected normalized positions in [0,1]')
    return array


@CameraConfig.register_subclass('sonic_zmq')
@dataclass
class SonicCameraConfig(ZMQCameraConfig):
    expected_usb_serial: str = ''
    expected_usb_vendor: str = ''
    expected_usb_product: str = ''
    expected_usb_interface: str = ''


class CameraStreamUnavailable(TimeoutError):
    """A missing/stale camera frame; no observation has been added yet."""


class SonicCamera(ZMQCamera):
    """Adapt the robot's MessagePack/JPEG wire format to the official camera.

    Buffering, latest-frame reads and stale-frame checks come from LeRobot.
    The same worker keeps waiting through temporary receive timeouts. Each
    camera is read independently.
    """

    def _read_loop(self):
        # LeRobot exits its worker after consecutive receive timeouts. Keep
        # that same worker waiting for a temporarily absent publisher instead.
        # Cached timestamps remain untouched: read_latest still rejects stale
        # frames, and startup warmup retains its original finite timeout.
        stop_event = self.stop_event
        if stop_event is None:
            return super()._read_loop()
        while not stop_event.is_set():
            try:
                return super()._read_loop()
            except RuntimeError as error:
                if stop_event.is_set():
                    return
                if not isinstance(error.__cause__, TimeoutError):
                    raise  # Identity/format errors must never be accepted.
                logger.warning('%s: stream timed out; waiting for fresh frames to resume', self)

    def _cleanup(self):
        # connect() also calls cleanup when warmup fails. Stop recv before its
        # socket/context are closed, including when no frame ever arrived.
        if self.thread is not None:
            self._stop_read_thread()
        super()._cleanup()

    def _read_from_hardware(self):
        if not self.is_connected or self.socket is None:
            raise DeviceNotConnectedError(f'{self} is not connected')
        try:
            raw = self.socket.recv()
        except zmq.Again as error:
            raise TimeoutError(f'{self}: no frame within {self.timeout_ms}ms') from error
        payload = msgpack.unpackb(raw, raw=False, object_hook=msgpack_numpy.decode)
        for field, expected in (
            ('serial', self.config.expected_usb_serial),
            ('idVendor', self.config.expected_usb_vendor),
            ('idProduct', self.config.expected_usb_product),
            ('usb_interface', self.config.expected_usb_interface),
        ):
            if expected and str(payload.get('camera', {}).get(field, '')).lower() != expected.lower():
                raise ValueError(f'{self}: camera {field} does not match the configured device')
        value = payload['images'][self.camera_name]
        if isinstance(value, str):
            value = base64.b64decode(value)
        if isinstance(value, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'{self}: invalid JPEG')
            if self.color_mode == ColorMode.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.asarray(value)
        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise ValueError(f'{self}: expected {self.width}x{self.height} uint8 RGB')
        return image


class SonicTelemetry:
    """Latest received body and actual hand state/command packets; no history matching."""

    def __init__(self, config_path, max_age_ms=200.):
        config = local_config(config_path)
        self.endpoints = {
            'body': f"tcp://127.0.0.1:{config.get('state_port',5557)}",
            'hands': config.get('gripper_endpoint', 'tcp://127.0.0.1:5563'),
        }
        self.max_age_s = max_age_ms / 1000
        if not np.isfinite(max_age_ms) or max_age_ms <= 0:
            raise ValueError('max_age_ms must be positive')
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.samples = {}
        self.error = None
        self.thread = None

    def connect(self, *, wait_for_feedback=True):
        self.thread = threading.Thread(target=self._receive, daemon=True, name='lerobot-sonic-telemetry')
        self.thread.start()
        try:
            if not self.ready.wait(5):
                raise RuntimeError('SONIC telemetry subscriber initialization timed out')
            if self.error is not None:
                raise RuntimeError('SONIC telemetry subscriber initialization failed') from self.error
            if not wait_for_feedback:
                # OFF has no body debug packets or actual hand commands yet.
                # The manual READY gate checks live feedback before any frame.
                return
            deadline = time.monotonic() + 5
            while True:
                try:
                    self.snapshot()
                    return
                except (TimeoutError, ValueError, KeyError):
                    if self.error is not None or time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
        except BaseException:
            self.close()
            raise

    def _receive(self):
        context = zmq.Context()
        sockets = {}
        poller = zmq.Poller()
        try:
            for kind, endpoint in self.endpoints.items():
                sock = context.socket(zmq.SUB)
                sock.setsockopt(zmq.SUBSCRIBE, b'g1_debug' if kind == 'body' else b'')
                sock.setsockopt(zmq.CONFLATE, 1)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)
                sockets[sock] = kind
                poller.register(sock, zmq.POLLIN)
            self.ready.set()
            while not self.stop.is_set():
                for sock in dict(poller.poll(20)):
                    raw = sock.recv()
                    kind = sockets[sock]
                    if kind == 'body':
                        payload = msgpack.unpackb(raw[len(b'g1_debug'):], raw=False,
                                                  object_hook=msgpack_numpy.decode)
                    else:
                        payload = json.loads(raw)
                    with self.lock:
                        self.samples[kind] = (payload, time.monotonic())
        except BaseException as error:
            self.error = error
        finally:
            self.ready.set()
            for sock in sockets:
                sock.close(linger=0)
            context.term()

    def snapshot(self):
        if self.error is not None:
            raise RuntimeError('SONIC telemetry receiver failed') from self.error
        with self.lock:
            samples = self.samples.copy()
        now = time.monotonic()
        for kind in self.endpoints:
            if kind not in samples or now - samples[kind][1] > self.max_age_s:
                raise TimeoutError(f'Waiting for fresh SONIC {kind} telemetry (max {self.max_age_s*1000:g}ms)')
        body, hands = samples['body'][0], samples['hands'][0]
        if hands.get('valid') is not True:
            raise ValueError('Actual BrainCo state/command feedback unavailable: ' + str(hands.get('reason')))
        packet_age = now - hands['timestamp_monotonic']
        channel_ages = np.asarray(hands['receive_ages'], dtype=float) + packet_age
        if (channel_ages.shape != (4,) or not np.isfinite(channel_ages).all()
                or np.any(channel_ages < 0) or np.any(channel_ages > self.max_age_s)):
            raise TimeoutError('Actual BrainCo state/command channels are stale')
        q = finite_vector(body.get('body_q'), (29,), 'body_q')
        token = finite_vector(body.get('token_state'), (64,), 'actual SONIC motion token')
        state = finite_vector(hands.get('state_joints'), (2, 6), 'actual hand feedback', normalized=True)
        action = finite_vector(hands.get('action_joints'), (2, 6), 'actual DDS hand commands', normalized=True)
        return q, state.reshape(12), token, action.reshape(12)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


@RobotConfig.register_subclass('sonic_g1')
@dataclass
class SonicRobotConfig(RobotConfig):
    local_config: str = DEFAULT_LOCAL_CONFIG
    recording_cameras: str = 'all'
    max_age_ms: float = 200.


class SonicRobot(Robot):
    config_class = SonicRobotConfig
    name = 'sonic_g1'

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.telemetry = SonicTelemetry(config.local_config, config.max_age_ms)
        self._connected = False
        self.cameras = {}
        if config.recording_cameras not in ('all', 'head'):
            raise ValueError('recording_cameras must be all or head')
        local = local_config(config.local_config)
        sources = local['camera_sources']
        names = [('ego_view', 'ego_view')]
        if config.recording_cameras == 'all':
            mapping = local['wrist_mapping']
            if set(mapping.values()) != {'left_wrist', 'right_wrist'}:
                raise ValueError('The two wrist cameras must have distinct left/right bindings')
            names += [(source, label) for source, label in mapping.items()]
        for source_name, label in names:
            source = sources[source_name]
            self.cameras[label] = SonicCamera(SonicCameraConfig(
                server_address=source['host'], port=source['port'], camera_name=source['key'],
                width=640, height=480, fps=30, warmup_s=3, timeout_ms=500,
                expected_usb_serial=source.get('usb_serial', ''),
                expected_usb_vendor=source.get('usb_vendor_id', ''),
                expected_usb_product=source.get('usb_product_id', ''),
                expected_usb_interface=source.get('usb_interface', ''),
            ))

    @property
    def observation_features(self):
        return {**dict.fromkeys(STATE_NAMES, float),
                **{name: (480, 640, 3) for name in self.cameras}}

    @property
    def action_features(self):
        return dict.fromkeys(ACTION_NAMES, float)

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        try:
            self.telemetry.connect(wait_for_feedback=False)
            for camera in self.cameras.values():
                camera.connect()
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self):
        if not self.is_connected:
            raise DeviceNotConnectedError('SONIC recorder is not connected')
        q, hands, _, _ = self.telemetry.snapshot()
        observation = dict(zip(STATE_NAMES, np.concatenate((q, hands)).tolist()))
        for name, camera in self.cameras.items():
            try:
                observation[name] = camera.read_latest(max_age_ms=self.config.max_age_ms).copy()
            except TimeoutError as error:
                raise CameraStreamUnavailable(str(error)) from error
        return observation

    def send_action(self, action):
        # These values were already sent by the external controller. Never
        # create a DDS writer or re-send robot commands from the recorder.
        finite_vector([action[name] for name in ACTION_NAMES], (76,), 'observed actions')
        return dict(action)

    def disconnect(self):
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self.telemetry.close()
        self._connected = False


@TeleoperatorConfig.register_subclass('sonic_actions')
@dataclass
class SonicActionsConfig(TeleoperatorConfig):
    local_config: str = DEFAULT_LOCAL_CONFIG
    max_age_ms: float = 200.


class SonicActions(Teleoperator):
    """Expose actions actually published by the existing PICO/SONIC session."""
    config_class = SonicActionsConfig
    name = 'sonic_actions'

    def __init__(self, config):
        super().__init__(config)
        self.telemetry = SonicTelemetry(config.local_config, config.max_age_ms)
        self._connected = False

    @property
    def action_features(self):
        return dict.fromkeys(ACTION_NAMES, float)

    @property
    def feedback_features(self):
        return {}

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        self.telemetry.connect(wait_for_feedback=False)
        self._connected = True

    def get_action(self):
        if not self.is_connected:
            raise DeviceNotConnectedError('SONIC action subscriber is not connected')
        _, _, token, hand_q = self.telemetry.snapshot()
        return dict(zip(ACTION_NAMES, np.concatenate((token, hand_q)).tolist()))

    def send_feedback(self, feedback):
        pass

    def disconnect(self):
        self.telemetry.close()
        self._connected = False
