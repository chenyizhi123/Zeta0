"""Acknowledged SONIC zero-velocity planning, with no sockets or robot imports.

The recovered controller publishes these diagnostics as top-level one-element
arrays in ``g1_debug``. Its steady clock is *not* the client's monotonic clock:
only local receipt ages and progress within one controller PID are compared.
Legacy diagnostics can confirm IDLE, but cannot authorize token resumption.
The companion C++ patch adds actual decoder-input commit acknowledgements.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from psi.deploy.sonic_wire import build_command_message, build_planner_message


ACK_FORMAT = "zeta0.sonic.velocity_fallback_ack.v1"
ACK_VERSION = 1
FEEDBACK_MAX_AGE_S = .1
TOKEN_MAX_AGE_S = .1
STREAM_ACK_TIMEOUT_S = .1  # Matches the native stream-entry grace; never renewed.
IDLE_ACK_TIMEOUT_S = 5.0  # Native planner initialization timeout.


def _number(packet, name, *, integer=False, boolean=False):
    if name not in packet:
        raise ValueError(f"Missing native SONIC {name} diagnostic")
    value = np.asarray(packet[name])
    if value.shape != (1,) or value.dtype.kind not in "biuf":
        raise ValueError(f"SONIC {name} must be a one-element numeric array")
    result = float(value[0])
    if not math.isfinite(result):
        raise ValueError(f"SONIC {name} must be finite")
    if integer and (result != int(result) or abs(result) > 2**53):
        raise ValueError(f"SONIC {name} must be an exactly represented integer")
    if boolean and result not in (0., 1.):
        raise ValueError(f"SONIC {name} must be 0 or 1")
    return int(result) if integer or boolean else result


def _clock(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Client monotonic time must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class ControllerStatus:
    pid: int
    controller_time: float
    received: float
    mode: int
    encoder_active: bool
    planner_ready: bool
    fallback_enabled: bool
    zero_velocity_requested: bool
    ack_version: int | None = None
    control_active: bool = True  # Legacy g1_debug is published only in CONTROL.
    token_stream_ready: bool = False
    token_input_valid: bool = False
    token_commit_sequence: int = 0
    token_frame_index: int = -1
    token_receive_age_s: float = -1.

    @classmethod
    def from_packet(cls, packet, received):
        if not isinstance(packet, Mapping):
            raise ValueError("SONIC g1_debug must be a top-level mapping")
        extension = (
            "velocity_fallback_ack_version", "control_active", "token_stream_ready",
            "decoder_token_input_valid", "decoder_token_commit_sequence",
            "decoder_token_frame_index", "decoder_token_receive_age_s",
        )
        present = [key in packet for key in extension]
        if any(present) and not all(present):
            raise ValueError("Incomplete native SONIC token ACK diagnostics")
        extras = {}
        if all(present):
            version = _number(packet, extension[0], integer=True)
            if version != ACK_VERSION:
                raise ValueError("Unsupported native SONIC token ACK version")
            extras = dict(
                ack_version=version,
                control_active=bool(_number(packet, "control_active", boolean=True)),
                token_stream_ready=bool(_number(packet, "token_stream_ready", boolean=True)),
                token_input_valid=bool(_number(packet, "decoder_token_input_valid", boolean=True)),
                token_commit_sequence=_number(packet, "decoder_token_commit_sequence", integer=True),
                token_frame_index=_number(packet, "decoder_token_frame_index", integer=True),
                token_receive_age_s=_number(packet, "decoder_token_receive_age_s"),
            )
            if extras["token_commit_sequence"] < 0 or extras["token_frame_index"] < -1:
                raise ValueError("Invalid native SONIC token ACK counter/frame")
        result = cls(
            pid=_number(packet, "controller_pid", integer=True),
            controller_time=_number(packet, "controller_time_monotonic"),
            received=_clock(received),
            mode=_number(packet, "input_control_mode", integer=True),
            encoder_active=bool(_number(packet, "encoder_active", boolean=True)),
            planner_ready=bool(_number(packet, "planner_ready", boolean=True)),
            fallback_enabled=bool(_number(packet, "velocity_fallback_enabled", boolean=True)),
            zero_velocity_requested=bool(_number(packet, "zero_velocity_requested", boolean=True)),
            **extras,
        )
        if result.pid <= 0 or result.controller_time < 0 or result.mode not in (0, 1):
            raise ValueError("Invalid native SONIC controller identity/clock/mode")
        return result


def idle_messages(*, facing=(1., 0., 0.), upper_body_position=None):
    """Stage IDLE before requesting PLANNER/start; never send network stop here."""
    direction = np.asarray(facing, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("Idle planner facing must be a finite nonzero 3-vector")
    if upper_body_position is not None:
        upper = np.asarray(upper_body_position, dtype=float)
        if upper.shape != (17,) or not np.isfinite(upper).all():
            raise ValueError("Idle planner upper body must be 17 finite native reference values")
        upper_body_position = upper.tolist()
    return (
        build_planner_message(0, (0., 0., 0.), direction, speed=0., height=-1.,
                              upper_body_position=upper_body_position),
        build_command_message(start=True, stop=False, planner=True),
    )


class VelocityFallbackHandshake:
    """One fail-closed handshake: IDLE -> stream entry -> acknowledged input.

    Send only new action slots while ``stream_accepting`` is true, then register
    each successful send. An ACK must identify an exact frame sent in this epoch;
    CONFLATE may omit earlier ACKs. Cached-token reuse does not increment the
    native commit counter. No mode packet or model ``token_state`` is an ACK.
    """

    def __init__(self):
        self.phase = "unbound"
        self.status = None
        self.error = None
        self._requested = None
        self._baseline_time = None
        self._sent = OrderedDict()
        self._first_frame = None

    def _fail(self, message):
        self.phase, self.error = "failed", message
        raise ValueError(message)

    def observe(self, packet, received_monotonic, now=None):
        if self.phase == "failed":
            raise ValueError(self.error)
        try:
            status = ControllerStatus.from_packet(packet, received_monotonic)
            now = status.received if now is None else _clock(now)
            if not 0 <= now-status.received <= FEEDBACK_MAX_AGE_S:
                raise ValueError("Native SONIC feedback receipt is stale or future")
            previous = self.status
            if previous is not None:
                if status.pid != previous.pid:
                    raise ValueError("Native SONIC controller PID changed; restart the handshake")
                if status.received < previous.received or status.controller_time < previous.controller_time:
                    raise ValueError("Native SONIC feedback clock regressed")
                if status.ack_version != previous.ack_version:
                    raise ValueError("Native SONIC ACK capability changed within one PID")
                if status.token_commit_sequence < previous.token_commit_sequence:
                    raise ValueError("Native SONIC token commit counter regressed")
                if status.controller_time == previous.controller_time:
                    return False  # Duplicate receipts must not refresh old feedback.
            self.status = status
            if self._requested is not None and self._baseline_time is None:
                self._baseline_time = status.controller_time
            if self.phase == "idle_pending" and self._idle_ack(now):
                self.phase = "idle"
            elif self.phase == "stream_pending" and self._stream_ack(now):
                self.phase = "stream"
            return True
        except (KeyError, TypeError, ValueError) as exc:
            self._fail(str(exc))

    def _fresh(self, now):
        now = _clock(now)
        return self.status is not None and 0 <= now-self.status.received <= FEEDBACK_MAX_AGE_S

    def strict_capable(self, now):
        return self.phase != "failed" and self._fresh(now) and (
            self.status.ack_version == ACK_VERSION and self.status.fallback_enabled
        )

    def _progressed(self):
        return (self._requested is not None and self._baseline_time is not None and
                self.status.received >= self._requested and
                self.status.controller_time > self._baseline_time)

    def _idle_ack(self, now):
        return self._fresh(now) and self._progressed() and (
            self.status.mode == 0 and self.status.planner_ready and
            self.status.fallback_enabled and self.status.zero_velocity_requested and
            self.status.control_active
        )

    def request_idle(self, now, *, facing=(1., 0., 0.), upper_body_position=None):
        now = _clock(now)
        if self.phase == "failed":
            raise ValueError(self.error)
        messages = idle_messages(facing=facing, upper_body_position=upper_body_position)
        if self.phase != "idle_pending":
            self.phase, self._requested = "idle_pending", now
            self._baseline_time = self.status.controller_time if self.status is not None else None
            self._sent.clear()
            self._first_frame = None
        return messages

    def idle_confirmed(self, now):
        now = _clock(now)
        if self.phase == "idle_pending" and now-self._requested > IDLE_ACK_TIMEOUT_S:
            self._fail("Native SONIC IDLE acknowledgement timed out")
        return self.phase == "idle" and self._idle_ack(now)

    def request_stream(self, now, frame_index=None):
        now = _clock(now)
        if not self.idle_confirmed(now) or not self.strict_capable(now):
            raise ValueError("Native SONIC stream requires acknowledged IDLE and token ACK v1")
        if frame_index is not None and (isinstance(frame_index, bool) or int(frame_index) != frame_index or frame_index < 0):
            raise ValueError("Native SONIC frame index must be a nonnegative integer")
        self.phase, self._requested = "stream_pending", now
        self._baseline_time = self.status.controller_time
        self._first_frame = frame_index
        self._sent.clear()
        return build_command_message(start=True, stop=False, planner=False)

    def stream_accepting(self, now):
        now = _clock(now)
        if self.phase == "stream_pending" and now-self._requested > STREAM_ACK_TIMEOUT_S:
            return False
        # Send immediately after the mode command. Waiting for mode/encoder
        # feedback can exhaust the native grace; a reset may discard the first
        # packet, so subsequent *new* slots may establish the real commit ACK.
        if self.phase == "stream_pending":
            return self.strict_capable(now) and self.status.control_active
        return self.phase == "stream" and self.strict_capable(now) and (
            self._progressed() and self.status.mode == 1 and
            self.status.control_active and self.status.token_stream_ready
        )

    def register_sent_frame(self, frame_index, now):
        now = _clock(now)
        if not self.stream_accepting(now):
            raise ValueError("Native SONIC stream has not acknowledged readiness")
        if isinstance(frame_index, bool) or int(frame_index) != frame_index or not 0 <= frame_index <= 2**53:
            raise ValueError("Native SONIC frame index must be an exactly represented nonnegative integer")
        frame_index = int(frame_index)
        if self._sent and frame_index <= next(reversed(self._sent)):
            raise ValueError("Native SONIC sends must use new increasing frame indices")
        if not self._sent and self._first_frame is not None and frame_index != self._first_frame:
            raise ValueError("Native SONIC first sent frame differs from requested frame")
        self._sent[frame_index] = (now, self.status.controller_time, self.status.token_commit_sequence)
        if len(self._sent) > 16:
            self._sent.popitem(last=False)

    note_token_sent = register_sent_frame

    def _stream_ack(self, now):
        if not self.stream_accepting(now) or self.status.token_frame_index not in self._sent:
            return False
        sent, clock, sequence = self._sent[self.status.token_frame_index]
        return (self._progressed() and self.status.mode == 1 and
                self.status.control_active and self.status.token_stream_ready and
                self.status.received >= sent and self.status.controller_time > clock and
                self.status.token_commit_sequence > sequence and self.status.token_input_valid and
                not self.status.encoder_active and
                0 <= self.status.token_receive_age_s <= TOKEN_MAX_AGE_S)

    def stream_confirmed(self, now):
        now = _clock(now)
        if self.phase == "stream_pending" and now-self._requested > STREAM_ACK_TIMEOUT_S:
            return False
        return self.phase == "stream" and self._stream_ack(now)

    def stream_alive(self, now):
        return self.stream_confirmed(now)
