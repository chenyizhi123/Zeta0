"""Wire and physical-time contracts for recorded G1 q29 / BrainCo12 policies.

All timestamps belong to the client workstation's monotonic clock. The server
uses only relative times and echoes the client anchor; clocks are never mixed.
"""
import threading
from collections import deque

import numpy as np

from psi.real_observation_contract import (
    validate_action_projection,
    validate_deployment_observation_contract,
    validate_snapshot_timing,
)
from psi.sonic_actions import controller_command_values

FORMAT = "zeta0.real_sonic.v1"
PROTOCOL = "real_sonic_motion64_brainco12_v1"
STATE_SCHEMA = "real_g1_q29_brainco12_v1"
CAMERAS = tuple("observation.images." + x for x in ("ego_view", "left_wrist", "right_wrist"))
PERIOD = 0.02
HORIZON = 30
ACTION_DIM = 76
STATE_DIM = 41
LEGACY_RTC_PREFIX = 6
MAX_RTC_PREFIX = 15


def validate_action_grid(action_hz, action_horizon):
    """Supported model output grids; the physical queue below remains 50Hz."""
    if (type(action_hz) not in (int, float) or type(action_horizon) is not int
            or (action_hz, action_horizon) not in {(50, 30), (30, 50)}):
        raise ValueError("Unsupported real-SONIC action grid: expected 50Hz/30 steps or 30Hz/50 steps")
    return float(action_hz), action_horizon


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def validate_request(request, *, state_count=21, rtc_max_prefix_steps=LEGACY_RTC_PREFIX,
                     observation_contract=None):
    contract = None if observation_contract is None else validate_deployment_observation_contract(observation_contract)
    if contract is not None and state_count != contract["state_history_count"]:
        raise ValueError("Collector request state count differs from checkpoint contract")
    state_age = .04 if contract is None else contract["state_max_age_s"]
    visual_age = .08 if contract is None else contract["visual_max_age_s"]
    if type(rtc_max_prefix_steps) is not int or not 0 <= rtc_max_prefix_steps <= MAX_RTC_PREFIX:
        raise ValueError("Unsupported real-SONIC RTC prefix capacity")
    if type(state_count) is not int or state_count not in (1, 21):
        raise ValueError("Real SONIC supports one current state or 21 history states")
    if request.get("format") != FORMAT or request.get("action_protocol_id") != PROTOCOL:
        raise ValueError("Unsupported wire format or native action protocol")
    if request.get("state_schema_id") != STATE_SCHEMA:
        raise ValueError("Unsupported measured state schema")
    for name in ("request_seq", "queue_revision"):
        if type(request.get(name)) is not int or request[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(request.get("session_id"), str) or not request["session_id"]:
        raise ValueError("session_id must be a nonempty string")
    if not np.isfinite(float(request["anchor_time"])):
        raise ValueError("anchor_time must be finite")
    state = request["state"]
    values = np.asarray(state["values"], dtype=np.float32)
    valid = np.asarray(state["valid"])
    times = np.asarray(state["times"], dtype=np.float64)
    if values.shape != (state_count, 41) or valid.shape != values.shape or valid.dtype != bool or times.shape != (state_count,):
        raise ValueError(f"State requires values/boolean valid [{state_count},41], times [{state_count}]")
    present = valid.any(-1)
    ages = np.arange(1 - state_count, 1) * PERIOD - times
    if (not valid[-1].all() or not np.isfinite(values[valid]).all()
            or not np.isfinite(times[present]).all()
            or (ages[present] < -1e-6).any() or (ages[present] > state_age + 1e-6).any()):
        raise ValueError(f"State is missing, future, nonfinite, or older than trained {state_age*1000:g}ms")
    if set(request["images"]) != set(CAMERAS):
        raise ValueError("This real-robot client requires all three trained cameras")
    for image in request["images"].values():
        time = float(image["time"])
        if not np.isfinite(time) or not -visual_age - 1e-6 <= time <= 1e-6:
            raise ValueError(f"Camera is future or older than trained {visual_age*1000:g}ms")
        if not isinstance(image.get("encoded"), str) or not image["encoded"]:
            raise ValueError("Camera needs a base64 encoded RGB image")
    if contract is not None:
        timing = request.get("snapshot_timing")
        if not isinstance(timing, dict) or set(timing) != {"camera_anchor_time", "body_time", "hand_times"}:
            raise ValueError("Collector request requires camera/body/hand snapshot_timing")
        validate_snapshot_timing(decision_time=0., camera_anchor_time=timing["camera_anchor_time"],
            camera_times={name: image["time"] for name, image in request["images"].items()},
            body_time=timing["body_time"], hand_times=timing["hand_times"], contract=contract["collector"],
            snapshot_max_age_s=contract.get("real_snapshot_max_age_s"))
        if abs(float(times[-1])-max(float(timing["body_time"]), *(float(t) for t in timing["hand_times"]))) > 1e-9:
            raise ValueError("Collector state time must equal its selected component max")
    prefix = np.asarray(request.get("committed_prefix", []), dtype=np.float32)
    if prefix.size == 0:
        prefix = np.empty((0, ACTION_DIM), dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != ACTION_DIM or len(prefix) > rtc_max_prefix_steps:
        raise ValueError(f"RTC prefix requires [0..{rtc_max_prefix_steps},76], as supported by this checkpoint")
    if not np.isfinite(prefix).all():
        raise ValueError("RTC prefix contains nonfinite commands")
    if contract is not None and contract.get("action_projection") is not None:
        canonical = controller_command_values(prefix, projection=contract["action_projection"])
        if not np.array_equal(prefix, canonical):
            raise ValueError("RTC prefix differs from the checkpoint's projected controller commands")
    return values, valid, times, prefix


class MeasuredHistory:
    """Retain genuine combined samples and their body/left/right receive times.

    Causal floor selection uses the combined max timestamp, while dimension
    masks check each selected sample's component age against the query. Legacy
    callers without component times retain their combined-timestamp behavior.
    """
    def __init__(self, *, state_count=21):
        if type(state_count) is not int or state_count not in (1, 21):
            raise ValueError("Measured history requires state_count 1 or 21")
        self.state_count = state_count
        self.samples = deque(maxlen=512)

    def append(self, received, values, *, component_times=None):
        value = finite_array(values, (STATE_DIM,), "measured q29 + hand12")
        if not np.isfinite(received):
            raise ValueError("Invalid receive time")
        components = None
        if component_times is not None:
            components = np.asarray(component_times, dtype=np.float64)
            if (components.shape != (3,) or not np.isfinite(components).all()
                    or abs(float(components.max())-received) > 1e-9):
                raise ValueError("Body/left/right component times must be finite and match the combined max receive time")
            components = components.copy()
        if self.samples and received < self.samples[-1][0]:
            raise ValueError("State receive clock regressed")
        if not self.samples or received > self.samples[-1][0]:
            self.samples.append((float(received), value.copy(), components))

    def aligned(self, anchor):
        queries = anchor + np.arange(1 - self.state_count, 1, dtype=np.float64) * PERIOD
        values = np.zeros((self.state_count, STATE_DIM), np.float32)
        valid = np.zeros_like(values, dtype=bool)
        relative = np.zeros(self.state_count, dtype=np.float64)
        if self.samples:
            times = np.asarray([sample[0] for sample in self.samples])
            indices = np.searchsorted(times, queries, side="right") - 1
            for i, idx in enumerate(indices):
                if idx >= 0 and queries[i] - times[idx] <= .04 + 1e-9:
                    values[i] = self.samples[idx][1]
                    valid[i] = True
                    components = self.samples[idx][2]
                    if components is not None:
                        age = queries[i]-components
                        present = (age >= -1e-9) & (age <= .04+1e-9)
                        valid[i,:29], valid[i,29:35], valid[i,35:] = present
                    relative[i] = times[idx] - anchor
        return dict(values=values.tolist(), valid=valid.tolist(), times=relative.tolist())


class AbsoluteActionQueue:
    """A single session's actions at anchor+20ms ... anchor+600ms.

    Replies never restart their horizon on arrival. A committed RTC prefix stays
    unchanged until its reserved slots have passed. Reset invalidates all replies.
    Stored actions and snapshots are controller command values: bounded FSQ
    tokens and normalized hand targets, before the bridge's SDK rate limit.
    """
    def __init__(self, *, rtc_max_prefix_steps=LEGACY_RTC_PREFIX, action_projection=None):
        if type(rtc_max_prefix_steps) is not int or not 0 <= rtc_max_prefix_steps <= MAX_RTC_PREFIX:
            raise ValueError("Unsupported real-SONIC RTC prefix capacity")
        self.rtc_max_prefix_steps = rtc_max_prefix_steps
        self.action_projection = None if action_projection is None else validate_action_projection(action_projection)
        self.lock = threading.Lock()
        self.revision = 0
        self.session = ""
        self.commands = {}

    @staticmethod
    def key(time):
        return round(float(time) / PERIOD)

    def reset(self, session):
        with self.lock:
            self.session = session
            self.revision += 1
            self.commands.clear()

    def snapshot(self, anchor, prefix_steps):
        if type(prefix_steps) is not int or not 0 <= prefix_steps <= self.rtc_max_prefix_steps:
            raise ValueError(f"RTC prefix steps must be 0..{self.rtc_max_prefix_steps}")
        first = self.key(anchor) + 1
        with self.lock:
            values = [self.commands.get(first + i) for i in range(prefix_steps)]
            prefix = [] if any(v is None for v in values) else [v.tolist() for v in values]
            return self.revision, prefix

    def install(self, request, response, now):
        actions = finite_array(response["actions"], (HORIZON, ACTION_DIM), "native action chunk")
        for name in ("format", "session_id", "request_seq", "queue_revision", "anchor_time", "action_protocol_id"):
            if response.get(name) != request[name]:
                raise ValueError(f"Response envelope mismatch: {name}")
        prefix = np.asarray(request.get("committed_prefix", []), dtype=np.float32)
        count = len(prefix)
        if count and (now > request["anchor_time"] + count * PERIOD
                      or not np.allclose(actions[:count], prefix, atol=1e-6, rtol=1e-5)):
            raise ValueError("RTC reply missed its reserved prefix or changed committed actions")
        # Compare the raw reply above: projection must not hide a changed prefix.
        actions = controller_command_values(actions, projection=self.action_projection)
        first = self.key(request["anchor_time"]) + 1
        # Only future command slots are eligible, even during initial priming.
        pending = {first+i: a.copy() for i, a in enumerate(actions)
                   if (first+i) * PERIOD > now and i >= count}
        if not pending:
            raise ValueError("Entire prediction horizon expired in transit")
        with self.lock:
            if self.session != request["session_id"] or self.revision != request["queue_revision"]:
                raise ValueError("Reply belongs to a stale session or queue revision")
            preserved = {k: v for k, v in self.commands.items() if k < first + count and k * PERIOD > now}
            self.commands = {**preserved, **pending}
            self.revision += 1
            return len(pending)

    def pop(self, scheduled_time):
        key = self.key(scheduled_time)
        with self.lock:
            for old in [k for k in self.commands if k < key]:
                del self.commands[old]
            return self.commands.pop(key, None)

    def skip_to(self, scheduled_time, *, missed_current=False):
        """Drop missed absolute slots and revoke replies conditioned on those commitments.

        Future commands retain their original deadlines; no old command is moved
        to a new slot. ``missed_current`` also revokes a command already popped
        by this tick when sensor work/scheduling delayed its physical send.
        """
        if type(missed_current) is not bool:
            raise ValueError("missed_current must be boolean")
        key = self.key(scheduled_time)
        with self.lock:
            missed = [slot for slot in self.commands if slot < key]
            for slot in missed:
                del self.commands[slot]
            if missed or missed_current:
                self.revision += 1
            return len(missed)+int(missed_current)

    def depth(self, now):
        with self.lock:
            return sum(k * PERIOD > now for k in self.commands)
