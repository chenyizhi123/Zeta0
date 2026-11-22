"""SONIC camera receipts and the supplied local collector composer's wire adapter.

Independent sources use this client's actual receive clock. The explicitly
selected local composer preserves its original same-workstation receive times.
Robot timestamps identify frames and remain audit fields, never converted into
the client's monotonic clock. Encoded image payloads stay encoded here.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import ipaddress
import json
import math
import time
from urllib.parse import urlparse

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from psi.real_observation_contract import CAMERA_KEYS, validate_observation_contract

_ALIASES = {"video6": "left_wrist", "video8": "right_wrist"}
_EPS = 1e-9
COMPOSER_FORMAT = "chenyz_composer_v1"


def canonical_camera_name(name):
    if not isinstance(name, str):
        raise ValueError("Camera source key must be a string")
    short = name.removeprefix("observation.images.")
    result = "observation.images." + _ALIASES.get(short, short)
    if result not in CAMERA_KEYS:
        raise ValueError(f"Unknown recorded camera source: {name}")
    return result


def normalize_collector_camera_bundle(packet, received_monotonic, contract, *, endpoint):
    """Adapt the audited local ``compose_camera_streams.py`` wire, without shifting clocks.

    This explicitly selected producer binds loopback and records each upstream
    receive time on the *same workstation* as the policy client. Remote sockets
    and forwarded ports cannot supply this same-clock contract. Original camera
    wall/capture timestamps are audit fields only. JPEG payloads are untouched.
    """
    location = urlparse(endpoint)
    local = location.scheme == "ipc" and bool(location.path)
    if location.scheme == "tcp":
        try:
            local = location.hostname == "localhost" or ipaddress.ip_address(location.hostname).is_loopback
        except (ValueError, TypeError):
            local = False
    if not local:
        raise ValueError("Collector composer requires a direct same-workstation loopback/IPC endpoint")
    bound = validate_observation_contract(contract)
    received = float(received_monotonic)
    if not math.isfinite(received) or received < 0:
        raise ValueError("Collector camera receipt must be finite client monotonic time")
    if not isinstance(packet, Mapping) or not isinstance(packet.get("sync"), Mapping):
        raise ValueError("Collector composer requires its original images/sources/sync payload")
    sync = packet["sync"]
    if (sync.get("method") != "workstation_receive_time" or sync.get("hardware_synchronized") is not False
            or validate_observation_contract(sync.get("contract")) != bound):
        raise ValueError("Collector composer timing contract differs from checkpoint")
    images, sources = packet.get("images"), packet.get("sources")
    if not isinstance(images, Mapping) or not isinstance(sources, Mapping):
        raise ValueError("Collector composer needs images and each camera's source receipt")
    frames, timings, identities = {}, {}, {}
    for source, image in images.items():
        name = canonical_camera_name(source)
        if name in frames:
            raise ValueError("Collector composer has duplicate camera aliases")
        info = sources.get(source)
        if not isinstance(info, Mapping) or "received_monotonic" not in info:
            raise ValueError("Collector composer requires genuine per-camera receive times")
        stamp = float(info["received_monotonic"])
        max_age = bound["max_age_ms"]
        if (not math.isfinite(stamp) or received-stamp < -_EPS
                or (max_age is not None and received-stamp > max_age/1000+_EPS)):
            raise ValueError("Collector camera receipt is stale/future on the client clock")
        identity = info.get("frame_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Collector composer needs the original source frame identity")
        frames[name], identities[name] = image, identity
        timings[name] = {**info, "received_monotonic": stamp}
    if set(frames) != set(CAMERA_KEYS):
        raise ValueError("Collector composer requires ego_view and both wrist cameras")
    stamps = [timings[name]["received_monotonic"] for name in CAMERA_KEYS]
    if bound["version"] == 1 and max(stamps)-min(stamps) > bound["camera_max_skew_ms"]/1000+_EPS:
        raise ValueError("Collector camera receipt spread exceeds checkpoint contract")
    anchor, emitted = float(sync.get("anchor_monotonic", math.nan)), float(sync.get("emitted_monotonic", math.nan))
    if (not math.isfinite(anchor) or abs(anchor-stamps[0]) > _EPS or not math.isfinite(emitted)
            or emitted < max(stamps)-_EPS or emitted > received+_EPS):
        raise ValueError("Collector composer anchor/emission timestamps are inconsistent")
    return dict(images=frames, frame_ids=identities, timestamps=packet.get("timestamps", {}),
                timing=dict(receipt_clock="client_monotonic", camera_anchor_time=anchor,
                    camera_pair_sequence=packet.get("sequence"), composer_session=packet.get("composer_session"),
                    source_format=COMPOSER_FORMAT, source_clock="same_workstation_monotonic",
                    cameras=timings))


@dataclass
class _Frame:
    received: float
    serial: int
    source_key: str
    identity: str | int | float
    original_timestamp: float | None
    image: object
    handled: bool = False


class CameraSnapshotAssembler:
    """Pair newest ego with the nearest feasible left/right receipt pair.

    Valid triples have receipt spread <=25ms and every receipt <=decision and
    age <=200ms, using the actual configured collector limits. Among feasible
    wrist pairs choose minimum summed distance to ego, then spread, then older
    receipt/serial. Repeated source identities retain their first receipt.
    Missing pairs are retried without blocking until ego receipt + camera_wait.
    A late poll can still emit a pair that was received before that deadline.
    """

    def __init__(self, contract):
        self.contract = validate_observation_contract(contract)
        self.histories = {name: deque(maxlen=self.contract["history_size"]) for name in CAMERA_KEYS}
        self.serial = self.pair_sequence = 0

    def ingest(self, source_key, packet, received_monotonic):
        name = canonical_camera_name(source_key)
        received = float(received_monotonic)
        if not math.isfinite(received) or received < 0:
            raise ValueError("Camera receipt must be finite client monotonic time")
        if not isinstance(packet, Mapping) or not isinstance(packet.get("images"), Mapping):
            raise ValueError("Camera requires the SONIC timestamps/images packet")
        aliases = {name, name.removeprefix("observation.images.")}
        aliases.update(key for key in _ALIASES if "observation.images."+_ALIASES[key] == name)
        aliases.update("observation.images."+key for key in tuple(aliases) if not key.startswith("observation.images."))
        matches = [(key, image) for key, image in packet["images"].items() if key in aliases]
        if len(matches) != 1:
            raise ValueError("Independent source must identify exactly one expected camera")
        key, image = matches[0]
        if isinstance(image, (str, bytes, bytearray)):
            if not image:
                raise ValueError("Camera encoded image cannot be empty")
        elif not isinstance(image, np.ndarray) or image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError("Camera image must be encoded or RGB uint8 [480,640,3]")
        timestamps = packet.get("timestamps", {})
        if not isinstance(timestamps, Mapping):
            raise ValueError("Camera timestamps must be a mapping")
        original = timestamps.get(key)
        if original is not None:
            original = float(original)
            if not math.isfinite(original):
                raise ValueError("Source camera timestamp must be finite")
        identities = packet.get("frame_ids", {})
        if not isinstance(identities, Mapping):
            raise ValueError("Camera frame_ids must be a mapping")
        identity = identities.get(key, packet.get("sequence", original))
        if isinstance(identity, bool) or not isinstance(identity, (str, int, float)):
            raise ValueError("Source needs a frame identity or source timestamp")
        if isinstance(identity, float) and not math.isfinite(identity):
            raise ValueError("Camera frame identity must be finite")
        session = packet.get("source_session")
        if session is not None:
            if not isinstance(session, str) or not session:
                raise ValueError("Camera source session must be a nonempty string")
            identity = json.dumps([session, type(identity).__name__, identity], separators=(",", ":"))
        history = self.histories[name]
        if history and received < history[-1].received:
            raise ValueError("Client camera receipt clock regressed")
        if any(frame.identity == identity for frame in history):
            return False
        self.serial += 1
        history.append(_Frame(received, self.serial, key, identity, original, image))
        return True

    def poll_bundle(self, decision):
        decision = float(decision)
        if not math.isfinite(decision) or decision < 0:
            raise ValueError("Camera decision must be finite client monotonic time")
        if self.contract["version"] in (2, 3):
            selected = []
            for camera in CAMERA_KEYS:
                eligible = [frame for frame in self.histories[camera] if frame.received <= decision + _EPS]
                if not eligible:
                    return None
                selected.append(eligible[-1])
            bound = self.contract["max_age_ms"]
            if bound is not None and any(decision-frame.received > bound/1000+_EPS for frame in selected):
                return None
            if selected[0].handled:
                return None
            selected[0].handled = True
            return self._bundle(dict(zip(CAMERA_KEYS, selected)), "latest_available_v1")
        age = self.contract["max_age_ms"] / 1000
        skew = self.contract["camera_max_skew_ms"] / 1000
        wait = self.contract["camera_wait_ms"] / 1000
        for ego in reversed(self.histories[CAMERA_KEYS[0]]):
            if ego.handled or ego.received > decision + _EPS:
                continue
            if decision - ego.received > age + _EPS:
                ego.handled = True
                continue
            deadline = ego.received + wait
            candidates = []
            for camera in CAMERA_KEYS[1:]:
                candidates.append([frame for frame in self.histories[camera]
                    if frame.received <= min(decision, deadline) + _EPS
                    and decision - frame.received <= age + _EPS
                    and abs(frame.received - ego.received) <= skew + _EPS])
            best = None
            right_times = [frame.received for frame in candidates[1]]
            for left in candidates[0]:
                start = bisect_left(right_times, left.received-skew-_EPS)
                stop = bisect_right(right_times, left.received+skew+_EPS)
                middle = bisect_left(right_times, ego.received, lo=start, hi=stop)
                # For a fixed left frame only nearest right frames can minimize
                # distance; check both sides for the spread/older-time tie break.
                nearest = {bisect_left(right_times, right_times[index], lo=start, hi=stop)
                           for index in (middle-1, middle) if start <= index < stop}
                for index in nearest:
                    right = candidates[1][index]
                    times = (ego.received, left.received, right.received)
                    span = max(times) - min(times)
                    if span > skew + _EPS:
                        continue
                    key = (abs(left.received - ego.received) + abs(right.received - ego.received),
                           span, left.received, right.received, left.serial, right.serial)
                    if best is None or key < best[0]:
                        best = key, (ego, left, right)
            if best is None:
                if decision >= deadline - _EPS:
                    ego.handled = True
                continue
            for old in self.histories[CAMERA_KEYS[0]]:
                if old.serial <= ego.serial:
                    old.handled = True
            return self._bundle(dict(zip(CAMERA_KEYS, best[1])), "nearest_feasible_ego_anchored_receipt_pair_v1")
        return None

    def _bundle(self, frames, rule):
        self.pair_sequence += 1
        return dict(
            images={camera: frame.image for camera, frame in frames.items()},
            timestamps={camera: frame.original_timestamp for camera, frame in frames.items()},
            frame_ids={camera: frame.identity for camera, frame in frames.items()},
            timing=dict(receipt_clock="client_monotonic", camera_anchor_time=frames[CAMERA_KEYS[0]].received,
                camera_pair_sequence=self.pair_sequence, pairing_rule=rule,
                cameras={camera: dict(received_monotonic=frame.received, source_key=frame.source_key,
                    original_timestamp=frame.original_timestamp, original_clock="unverified_source_clock",
                    frame_identity=frame.identity) for camera, frame in frames.items()}),
        )


class IndependentCameraReceiver:
    """Three independent SUB sockets; each poll has bounded, nonblocking work."""

    def __init__(self, endpoints, contract, *, context=None, clock=time.monotonic,
                 max_messages_per_stream=8):
        if not isinstance(endpoints, Mapping) or len(endpoints) != len(CAMERA_KEYS):
            raise ValueError("Independent cameras require exactly three source endpoints")
        canonical = {canonical_camera_name(key): (key, endpoint) for key, endpoint in endpoints.items()}
        if set(canonical) != set(CAMERA_KEYS):
            raise ValueError("Independent cameras require ego_view, left_wrist and right_wrist")
        if any(not isinstance(endpoint, str) or not endpoint for _, endpoint in canonical.values()):
            raise ValueError("Camera endpoints must be nonempty strings")
        if type(max_messages_per_stream) is not int or max_messages_per_stream < 1:
            raise ValueError("Camera polling budget must be a positive integer")
        self.assembler = CameraSnapshotAssembler(contract)
        self.clock = clock
        self.max_messages_per_stream = max_messages_per_stream
        self._owns_context = context is None
        self.context = zmq.Context() if context is None else context
        self.sockets = {}
        self.closed = False
        try:
            for camera in CAMERA_KEYS:
                source, endpoint = canonical[camera]
                socket = self.context.socket(zmq.SUB)
                self.sockets[camera] = source, socket
                socket.setsockopt(zmq.SUBSCRIBE, b"")
                socket.setsockopt(zmq.RCVHWM, self.assembler.contract["history_size"])
                # Source messages are single-part msgpack. Avoid an old transport
                # backlog on reconnection; our received-frame histories stay 128.
                socket.setsockopt(zmq.CONFLATE, 1)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(endpoint)
        except Exception:
            self.close()
            raise

    def poll(self):
        if self.closed:
            raise RuntimeError("Camera receiver is closed")
        for source, socket in self.sockets.values():
            for _ in range(self.max_messages_per_stream):
                try:
                    raw = socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                received = self.clock()
                packet = msgpack.unpackb(raw, raw=False, object_hook=msgpack_numpy.decode)
                self.assembler.ingest(source, packet, received)
        return self.assembler.poll_bundle(self.clock())

    def close(self):
        self.closed = True
        for _, socket in getattr(self, "sockets", {}).values():
            socket.close(linger=0)
        self.sockets = {}
        if getattr(self, "_owns_context", False) and not self.context.closed:
            self.context.term()
