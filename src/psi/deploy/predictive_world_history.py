"""Fail-closed causal history and queue contract for predictive-world serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from psi.data.predictive_world.alignment import causal_floor_indices


@dataclass(frozen=True)
class TimedImage:
    camera_id: str
    capture_time: float
    frame_id: str
    encoded_image: bytes
    # The simulator HTTP bridge has already decoded its JSON/base64 RGB frame.
    # Retaining it here avoids an otherwise lossless PNG encode/decode before
    # the Qwen preprocessor.  Network-facing producers continue using the
    # encoded form; raw pixels are an internal, optional fast path only.
    raw_rgb: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class TimedState:
    capture_time: float
    source_seq: int
    state_schema_id: str
    values: tuple[float, ...]
    valid: tuple[bool, ...]


@dataclass(frozen=True)
class CommittedActionPrefix:
    queue_revision: int
    command_times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    valid: tuple[tuple[bool, ...], ...]
    acknowledged_until: float

    def validate(self, *, anchor_time: float, action_hz: float, native_dim: int) -> None:
        count = len(self.command_times)
        if len(self.values) != count or len(self.valid) != count:
            raise ValueError("committed prefix arrays have different lengths")
        if self.queue_revision < 0 or not np.isfinite(self.acknowledged_until):
            raise ValueError("committed prefix queue metadata is invalid")
        expected_step = 1.0 / action_hz
        for index, (time, values, valid) in enumerate(zip(self.command_times, self.values, self.valid)):
            if len(values) != native_dim or len(valid) != native_dim or not all(valid):
                raise ValueError("every committed prefix step must contain all native dimensions")
            if not np.isfinite(time) or not np.isfinite(values).all():
                raise ValueError("committed prefix values/times must be finite")
            expected = anchor_time + index * expected_step
            if abs(time - expected) > 1e-6:
                raise ValueError("committed prefix command grid is not left-contiguous")


@dataclass(frozen=True)
class PredictiveWorldRequest:
    session_id: str
    episode_id: str
    request_seq: int
    observation_anchor_time: float
    clock_domain: str
    clock_mapping_id: str
    task_text: str
    task_revision: str
    images: tuple[TimedImage, ...]
    states: tuple[TimedState, ...]
    action_protocol_id: str
    action_frequency: float
    action_grid_start_time: float
    native_action_dim: int
    committed_prefix: CommittedActionPrefix
    reset: bool = False


@dataclass(frozen=True)
class CausalHistoryWindow:
    images: tuple[TimedImage | None, ...]
    image_requested_times: tuple[float, ...]
    state_values: np.ndarray
    state_dimension_valid: np.ndarray
    state_requested_times: np.ndarray
    state_actual_times: np.ndarray
    state_source_sequences: np.ndarray
    state_time_valid: np.ndarray
    state_schema_id: str
    committed_prefix: CommittedActionPrefix


@dataclass
class _SessionState:
    episode_id: str
    task_revision: str
    clock_domain: str
    clock_mapping_id: str
    last_request_seq: int = -1
    last_anchor_time: float = float("-inf")


class PredictiveWorldHistoryBuffer:
    """Validate history carried by a request; never invent high-rate samples."""

    def __init__(
        self,
        *,
        action_hz: float = 50.0,
        visual_offsets_s: Sequence[float] = (-0.4, -0.2, 0.0),
        state_count: int = 21,
        state_hz: float = 50.0,
        visual_max_age_s: float = 0.08,
        state_max_age_s: float = 0.04,
        max_observation_age_s: float = 0.12,
        observed_visual_mode: str = "history",
    ) -> None:
        if min(action_hz, state_count, state_hz, visual_max_age_s, state_max_age_s) <= 0:
            raise ValueError("history frequencies/count/ages must be positive")
        self.action_hz = float(action_hz)
        if observed_visual_mode not in {"history", "current_multiview"}:
            raise ValueError("unsupported observed_visual_mode")
        self.observed_visual_mode = observed_visual_mode
        self.visual_offsets_s = (0.0,) if observed_visual_mode == "current_multiview" else tuple(float(value) for value in visual_offsets_s)
        self.state_offsets_s = np.linspace(-(state_count - 1) / state_hz, 0.0, state_count)
        self.visual_max_age_s = float(visual_max_age_s)
        self.state_max_age_s = float(state_max_age_s)
        self.max_observation_age_s = float(max_observation_age_s)
        self._sessions: dict[str, _SessionState] = {}

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def build(self, request: PredictiveWorldRequest, *, receive_time: float) -> CausalHistoryWindow:
        self._validate_request(request, receive_time=receive_time)
        if request.reset:
            self.reset(request.session_id)
        session = self._sessions.get(request.session_id)
        if session is None:
            session = _SessionState(
                episode_id=request.episode_id,
                task_revision=request.task_revision,
                clock_domain=request.clock_domain,
                clock_mapping_id=request.clock_mapping_id,
            )
            self._sessions[request.session_id] = session
        else:
            expected = (
                session.episode_id,
                session.task_revision,
                session.clock_domain,
                session.clock_mapping_id,
            )
            actual = (
                request.episode_id,
                request.task_revision,
                request.clock_domain,
                request.clock_mapping_id,
            )
            if actual != expected:
                raise ValueError("episode/task/clock changed without an explicit reset")
            if request.request_seq <= session.last_request_seq:
                raise ValueError("request sequence is duplicate or out of order")
            if request.observation_anchor_time < session.last_anchor_time:
                raise ValueError("observation time moved backwards")

        images = self._select_images(request)
        state_window = self._select_states(request)
        request.committed_prefix.validate(
            anchor_time=request.action_grid_start_time,
            action_hz=self.action_hz,
            native_dim=request.native_action_dim,
        )
        session.last_request_seq = request.request_seq
        session.last_anchor_time = request.observation_anchor_time
        return CausalHistoryWindow(
            images=images,
            image_requested_times=tuple(
                request.observation_anchor_time + value
                for value in ((0.0,) * len(images) if self.observed_visual_mode == "current_multiview" else self.visual_offsets_s)
            ),
            committed_prefix=request.committed_prefix,
            **state_window,
        )

    def _validate_request(self, request: PredictiveWorldRequest, *, receive_time: float) -> None:
        if not request.session_id or not request.episode_id or request.request_seq < 0:
            raise ValueError("request session/episode/sequence is invalid")
        if not request.clock_domain or not request.clock_mapping_id:
            raise ValueError("request clock identity is required")
        if not np.isfinite(request.observation_anchor_time) or not np.isfinite(receive_time):
            raise ValueError("request/receive times must be finite")
        age = receive_time - request.observation_anchor_time
        if age < -1e-6 or age > self.max_observation_age_s:
            raise ValueError("observation is stale or from the future")
        if abs(request.action_frequency - self.action_hz) > 1e-9:
            raise ValueError(
                f"action frequency mismatch: request={request.action_frequency}, model={self.action_hz}"
            )
        if abs(request.action_grid_start_time - request.observation_anchor_time) > 1e-6:
            raise ValueError("action grid start must match the trained observation anchor contract")
        if request.native_action_dim <= 0 or not request.action_protocol_id:
            raise ValueError("action protocol and native dimension are required")

    def _select_images(self, request: PredictiveWorldRequest) -> tuple[TimedImage | None, ...]:
        if self.observed_visual_mode == "current_multiview":
            selected = []
            for camera in sorted({item.camera_id for item in request.images}):
                frames = sorted(
                    (item for item in request.images if item.camera_id == camera),
                    key=lambda item: item.capture_time,
                )
                times = [item.capture_time for item in frames]
                if len(set(times)) != len(times):
                    raise ValueError("camera timestamps must be unique")
                indices, valid, _ = causal_floor_indices(
                    times, [request.observation_anchor_time],
                    max_age_seconds=self.visual_max_age_s,
                )
                if valid[0]:
                    selected.append(frames[int(indices[0])])
            if not any(item.camera_id == "primary" for item in selected):
                raise ValueError("current primary image is missing or stale")
            return tuple(selected)
        primary = sorted(
            (item for item in request.images if item.camera_id == "primary"),
            key=lambda item: (item.capture_time, item.frame_id),
        )
        if not primary:
            raise ValueError("request contains no primary-camera history")
        times = [item.capture_time for item in primary]
        if len(set(times)) != len(times):
            raise ValueError("primary camera timestamps must be unique")
        queries = [request.observation_anchor_time + value for value in self.visual_offsets_s]
        indices, valid, _ = causal_floor_indices(
            times, queries, max_age_seconds=self.visual_max_age_s
        )
        selected = tuple(primary[index] if present else None for index, present in zip(indices, valid))
        if selected[-1] is None:
            raise ValueError("current primary image is missing or stale")
        return selected

    def _select_states(self, request: PredictiveWorldRequest) -> dict[str, object]:
        by_sequence: dict[int, TimedState] = {}
        for item in request.states:
            previous = by_sequence.get(item.source_seq)
            if previous is not None and previous != item:
                raise ValueError("one state source sequence maps to different payloads")
            by_sequence[item.source_seq] = item
        states = sorted(by_sequence.values(), key=lambda item: (item.capture_time, item.source_seq))
        if not states:
            raise ValueError("request contains no state history")
        schemas = {item.state_schema_id for item in states}
        widths = {len(item.values) for item in states}
        if len(schemas) != 1 or len(widths) != 1:
            raise ValueError("one request cannot mix state schemas or widths")
        width = widths.pop()
        if width <= 0 or any(len(item.valid) != width for item in states):
            raise ValueError("state values and dimension validity disagree")
        times = [item.capture_time for item in states]
        if len(set(times)) != len(times):
            raise ValueError("different state samples cannot share an ambiguous timestamp")
        queries = request.observation_anchor_time + self.state_offsets_s
        indices, time_valid, actual = causal_floor_indices(
            times, queries, max_age_seconds=self.state_max_age_s
        )
        values = np.zeros((len(queries), width), dtype=np.float32)
        dimension_valid = np.zeros((len(queries), width), dtype=bool)
        source_sequences = np.full(len(queries), -1, dtype=np.int64)
        for slot, (index, present) in enumerate(zip(indices, time_valid)):
            if not present:
                continue
            item = states[index]
            row = np.asarray(item.values, dtype=np.float32)
            row_valid = np.asarray(item.valid, dtype=bool)
            if not np.isfinite(row[row_valid]).all():
                raise ValueError("valid state dimensions must be finite")
            values[slot] = np.where(row_valid, row, 0)
            dimension_valid[slot] = row_valid
            source_sequences[slot] = item.source_seq
        if not time_valid[-1]:
            raise ValueError("current state is missing or stale")
        return {
            "state_values": values,
            "state_dimension_valid": dimension_valid,
            "state_requested_times": queries,
            "state_actual_times": actual,
            "state_source_sequences": source_sequences,
            "state_time_valid": time_valid,
            "state_schema_id": next(iter(schemas)),
        }


class QueueRevisionGuard:
    """Prevent a late inference response from overwriting a newer action plan."""

    def __init__(self, revision: int) -> None:
        if revision < 0:
            raise ValueError("queue revision must be non-negative")
        self._revision = int(revision)

    @property
    def revision(self) -> int:
        return self._revision

    def advance(self) -> int:
        self._revision += 1
        return self._revision

    def authorize_publish(self, expected_revision: int) -> None:
        if expected_revision != self._revision:
            raise RuntimeError(
                f"stale action response: expected queue revision {expected_revision}, "
                f"current revision {self._revision}"
            )


__all__ = [
    "CausalHistoryWindow",
    "CommittedActionPrefix",
    "PredictiveWorldHistoryBuffer",
    "PredictiveWorldRequest",
    "QueueRevisionGuard",
    "TimedImage",
    "TimedState",
]
