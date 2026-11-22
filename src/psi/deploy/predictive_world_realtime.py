"""Client-side asynchronous control for causal predictive-world action chunks.

The controller deliberately owns no robot SDK.  A hardware adapter calls
``tick`` exactly once per control period and sends the returned action to the
robot.  Inference happens in one background worker while the control thread
continues consuming the local queue.

The queue keeps model-space actions separately from post-processed executable
actions.  This matters for training-time RTC: the model must receive the
unmodified, still-committed tail, while the robot receives the safety-filtered
version.  A session epoch and queue revision make delayed replies fail closed.
"""

from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from psi.deploy.predictive_world_history import CommittedActionPrefix, QueueRevisionGuard


class ActionChunkPredictor(Protocol):
    """Transport adapter returning a JSON-like policy response."""

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RealtimeControlConfig:
    """Timing and native-action contract for :class:`RealtimeChunkController`."""

    action_hz: float
    action_horizon: int
    action_dim: int
    max_rtc_delay_steps: int
    queue_low_watermark: int
    initial_delay_steps: int = 0
    latency_window: int = 100
    require_response_envelope: bool = True

    def __post_init__(self) -> None:
        if self.action_hz <= 0 or not np.isfinite(self.action_hz):
            raise ValueError("action_hz must be finite and positive")
        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        if not 0 <= self.max_rtc_delay_steps <= self.action_horizon:
            raise ValueError("max_rtc_delay_steps must fit the action horizon")
        if not 0 <= self.initial_delay_steps <= self.max_rtc_delay_steps:
            raise ValueError("initial_delay_steps exceeds RTC training support")
        if not self.max_rtc_delay_steps <= self.queue_low_watermark <= self.action_horizon:
            raise ValueError("queue_low_watermark must cover max_rtc_delay_steps")
        if self.latency_window <= 0:
            raise ValueError("latency_window must be positive")

    @property
    def action_period_s(self) -> float:
        return 1.0 / self.action_hz


@dataclass(frozen=True)
class ActionQueueSnapshot:
    """An immutable action-queue view taken when an inference starts."""

    revision: int
    consumed_steps: int
    raw_remaining: np.ndarray


def _coerce_action_chunk(
    value: Any,
    *,
    name: str,
    horizon: int,
    action_dim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (horizon, action_dim):
        raise ValueError(
            f"{name} must have shape {(horizon, action_dim)}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


class RealtimeActionQueue:
    """Thread-safe aligned raw/executable action queues.

    ``publish`` replaces the remaining plan only when the revision captured at
    inference start is still current.  Consuming actions does not advance the
    revision: the reply is expected to account for actions consumed while it
    was being computed via ``discard_steps``.
    """

    def __init__(self, *, action_horizon: int, action_dim: int) -> None:
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self._lock = threading.Lock()
        self._guard = QueueRevisionGuard(0)
        self._raw = np.zeros((0, self.action_dim), dtype=np.float32)
        self._executable = np.zeros((0, self.action_dim), dtype=np.float32)
        self._index = 0
        self._consumed_steps = 0

    def reset(self) -> None:
        with self._lock:
            self._guard = QueueRevisionGuard(0)
            self._raw = np.zeros((0, self.action_dim), dtype=np.float32)
            self._executable = np.zeros((0, self.action_dim), dtype=np.float32)
            self._index = 0
            self._consumed_steps = 0

    @property
    def revision(self) -> int:
        with self._lock:
            return self._guard.revision

    def qsize(self) -> int:
        with self._lock:
            return len(self._executable) - self._index

    def snapshot(self) -> ActionQueueSnapshot:
        with self._lock:
            return ActionQueueSnapshot(
                revision=self._guard.revision,
                consumed_steps=self._consumed_steps,
                raw_remaining=self._raw[self._index :].copy(),
            )

    def consumed_since(self, snapshot: ActionQueueSnapshot) -> int:
        with self._lock:
            return max(0, self._consumed_steps - snapshot.consumed_steps)

    def pop(self) -> np.ndarray | None:
        with self._lock:
            if self._index >= len(self._executable):
                return None
            action = self._executable[self._index].copy()
            self._index += 1
            self._consumed_steps += 1
            return action

    def publish(
        self,
        *,
        expected_revision: int,
        raw_actions: Any,
        executable_actions: Any,
        discard_steps: int,
    ) -> int:
        raw = _coerce_action_chunk(
            raw_actions,
            name="raw_actions",
            horizon=self.action_horizon,
            action_dim=self.action_dim,
        )
        executable = _coerce_action_chunk(
            executable_actions,
            name="executable_actions",
            horizon=self.action_horizon,
            action_dim=self.action_dim,
        )
        if not 0 <= discard_steps <= self.action_horizon:
            raise ValueError("discard_steps must fit the action horizon")
        with self._lock:
            self._guard.authorize_publish(expected_revision)
            self._raw = raw[discard_steps:].copy()
            self._executable = executable[discard_steps:].copy()
            self._index = 0
            return self._guard.advance()


class JsonHttpPolicyClient:
    """Small JSON adapter for the loopback predictive-world endpoint.

    The simulator endpoint is intentionally loopback-only.  A real deployment
    may provide another adapter implementing :class:`ActionChunkPredictor` with
    authenticated streaming transport; this class does not claim to provide
    transport security.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 2.0, session: Any = None) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must include an HTTP scheme")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        if session is None:
            import requests

            session = requests.Session()
        self._session = session

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self._session.post(
            f"{self.base_url}/infer", json=dict(payload), timeout=self.timeout_s
        )
        response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, Mapping):
            raise RuntimeError("policy endpoint returned a non-object response")
        return decoded

    def reset(
        self,
        *,
        session_id: str,
        session_epoch: int,
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "session_epoch": int(session_epoch),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        response = self._session.post(
            f"{self.base_url}/reset", json=payload, timeout=self.timeout_s
        )
        response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, Mapping):
            raise RuntimeError("reset endpoint returned a non-object response")
        return decoded


class RealtimeChunkController:
    """LeRobot-style asynchronous RTC scheduling for one robot control stream.

    A caller invokes ``start_session`` once, then ``tick`` at the exact physical
    control frequency.  The background worker retains only the newest causal
    observation, creates a committed-prefix request from unexecuted raw actions,
    and replaces the future queue only if its session epoch/revision still match.
    ``fallback_action`` is required explicitly: zero is not universally safe for
    every robot/action representation.
    """

    def __init__(
        self,
        config: RealtimeControlConfig,
        predictor: ActionChunkPredictor,
        *,
        fallback_action: Any,
        postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        remote_reset: Callable[..., Mapping[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.predictor = predictor
        self.postprocess = postprocess or (lambda actions: actions.copy())
        self.remote_reset = remote_reset
        self._clock = clock
        fallback = np.asarray(fallback_action, dtype=np.float32)
        if fallback.shape != (config.action_dim,) or not np.isfinite(fallback).all():
            raise ValueError("fallback_action must be finite with shape (action_dim,)")
        self._fallback = fallback.copy()
        self._queue = RealtimeActionQueue(
            action_horizon=config.action_horizon, action_dim=config.action_dim
        )
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="predictive-world-rtc"
        )
        self._session_id: str | None = None
        self._session_epoch = 0
        self._next_request_id = 0
        self._control_steps = 0
        self._latest_observation: tuple[dict[str, Any], float] | None = None
        self._inflight: tuple[int, int, Future[None]] | None = None
        self._latencies_s: deque[float] = deque(maxlen=config.latency_window)
        self._stats: Counter[str] = Counter()

    def start_session(self, session_id: str | None = None, *, seed: int | None = None) -> str:
        """Invalidate local work and optionally synchronize a remote policy reset."""

        selected_session_id = session_id or uuid.uuid4().hex
        if not selected_session_id:
            raise ValueError("session_id must not be empty")
        with self._lock:
            self._session_epoch += 1
            epoch = self._session_epoch
            self._session_id = selected_session_id
            self._queue.reset()
            self._latest_observation = None
            self._inflight = None
            self._control_steps = 0
            self._stats["sessions"] += 1
        if self.remote_reset is not None:
            try:
                reset_response = self.remote_reset(
                    session_id=selected_session_id, session_epoch=epoch, seed=seed
                )
                if not isinstance(reset_response, Mapping):
                    raise RuntimeError("remote reset returned a non-object response")
                if (
                    reset_response.get("session_id") != selected_session_id
                    or type(reset_response.get("session_epoch")) is not int
                    or reset_response["session_epoch"] != epoch
                ):
                    raise RuntimeError("remote reset did not acknowledge the active session")
            except Exception:
                # A local session must never look usable when its matching
                # remote reset failed or was acknowledged by another server.
                with self._lock:
                    if self._session_epoch == epoch:
                        self._session_id = None
                        self._latest_observation = None
                        self._queue.reset()
                raise
        return selected_session_id

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def prime(self, observation_payload: Mapping[str, Any], *, anchor_time: float) -> None:
        """Start the initial inference before allowing the robot to move.

        A normal RTC request is conditioned on actions that are already queued
        for the immediately upcoming control slots.  At cold start that queue
        is empty, so deployment code should call ``prime`` with a measured
        observation, wait for a first chunk, and only then begin ``tick``.
        ``tick`` still has an explicit fallback for transport failures, but a
        fallback is deliberately not silently presented to the model as an RTC
        action prefix.
        """

        payload = self._prepare_observation_payload(observation_payload, anchor_time)
        with self._lock:
            if self._session_id is None:
                raise RuntimeError("call start_session before prime")
            self._latest_observation = (payload, float(anchor_time))
            # Before the first command is released to the robot, wall-clock
            # time spent obtaining the initial plan is not a control delay.
            # Mark this request explicitly so the worker can accept it only
            # while no action has subsequently been consumed.
            self._maybe_schedule_locked(allow_cold_start=True)

    def tick(self, observation_payload: Mapping[str, Any], *, anchor_time: float) -> np.ndarray:
        """Consume one action and asynchronously refresh the future queue.

        ``anchor_time`` is the causal observation/action-grid anchor in the
        payload's clock domain.  The controller never invents it from wall time.
        """

        payload = self._prepare_observation_payload(observation_payload, anchor_time)
        with self._lock:
            if self._session_id is None:
                raise RuntimeError("call start_session before tick")
            self._latest_observation = (payload, float(anchor_time))
            # Schedule before popping: the prefix begins at this observation
            # anchor and must therefore include the action sent by this tick.
            self._maybe_schedule_locked()
            action = self._queue.pop()
            if action is None:
                action = self._fallback.copy()
                self._stats["fallback_actions"] += 1
            else:
                self._stats["executed_actions"] += 1
            self._control_steps += 1
            return action

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            latencies = np.asarray(self._latencies_s, dtype=np.float64)
            return {
                "session_id": self._session_id,
                "session_epoch": self._session_epoch,
                "queue_depth": self._queue.qsize(),
                "queue_revision": self._queue.revision,
                "control_steps": self._control_steps,
                "inflight": self._inflight is not None,
                "latency_count": int(latencies.size),
                "latency_p50_s": float(np.quantile(latencies, 0.5)) if latencies.size else None,
                "latency_p95_s": float(np.quantile(latencies, 0.95)) if latencies.size else None,
                "latency_max_s": float(latencies.max()) if latencies.size else None,
                **dict(self._stats),
            }

    def wait_until_idle(self, timeout_s: float = 1.0) -> bool:
        """Testing/teardown helper; control code should not block on inference."""

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._inflight is None:
                    return True
            time.sleep(0.001)
        with self._lock:
            return self._inflight is None

    def _planned_delay_steps_locked(self) -> int:
        if not self._latencies_s:
            return self.config.initial_delay_steps
        # RTC must reserve a conservative prefix.  Using a p95 here would let
        # a known slower request overwrite actions it was not conditioned on.
        # LeRobot's reference loop uses the measured maximum for this reason.
        estimated = math.ceil(max(self._latencies_s) / self.config.action_period_s)
        return min(self.config.max_rtc_delay_steps, max(self.config.initial_delay_steps, estimated))

    def _maybe_schedule_locked(self, *, allow_cold_start: bool = False) -> None:
        if self._inflight is not None or self._latest_observation is None:
            return
        # This call occurs before the current action is consumed.  Start when
        # that consumption would leave the configured low-watermark depth.
        if self._queue.qsize() > self.config.queue_low_watermark + 1:
            return
        session_id = self._session_id
        assert session_id is not None
        snapshot = self._queue.snapshot()
        payload, anchor_time = self._latest_observation
        requested_delay = self._planned_delay_steps_locked()
        if snapshot.raw_remaining.size and len(snapshot.raw_remaining) < requested_delay:
            # Do not create a shorter prefix merely because the old plan is
            # nearly exhausted: a response that arrives at the expected delay
            # would then be conditioned on too few actions.
            return
        planned_delay = requested_delay if snapshot.raw_remaining.size else 0
        cold_start = (
            allow_cold_start
            and snapshot.raw_remaining.size == 0
            and self._control_steps == 0
        )
        prefix = self._build_prefix(snapshot, anchor_time, planned_delay)
        request_id = self._next_request_id
        self._next_request_id += 1
        request = copy.deepcopy(payload)
        request["committed_prefix"] = {
            "queue_revision": prefix.queue_revision,
            "command_times": list(prefix.command_times),
            "values": [list(row) for row in prefix.values],
            "valid": [list(row) for row in prefix.valid],
            "acknowledged_until": prefix.acknowledged_until,
        }
        request["realtime"] = {
            "session_id": session_id,
            "session_epoch": self._session_epoch,
            "request_id": request_id,
            "control_step": self._control_steps,
            "planned_delay_steps": planned_delay,
        }
        future = self._executor.submit(
            self._run_inference,
            self._session_epoch,
            request_id,
            snapshot,
            planned_delay,
            cold_start,
            self._control_steps,
            request,
        )
        self._inflight = (self._session_epoch, request_id, future)
        self._stats["requests"] += 1

    def _build_prefix(
        self,
        snapshot: ActionQueueSnapshot,
        anchor_time: float,
        count: int,
    ) -> CommittedActionPrefix:
        values = snapshot.raw_remaining[:count]
        return CommittedActionPrefix(
            queue_revision=snapshot.revision,
            command_times=tuple(
                anchor_time + index * self.config.action_period_s for index in range(count)
            ),
            values=tuple(tuple(float(value) for value in row) for row in values),
            valid=tuple((True,) * self.config.action_dim for _ in range(count)),
            acknowledged_until=(
                anchor_time + (count - 1) * self.config.action_period_s
                if count
                else anchor_time
            ),
        )

    def _run_inference(
        self,
        epoch: int,
        request_id: int,
        snapshot: ActionQueueSnapshot,
        planned_delay: int,
        cold_start: bool,
        control_step_at_start: int,
        request: Mapping[str, Any],
    ) -> None:
        started = self._clock()
        response: Mapping[str, Any] | None = None
        error: Exception | None = None
        try:
            response = self.predictor(request)
        except Exception as exc:  # The control loop must remain alive on transport failure.
            error = exc
        elapsed_s = max(0.0, self._clock() - started)
        with self._lock:
            current = self._inflight
            if current is None or current[:2] != (epoch, request_id) or self._session_epoch != epoch:
                self._stats["stale_responses"] += 1
                return
            self._inflight = None
            self._latencies_s.append(elapsed_s)
            if error is not None:
                self._stats["inference_errors"] += 1
                return
            assert response is not None
            try:
                self._validate_response_envelope(
                    response,
                    epoch=epoch,
                    request_id=request_id,
                    queue_revision=snapshot.revision,
                    control_step=control_step_at_start,
                    planned_delay=planned_delay,
                )
                raw_actions = response.get("model_action_chunk", response.get("action_chunk"))
                raw = _coerce_action_chunk(
                    raw_actions,
                    name="policy action chunk",
                    horizon=self.config.action_horizon,
                    action_dim=self.config.action_dim,
                )
                executable = _coerce_action_chunk(
                    self.postprocess(raw.copy()),
                    name="postprocessed action chunk",
                    horizon=self.config.action_horizon,
                    action_dim=self.config.action_dim,
                )
                if cold_start and self._control_steps == 0:
                    # ``prime`` blocks deployment before the first motion, so
                    # no command slots elapsed in the robot/action clock.
                    # If a caller starts ticking before this reply arrives,
                    # this condition becomes false and the normal fail-closed
                    # wall/control delay calculation applies.
                    actual_delay = 0
                else:
                    elapsed_steps = math.ceil(elapsed_s / self.config.action_period_s)
                    consumed_steps = self._control_steps - control_step_at_start
                    actual_delay = max(elapsed_steps, consumed_steps)
                if actual_delay > planned_delay:
                    self._stats["delay_underestimates"] += 1
                    # The response was conditioned on exactly
                    # ``planned_delay`` committed actions.  A larger actual
                    # delay means the robot has executed actions absent from
                    # that condition, so retaining any suffix would be
                    # temporally invalid even if it fits the training bound.
                    self._stats["deadline_misses"] += 1
                    return
                if actual_delay > self.config.max_rtc_delay_steps:
                    self._stats["deadline_misses"] += 1
                    return
                self._queue.publish(
                    expected_revision=snapshot.revision,
                    raw_actions=raw,
                    executable_actions=executable,
                    discard_steps=actual_delay,
                )
                self._stats["published_chunks"] += 1
            except (KeyError, TypeError, ValueError, RuntimeError):
                self._stats["rejected_responses"] += 1

    @staticmethod
    def _prepare_observation_payload(
        observation_payload: Mapping[str, Any], anchor_time: float
    ) -> dict[str, Any]:
        if not np.isfinite(anchor_time):
            raise ValueError("anchor_time must be finite")
        payload = copy.deepcopy(dict(observation_payload))
        if "committed_prefix" in payload or "realtime" in payload:
            raise ValueError("caller payload must not pre-populate realtime reserved fields")
        return payload

    def _validate_response_envelope(
        self,
        response: Mapping[str, Any],
        *,
        epoch: int,
        request_id: int,
        queue_revision: int,
        control_step: int,
        planned_delay: int,
    ) -> None:
        metadata = response.get("realtime")
        if metadata is None and not self.config.require_response_envelope:
            return
        if not isinstance(metadata, Mapping):
            raise ValueError("policy response lacks realtime metadata")
        if metadata.get("session_id") != self._session_id:
            raise ValueError("policy response session_id is stale")
        expected = {
            "session_epoch": epoch,
            "request_id": request_id,
            "queue_revision": queue_revision,
            "control_step": control_step,
            "planned_delay_steps": planned_delay,
        }
        for key, value in expected.items():
            if type(metadata.get(key)) is not int or metadata[key] != value:
                raise ValueError(f"policy response {key} is stale or malformed")


__all__ = [
    "ActionChunkPredictor",
    "ActionQueueSnapshot",
    "JsonHttpPolicyClient",
    "RealtimeActionQueue",
    "RealtimeChunkController",
    "RealtimeControlConfig",
]
