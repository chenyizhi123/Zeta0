"""Regression coverage for asynchronous trained-prefix RTC control."""

from __future__ import annotations

import copy
import threading

import numpy as np
import pytest

from psi.deploy.predictive_world_realtime import (
    JsonHttpPolicyClient,
    RealtimeActionQueue,
    RealtimeChunkController,
    RealtimeControlConfig,
)


class _ManualClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.value += seconds


class _GatedPredictor:
    def __init__(self, chunks: list[np.ndarray], *, block_call: int | None = None) -> None:
        self.chunks = chunks
        self.block_call = block_call
        self.requests: list[dict] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.Lock()

    def __call__(self, request):
        try:
            with self.lock:
                index = len(self.requests)
                self.requests.append(copy.deepcopy(dict(request)))
            if index + 1 == self.block_call:
                self.started.set()
                if not self.release.wait(timeout=2.0):
                    raise TimeoutError("test predictor was not released")
            metadata = dict(request["realtime"])
            metadata["queue_revision"] = request["committed_prefix"]["queue_revision"]
            return {
                "model_action_chunk": self.chunks[index].tolist(),
                "realtime": metadata,
            }
        finally:
            self.finished.set()


def _config(*, horizon=5, dim=2, delay=2, low_watermark=2):
    return RealtimeControlConfig(
        action_hz=50.0,
        action_horizon=horizon,
        action_dim=dim,
        max_rtc_delay_steps=delay,
        queue_low_watermark=low_watermark,
        initial_delay_steps=delay,
    )


def test_async_rtc_uses_raw_prefix_and_preserves_remaining_prefix() -> None:
    clock = _ManualClock()
    first = np.arange(10, dtype=np.float32).reshape(5, 2)
    second = (100 + np.arange(10, dtype=np.float32)).reshape(5, 2)
    predictor = _GatedPredictor([first, second], block_call=2)
    controller = RealtimeChunkController(
        _config(),
        predictor,
        fallback_action=np.array([-1.0, -1.0], dtype=np.float32),
        postprocess=lambda actions: actions + 1000.0,
        clock=clock,
    )
    try:
        controller.start_session("episode-a")
        controller.prime({"observation": "cold-start"}, anchor_time=0.0)
        assert controller.wait_until_idle()
        assert controller.metrics()["queue_depth"] == 5

        np.testing.assert_array_equal(
            controller.tick({"observation": 0}, anchor_time=0.0), first[0] + 1000.0
        )
        np.testing.assert_array_equal(
            controller.tick({"observation": 1}, anchor_time=0.02), first[1] + 1000.0
        )
        np.testing.assert_array_equal(
            controller.tick({"observation": 2}, anchor_time=0.04), first[2] + 1000.0
        )
        assert predictor.started.wait(timeout=1.0)

        second_request = predictor.requests[1]
        np.testing.assert_array_equal(
            np.asarray(second_request["committed_prefix"]["values"]), first[2:4]
        )
        assert second_request["realtime"]["planned_delay_steps"] == 2
        # Two known actions are sent while this request is in flight.  The
        # reply may replace the plan only after discarding those two rows.
        np.testing.assert_array_equal(
            controller.tick({"observation": 3}, anchor_time=0.06), first[3] + 1000.0
        )
        clock.advance(0.001)
        predictor.release.set()
        assert controller.wait_until_idle()

        np.testing.assert_array_equal(
            controller.tick({"observation": 4}, anchor_time=0.08), second[2] + 1000.0
        )
        metrics = controller.metrics()
        assert metrics["published_chunks"] == 2
        assert metrics.get("fallback_actions", 0) == 0
        assert metrics.get("deadline_misses", 0) == 0
    finally:
        controller.close()


def test_session_reset_drops_a_late_reply() -> None:
    clock = _ManualClock()
    predictor = _GatedPredictor(
        [np.ones((4, 2), dtype=np.float32)], block_call=1
    )
    controller = RealtimeChunkController(
        _config(horizon=4),
        predictor,
        fallback_action=np.zeros(2, dtype=np.float32),
        clock=clock,
    )
    try:
        controller.start_session("old")
        controller.prime({"observation": "old"}, anchor_time=0.0)
        assert predictor.started.wait(timeout=1.0)
        controller.start_session("new")
        predictor.release.set()
        assert predictor.finished.wait(timeout=1.0)
        assert controller.wait_until_idle()
        metrics = controller.metrics()
        assert metrics["session_id"] == "new"
        assert metrics["queue_depth"] == 0
        assert metrics["stale_responses"] == 1
    finally:
        controller.close()


def test_prime_accepts_wall_latency_before_the_first_action_is_released() -> None:
    """Cold-start waiting is not a delayed action-prefix condition."""

    clock = _ManualClock()
    chunk = np.arange(10, dtype=np.float32).reshape(5, 2)
    predictor = _GatedPredictor([chunk], block_call=1)
    controller = RealtimeChunkController(
        _config(),
        predictor,
        fallback_action=np.zeros(2, dtype=np.float32),
        clock=clock,
    )
    try:
        controller.start_session("episode")
        controller.prime({"observation": "cold-start"}, anchor_time=0.0)
        assert predictor.started.wait(timeout=1.0)
        # Deployment is still waiting before motion, even though wall time
        # exceeds the trained RTC prefix budget.
        clock.advance(1.0)
        predictor.release.set()
        assert controller.wait_until_idle()
        metrics = controller.metrics()
        assert metrics["queue_depth"] == 5
        assert metrics["published_chunks"] == 1
        assert metrics.get("deadline_misses", 0) == 0
    finally:
        controller.close()


def test_remote_reset_must_acknowledge_the_local_session() -> None:
    controller = RealtimeChunkController(
        _config(),
        _GatedPredictor([np.ones((5, 2), dtype=np.float32)]),
        fallback_action=np.zeros(2, dtype=np.float32),
        remote_reset=lambda **_kwargs: {"session_id": "wrong", "session_epoch": 1},
    )
    try:
        with pytest.raises(RuntimeError, match="did not acknowledge"):
            controller.start_session("expected")
        with pytest.raises(RuntimeError, match="start_session"):
            controller.tick({"observation": 0}, anchor_time=0.0)
    finally:
        controller.close()


def test_underestimated_delay_rejects_a_temporally_invalid_suffix() -> None:
    clock = _ManualClock()
    first = np.arange(10, dtype=np.float32).reshape(5, 2)
    second = (100 + np.arange(10, dtype=np.float32)).reshape(5, 2)
    predictor = _GatedPredictor([first, second], block_call=2)
    controller = RealtimeChunkController(
        _config(),
        predictor,
        fallback_action=np.array([-1.0, -1.0], dtype=np.float32),
        clock=clock,
    )
    try:
        controller.start_session("episode")
        controller.prime({"observation": "cold-start"}, anchor_time=0.0)
        assert controller.wait_until_idle()
        for step in range(3):
            controller.tick({"observation": step}, anchor_time=step * 0.02)
        assert predictor.started.wait(timeout=1.0)
        # The request reserved two prefix actions but three actions are sent
        # before the delayed reply returns.  Its predicted suffix is unsafe to
        # splice into the queue and must not be published.
        controller.tick({"observation": 3}, anchor_time=0.06)
        controller.tick({"observation": 4}, anchor_time=0.08)
        predictor.release.set()
        assert controller.wait_until_idle()
        metrics = controller.metrics()
        assert metrics["queue_depth"] == 0
        assert metrics["delay_underestimates"] == 1
        assert metrics["deadline_misses"] == 1
        assert metrics.get("published_chunks", 0) == 1
    finally:
        controller.close()


def test_queue_rejects_stale_publish_and_keeps_raw_separate() -> None:
    queue = RealtimeActionQueue(action_horizon=3, action_dim=2)
    raw = np.arange(6, dtype=np.float32).reshape(3, 2)
    executable = raw + 50.0
    assert queue.publish(
        expected_revision=0,
        raw_actions=raw,
        executable_actions=executable,
        discard_steps=1,
    ) == 1
    snapshot = queue.snapshot()
    np.testing.assert_array_equal(snapshot.raw_remaining, raw[1:])
    np.testing.assert_array_equal(queue.pop(), executable[1])
    with pytest.raises(RuntimeError, match="stale action response"):
        queue.publish(
            expected_revision=0,
            raw_actions=raw,
            executable_actions=executable,
            discard_steps=0,
        )


def test_http_policy_client_posts_infer_and_session_reset() -> None:
    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, url, *, json, timeout):
            self.calls.append((url, json, timeout))
            return Response({"ok": True})

    session = Session()
    client = JsonHttpPolicyClient("http://127.0.0.1:10000/", timeout_s=0.4, session=session)
    assert client({"request": 1}) == {"ok": True}
    assert client.reset(session_id="session", session_epoch=3, seed=7) == {"ok": True}
    assert session.calls == [
        ("http://127.0.0.1:10000/infer", {"request": 1}, 0.4),
        (
            "http://127.0.0.1:10000/reset",
            {"session_id": "session", "session_epoch": 3, "seed": 7},
            0.4,
        ),
    ]
