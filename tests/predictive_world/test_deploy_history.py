import numpy as np
import pytest

from psi.deploy.predictive_world_history import (
    CommittedActionPrefix,
    PredictiveWorldHistoryBuffer,
    PredictiveWorldRequest,
    QueueRevisionGuard,
    TimedImage,
    TimedState,
)


def _request(*, request_seq=0, reset=False, frequency=50.0, task_revision="task-1"):
    anchor = 10.0
    images = tuple(
        TimedImage("primary", anchor + offset, str(index), b"image")
        for index, offset in enumerate((-0.4, -0.2, 0.0))
    )
    states = tuple(
        TimedState(anchor + offset, index, "state", (float(index), 2.0), (True, True))
        for index, offset in enumerate(np.linspace(-0.4, 0, 21))
    )
    prefix = CommittedActionPrefix(
        queue_revision=3,
        command_times=(10.0, 10.02),
        values=((1.0, 2.0), (3.0, 4.0)),
        valid=((True, True), (True, True)),
        acknowledged_until=10.02,
    )
    return PredictiveWorldRequest(
        session_id="session",
        episode_id="episode",
        request_seq=request_seq,
        observation_anchor_time=anchor,
        clock_domain="robot-monotonic",
        clock_mapping_id="map-1",
        task_text="move",
        task_revision=task_revision,
        images=images,
        states=states,
        action_protocol_id="toy",
        action_frequency=frequency,
        action_grid_start_time=anchor,
        native_action_dim=2,
        committed_prefix=prefix,
        reset=reset,
    )


def test_deploy_builds_21_real_timestamped_states_and_three_images():
    window = PredictiveWorldHistoryBuffer().build(_request(), receive_time=10.05)
    assert len(window.images) == 3 and all(item is not None for item in window.images)
    assert window.state_values.shape == (21, 2)
    assert window.state_time_valid.all()
    assert len(set(window.state_source_sequences.tolist())) == 21
    assert np.all(window.state_actual_times <= window.state_requested_times + 1e-12)


def test_low_frequency_states_are_not_presented_as_21_unique_measurements():
    request = _request()
    sparse = tuple(request.states[index] for index in (0, 10, 20))
    request = PredictiveWorldRequest(**{**request.__dict__, "states": sparse})
    window = PredictiveWorldHistoryBuffer(state_max_age_s=0.21).build(request, receive_time=10.05)
    unique = set(window.state_source_sequences[window.state_time_valid].tolist())
    assert unique == {0, 10, 20}


def test_frequency_stale_sequence_and_task_changes_fail_closed():
    history = PredictiveWorldHistoryBuffer()
    with pytest.raises(ValueError, match="frequency mismatch"):
        history.build(_request(frequency=30), receive_time=10.05)
    history.build(_request(), receive_time=10.05)
    with pytest.raises(ValueError, match="duplicate or out of order"):
        history.build(_request(), receive_time=10.05)
    with pytest.raises(ValueError, match="changed without"):
        history.build(_request(request_seq=1, task_revision="task-2"), receive_time=10.05)
    # Explicit reset starts the new task/session history contract.
    history.build(_request(request_seq=1, task_revision="task-2", reset=True), receive_time=10.05)


def test_queue_revision_rejects_a_late_response():
    guard = QueueRevisionGuard(7)
    guard.authorize_publish(7)
    guard.advance()
    with pytest.raises(RuntimeError, match="stale action response"):
        guard.authorize_publish(7)
