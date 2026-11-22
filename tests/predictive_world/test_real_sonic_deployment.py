import base64
import json
import threading
from copy import deepcopy
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pytest

from psi.deploy.brainco_bridge import limited_target, normalized_motor_positions
from psi.deploy.real_sonic_client import HttpClient, check_health
from psi.deploy.real_sonic_protocol import (
    CAMERAS,
    FORMAT,
    PERIOD,
    PROTOCOL,
    STATE_SCHEMA,
    AbsoluteActionQueue,
    MeasuredHistory,
    validate_request,
)
from psi.deploy.real_sonic_server import InferenceService, make_handler


def test_native_30hz_service_advertises_and_returns_its_actual_action_grid():
    service = InferenceService(lambda data, seed: np.zeros((50, 76), np.float32),
        dict(task='clean table', source_completed_updates=10000, weights_sha256='test',
             temporal_abi=dict(action_hz=30, action_horizon=50, state_history_count=21)))
    health = service.health()
    assert (health['action_hz'], health['action_horizon']) == (30, 50)
    # The existing physical 50Hz queue/client must reject this different grid.
    with pytest.raises(ValueError, match='action_hz'):
        check_health(health)
    req = request()
    service.reset({'session_id': req['session_id']})
    response = service.predict(req)
    assert np.asarray(response['actions']).shape == (50, 76)
    assert response['action_offsets_s'] == [(i+1)/30 for i in range(50)]


@pytest.mark.parametrize('hz,horizon', [(30, 30), (50, 50), (float('nan'), 50), (True, 30), (30, 50.)])
def test_unsupported_action_grids_fail_instead_of_being_relabelled(hz, horizon):
    from psi.deploy.real_sonic_protocol import validate_action_grid
    with pytest.raises(ValueError, match='action grid'):
        validate_action_grid(hz, horizon)
from psi.deploy.sonic_wire import HEADER_SIZE, build_command_message, pack_pose_message
from psi.sonic_actions import controller_command_values


def request(anchor=1., revision=1, prefix=None):
    return dict(format=FORMAT, session_id="test", request_seq=0, queue_revision=revision,
                anchor_time=anchor, action_protocol_id=PROTOCOL, state_schema_id=STATE_SCHEMA,
                state=dict(values=np.zeros((21,41)).tolist(), valid=np.ones((21,41), bool).tolist(),
                           times=(np.arange(-20,1)*PERIOD).tolist()),
                images={name: dict(time=0., encoded=base64.b64encode(b"image").decode()) for name in CAMERAS},
                committed_prefix=[] if prefix is None else prefix)


def response(req, actions=None):
    data = {key: req[key] for key in ("format", "session_id", "request_seq", "queue_revision", "anchor_time", "action_protocol_id")}
    data["actions"] = np.zeros((30,76)).tolist() if actions is None else actions.tolist()
    return data


def test_measured_history_uses_causal_floor_and_masks_interruption():
    history = MeasuredHistory()
    history.append(.6, np.ones(41))
    history.append(.635, np.full(41, 2))
    history.append(1., np.full(41, 3))
    data = history.aligned(1.)
    assert data["values"][0] == [1.]*41
    assert data["times"][2] == pytest.approx(-.365)
    assert data["values"][2] == [2.]*41
    assert not any(data["valid"][10])
    assert all(data["valid"][-1])
    assert data["values"][-1] == [3.]*41
    with pytest.raises(ValueError, match="regressed"):
        history.append(.99, np.zeros(41))


def test_selected_history_masks_each_component_at_its_own_query_time():
    history = MeasuredHistory()
    history.append(.95, np.ones(41), component_times=(.92,.95,.925))
    history.append(.999, np.full(41,2), component_times=(.979,.999,.995))
    data = history.aligned(1.)
    # The -20ms query selects the old snapshot, whose combined age is 30ms.
    # Its body and right hand have actually aged 60ms and 55ms respectively.
    assert data["times"][-2] == pytest.approx(-.05)
    assert data["valid"][-2] == [False]*29+[True]*6+[False]*6
    assert data["values"][-2] == [1.]*41
    assert data["valid"][-1] == [True]*41
    assert data["values"][-1] == [2.]*41


@pytest.mark.parametrize("components", [(1.,1.), (1.,1.,float("nan")), (1.,1.,1.001)])
def test_history_component_times_must_match_the_combined_receive_clock(components):
    with pytest.raises(ValueError, match="component times"):
        MeasuredHistory().append(1., np.zeros(41), component_times=components)


@pytest.mark.parametrize("change", ["future_state", "old_state", "future_camera", "old_camera", "missing_camera", "wrong_dim", "long_prefix", "nan"])
def test_wire_rejects_semantic_mismatch(change):
    req = request()
    validate_request(req)
    if change == "future_state": req["state"]["times"][-1] = .001
    elif change == "old_state": req["state"]["times"][-1] = -.041
    elif change == "future_camera": req["images"][CAMERAS[0]]["time"] = .001
    elif change == "old_camera": req["images"][CAMERAS[0]]["time"] = -.081
    elif change == "missing_camera": del req["images"][CAMERAS[-1]]
    elif change == "wrong_dim": req["state"]["values"][0].pop()
    elif change == "long_prefix": req["committed_prefix"] = np.zeros((7,76)).tolist()
    elif change == "nan": req["state"]["values"][-1][0] = float("nan")
    with pytest.raises(ValueError): validate_request(req)


def test_partial_arrival_never_restarts_action_horizon():
    queue = AbsoluteActionQueue()
    queue.reset("test")
    req = request()
    acts = np.full((30,76), .25, np.float32)
    acts[:,64:] = np.arange(30, dtype=np.float32)[:,None] / 32
    assert queue.install(req, response(req, acts), now=1.205) == 20
    assert queue.pop(1.2) is None
    assert np.array_equal(queue.pop(1.22), [.25]*64+[10/32]*12)
    assert np.array_equal(queue.pop(1.6), [.25]*64+[29/32]*12)
    assert queue.pop(1.62) is None


def test_expired_session_and_queue_replies_are_rejected():
    queue = AbsoluteActionQueue()
    queue.reset("test")
    req = request()
    with pytest.raises(ValueError, match="expired"):
        queue.install(req, response(req), now=1.7)
    queue.install(req, response(req), now=1.001)
    with pytest.raises(ValueError, match="revision"):
        queue.install(req, response(req), now=1.002)
    queue.reset("new")
    with pytest.raises(ValueError, match="stale"):
        queue.install(req, response(req), now=1.003)


def test_rtc15_requires_trained_support_and_keeps_a_300ms_reservation():
    service = InferenceService(lambda data, seed: np.zeros((30,76), np.float32),
        dict(task="doll", source_completed_updates=1, weights_sha256="test"),
        rtc_max_prefix_steps=15)
    assert check_health(service.health(), rtc_prefix_steps=15) == 15
    legacy = InferenceService(service.infer, service.manifest)
    with pytest.raises(ValueError, match="checkpoint support"):
        check_health(legacy.health(), rtc_prefix_steps=15)
    prefix = np.full((15,76), .25, np.float32).tolist()
    req = request(prefix=prefix)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_request(req)
    validate_request(req, rtc_max_prefix_steps=15)
    queue = AbsoluteActionQueue(rtc_max_prefix_steps=15)
    queue.reset("test")
    initial = request()
    queue.install(initial, response(initial, np.full((30,76), .25, np.float32)), now=1.001)
    revision, actual = queue.snapshot(1., 15)
    assert actual == prefix
    req = request(revision=revision, prefix=prefix)
    new = np.full((30,76), .5, np.float32)
    new[:15] = prefix
    queue.install(req, response(req,new), now=1.281)
    assert np.all(queue.pop(1.3) == .25)
    assert np.all(queue.pop(1.32) == .5)
    req["queue_revision"] = queue.revision
    with pytest.raises(ValueError, match="reserved prefix"):
        queue.install(req, response(req,new), now=1.301)
    with pytest.raises(ValueError, match="0..6"):
        AbsoluteActionQueue().snapshot(1.,15)


def test_runtime_rtc_capacity_is_bound_to_source_config_and_checkpoint(tmp_path):
    from psi.data.predictive_world.cache import canonical_sha256, file_sha256
    from psi.deploy.real_sonic_runtime import ResearchInferenceRuntime
    config = dict(training=dict(rtc_training_max_delay=15))
    source = dict(metadata=dict(resolved_config_sha256=canonical_sha256(config),
        optimization_contract=dict(rtc_training_max_delay=15)))
    config_path = tmp_path / "original_run_config.json"
    source_path = tmp_path / "source_checkpoint.json"
    config_path.write_text(json.dumps(config))
    source_path.write_text(json.dumps(source))
    manifest = dict(format="zeta0.research_inference_weights.v1", action_protocol_id=PROTOCOL,
        state_schema_id=STATE_SCHEMA, source_checkpoint_sha256=file_sha256(source_path),
        temporal_abi=dict(action_hz=50, action_horizon=30, state_query_hz=50, state_history_count=1))
    (tmp_path / "inference_manifest.json").write_text(json.dumps(manifest))
    assert ResearchInferenceRuntime(tmp_path).rtc_max_prefix_steps == 15
    config_path.write_text(json.dumps(dict(training=dict(rtc_training_max_delay=29))))
    with pytest.raises(ValueError, match="config checksum"):
        ResearchInferenceRuntime(tmp_path)
    config_path.write_text(json.dumps(config))
    source["metadata"]["optimization_contract"]["rtc_training_max_delay"] = 29
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="checkpoint checksum"):
        ResearchInferenceRuntime(tmp_path)


def test_rtc_prefix_keeps_reserved_commands_and_rejects_late_reply():
    queue = AbsoluteActionQueue()
    queue.reset("test")
    initial = request()
    old = np.full((30,76), .25, np.float32)
    queue.install(initial, response(initial, old), now=1.001)
    assert np.all(queue.pop(1.02) == .25)
    revision, prefix = queue.snapshot(1.02, 6)
    assert len(prefix) == 6
    req = request(1.02, revision, prefix)
    new = np.full((30,76), .5, np.float32)
    new[:6] = prefix
    queue.install(req, response(req,new), now=1.05)
    assert np.all(queue.pop(1.06) == .25)
    assert np.all(queue.pop(1.14) == .25)
    assert np.all(queue.pop(1.16) == .5)
    late = deepcopy(req)
    late["queue_revision"] = queue.revision
    with pytest.raises(ValueError, match="RTC"):
        queue.install(late, response(late,new), now=1.141)


@pytest.mark.parametrize("prefix_steps", [6, 15])
def test_rtc_prefix_is_the_committed_controller_command_not_raw_model_output(prefix_steps):
    queue = AbsoluteActionQueue(rtc_max_prefix_steps=prefix_steps)
    queue.reset("test")
    req = request()
    raw = np.full((30,76), .03, np.float32)
    raw[:,1] = .8
    raw[:,2] = -.8
    raw[:,64] = -.2
    raw[:,65] = 1.2
    expected = np.array([0., .625, -.625]+[0.]*61+[0., 1.]+[.03]*10, np.float32)
    queue.install(req, response(req, raw), now=1.001)
    revision, prefix = queue.snapshot(1., prefix_steps)
    assert np.array_equal(prefix, np.repeat(expected[None], prefix_steps, axis=0))
    req = request(revision=revision, prefix=prefix)
    new = np.full((30,76), .5, np.float32)
    new[:prefix_steps] = prefix
    changed = new.copy()
    changed[0,0] = .01  # This also maps to zero; the raw prefix guard must catch it.
    with pytest.raises(ValueError, match="changed committed actions"):
        queue.install(req, response(req, changed), now=1.002)
    queue.install(req, response(req, new), now=1.002)
    assert np.array_equal(queue.pop(1.02), expected)
    assert np.all(queue.pop(1.+(prefix_steps+1)*PERIOD) == .5)


def test_controller_command_mapping_is_copying_idempotent_and_preserves_sdk_targets():
    values = np.array([.03]*64+[.123]*12, np.float32)
    original = values.copy()
    command = controller_command_values(values)
    assert np.array_equal(values, original)
    assert np.array_equal(command[:64], np.zeros(64, np.float32))
    assert np.array_equal(command[64:], original[64:])
    assert np.array_equal(controller_command_values(command), command)
    with pytest.raises(ValueError, match="finite"):
        controller_command_values(np.full(76, float("nan")))
    with pytest.raises(ValueError, match="shape"):
        controller_command_values(np.zeros(75))


def test_http_round_trip_session_sequence_and_native_width():
    manifest = dict(task="Place the doll on the table.", source_completed_updates=10000, weights_sha256="test")
    service = InferenceService(lambda data, seed: np.full((30,76), seed, np.float32), manifest, seed=7)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HttpClient(f"http://127.0.0.1:{server.server_port}")
        check_health(client.call("/health"))
        client.call("/reset", {"session_id": "test"})
        req = request()
        result = client.call("/infer", req)
        assert np.array(result["actions"]).shape == (30,76)
        assert result["actions"][0][0] == 7
        assert result["action_offsets_s"][0] == .02
        assert result["action_offsets_s"][-1] == .6
        with pytest.raises(HTTPError) as exc: client.call("/infer", req)
        assert exc.value.code == 400
        client.call("/reset", {"session_id": "new"})
        req["request_seq"] = 1
        with pytest.raises(HTTPError): client.call("/infer", req)
    finally:
        server.shutdown()
        thread.join(2)
        server.server_close()


@pytest.mark.parametrize("actions", [np.zeros((29, 76)), np.full((30, 76), np.nan)])
def test_service_rejects_invalid_native_outputs_before_serializing(actions):
    service = InferenceService(lambda data, seed: actions,
        dict(task="doll", source_completed_updates=20000, weights_sha256="test"))
    service.reset(dict(session_id="test"))
    with pytest.raises(ValueError, match="finite native actions"):
        service.predict(request())


def test_sonic_binary_protocol_is_motion64_without_synthetic_seven_joint_hands():
    token = np.arange(64, dtype=np.float32).reshape(1,64)
    data = pack_pose_message({"token_state": token, "frame_index": np.array([3],np.int64)}, version=4)
    assert HEADER_SIZE == 1280
    header = json.loads(data[4:4+HEADER_SIZE].rstrip(b"\0"))
    assert header["v"] == 4
    assert header["fields"][0]["shape"] == [1,64]
    assert [field["name"] for field in header["fields"]] == ["token_state", "frame_index"]
    assert np.array_equal(np.frombuffer(data[4+HEADER_SIZE:4+HEADER_SIZE+256],dtype="<f4"), token[0])
    stop = build_command_message(False, True, False)
    assert stop[-3:] == b"\x00\x01\x00"


def test_brainco_uses_measured_normalized_native_six_coordinates():
    status = SimpleNamespace(positions=[0,100,200,500,900,1000])
    assert np.allclose(normalized_motor_positions(status), [0,.1,.2,.5,.9,1])
    assert np.allclose(limited_target(np.ones(6), np.zeros(6), .02), .06)
    with pytest.raises(ValueError): normalized_motor_positions(SimpleNamespace(positions=[2000]*6))
