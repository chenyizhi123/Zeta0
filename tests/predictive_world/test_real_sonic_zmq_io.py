import base64
import json
import subprocess
import sys
import textwrap
import time

import cv2
import msgpack
import numpy as np
import pytest
import zmq

from psi.deploy.real_sonic_io import SonicIOConfig, SonicZmqIO, decode_camera_image
from psi.deploy.real_sonic_protocol import (
    CAMERAS,
    FORMAT,
    PROTOCOL,
    AbsoluteActionQueue,
)
from psi.deploy.sonic_wire import HEADER_SIZE


def test_robot_client_imports_without_training_packages_or_hardware():
    code = textwrap.dedent("""\
        import importlib
        import importlib.abc
        import sys
        forbidden = {"torch", "transformers", "diffusers", "pyarrow", "pandas"}
        class RobotEnvironment(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in forbidden:
                    raise ImportError("Unexpected robot training dependency: " + fullname)
        sys.meta_path.insert(0, RobotEnvironment())
        import zmq
        def no_context(*args, **kwargs):
            raise AssertionError("Robot imports must not initialize ZMQ resources")
        zmq.Context = no_context
        for module in (
            "psi.sonic_actions", "psi.deploy.real_sonic_protocol",
            "psi.deploy.real_sonic_client", "psi.deploy.real_sonic_io",
            "psi.deploy.brainco_bridge",
        ):
            importlib.import_module(module)
        assert not any(name.split(".")[0] in forbidden for name in sys.modules)
        """)
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30)


def test_actual_zmq_measured_streams_and_repeated_frame_identity():
    context = zmq.Context()
    pubs, endpoints = [], []
    for _ in range(3):
        pub = context.socket(zmq.PUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        pubs.append(pub)
        endpoints.append(f"tcp://127.0.0.1:{port}")
    io = SonicZmqIO(SonicIOConfig(body_endpoint=endpoints[0], camera_endpoint=endpoints[1],
                                 hand_state_endpoint=endpoints[2]))
    try:
        rgb = np.zeros((480,640,3),np.uint8)
        rgb[..., 0] = 220
        ok, jpeg = cv2.imencode(".jpg", rgb[..., ::-1])
        assert ok
        camera = dict(images={name.split(".")[-1]: jpeg.tobytes() for name in CAMERAS},
                      timestamps={name.split(".")[-1]: 123. for name in CAMERAS})
        hand = dict(format="zeta0.brainco12.state.v1", session_id="hands", sequence=[1,1],
                    positions=[.1]*6+[.9]*6, age_s=[0.,0.])
        deadline = time.monotonic()+2.
        observed = None
        while time.monotonic() < deadline:
            pubs[0].send(b"g1_debug"+msgpack.packb(dict(body_q=[.5]*29),use_bin_type=True))
            pubs[1].send(msgpack.packb(camera,use_bin_type=True))
            pubs[2].send(msgpack.packb(hand,use_bin_type=True))
            time.sleep(.002)
            io.poll(time.monotonic())
            try:
                observed = io.observation(time.monotonic())
                break
            except ValueError:
                pass
        assert observed is not None
        assert np.allclose(observed["state"]["values"][-1], [.5]*29+[.1]*6+[.9]*6)
        component_times = io.history.samples[-1][2]
        assert np.array_equal(component_times, [io.body_sample[0], *(sample[0] for sample in io.hand_samples)])
        assert set(observed["images"]) == set(CAMERAS)
        before = io.images[CAMERAS[0]][-1][0]
        pubs[1].send(msgpack.packb(camera,use_bin_type=True))
        time.sleep(.01)
        io.poll(time.monotonic())
        assert io.images[CAMERAS[0]][-1][0] == before
        decoded = decode_camera_image(observed["images"][CAMERAS[0]]["_source"])
        assert decoded[0,0,0] > 200 and decoded[0,0,2] < 10
    finally:
        io.close()
        for pub in pubs:
            pub.close(linger=0)
        context.term()


def test_legacy_camera_array_semantics_match_existing_sonic_decoder():
    array = np.zeros((480,640,3),np.uint8)
    array[...,0] = 190
    ok, jpeg = cv2.imencode(".jpg", array)
    assert ok
    legacy = base64.b64encode(jpeg).decode()
    assert np.array_equal(decode_camera_image(legacy), cv2.imdecode(jpeg,cv2.IMREAD_COLOR))


def test_stream_selection_precedes_new_tokens_and_start_retries_until_fresh_native_mode_ack(monkeypatch):
    from test_real_sonic_client import Clock
    from psi.deploy import real_sonic_io

    class Capture:
        def __init__(self):
            self.sent = []
        def send(self, packet):
            self.sent.append(packet)

    clock = Clock()
    monkeypatch.setattr(real_sonic_io, "time", clock)
    io = object.__new__(SonicZmqIO)
    io.config = SonicIOConfig(execute=True)
    io.pose, io.hand_commands = Capture(), Capture()
    io.frame_index, io.active = 0, False
    io.body_sample = (clock.now-.01, np.zeros(29))
    io.controller_status = dict(input_control_mode=[0.])
    io.execute(np.full(76, .25))
    assert [packet[:4] for packet in io.pose.sent] == [b"comm", b"pose", b"comm"]
    assert io.pose.sent[0][len(b"command")+HEADER_SIZE:] == bytes([0,0,0])
    assert io.pose.sent[2][len(b"command")+HEADER_SIZE:] == bytes([1,0,0])
    clock.now += .02
    io.execute(np.full(76, .5))  # Only a new slot's token; first reset may have dropped frame zero.
    assert [packet[:4] for packet in io.pose.sent[3:]] == [b"pose", b"comm"]
    clock.now += .02
    io.body_sample = (clock.now-.001, np.zeros(29))
    io.controller_status = dict(input_control_mode=[1.])
    io.execute(np.full(76, .625))
    assert [packet[:4] for packet in io.pose.sent[5:]] == [b"pose"]
    frames = []
    for packet in io.pose.sent:
        if not packet.startswith(b"pose"):
            continue
        header = json.loads(packet[4:4+HEADER_SIZE].rstrip(b"\0"))
        assert header["v"] == 4
        token = np.frombuffer(packet[4+HEADER_SIZE:4+HEADER_SIZE+256], dtype="<f4")
        frame = int(np.frombuffer(packet[4+HEADER_SIZE+256:], dtype="<i8")[0])
        frames.append((frame, float(token[0])))
    assert frames == [(0,.25), (1,.5), (2,.625)]
    io.controller_status = dict(input_control_mode=[0.])
    with pytest.raises(ValueError, match="left token stream mode"):
        io.execute(np.full(76, .25))
    assert len(io.pose.sent) == 6


def test_actual_native_status_without_stream_ack_has_a_bounded_start_deadline():
    io = object.__new__(SonicZmqIO)
    io.stream_selected_at = 1.
    io.body_sample = (1.49, np.zeros(29))
    io.controller_status = dict(input_control_mode=[0.])
    io._check_stream_mode(1.49)
    with pytest.raises(ValueError, match="within 500ms"):
        io._check_stream_mode(1.501)


@pytest.mark.parametrize("recorded_projection", [False,True])
def test_rtc_snapshot_matches_actual_local_zmq_controller_commands(recorded_projection):
    from test_real_sonic_collector_client import contract
    bound = contract(projection=True) if recorded_projection else None
    projection = None if bound is None else bound["action_projection"]
    context = zmq.Context()
    pubs, endpoints = [], []
    for _ in range(3):
        pub = context.socket(zmq.PUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        pubs.append(pub)
        endpoints.append(f"tcp://127.0.0.1:{port}")
    hands = context.socket(zmq.SUB)
    hands.setsockopt(zmq.SUBSCRIBE, b"")
    hand_port = hands.bind_to_random_port("tcp://127.0.0.1")
    poses = context.socket(zmq.SUB)
    poses.setsockopt(zmq.SUBSCRIBE, b"pose")
    io = SonicZmqIO(SonicIOConfig(body_endpoint=endpoints[0], camera_endpoint=endpoints[1],
        hand_state_endpoint=endpoints[2], hand_command_endpoint=f"tcp://127.0.0.1:{hand_port}",
        action_bind="tcp://127.0.0.1:*", execute=True,state_history_count=1 if recorded_projection else 21))
    if bound is not None:
        io.configure_observation_contract(bound)
    poses.connect(io.pose.getsockopt(zmq.LAST_ENDPOINT).decode())
    try:
        queue = AbsoluteActionQueue(rtc_max_prefix_steps=15,action_projection=projection)
        queue.reset("local-test")
        req = dict(format=FORMAT, session_id="local-test", request_seq=0, queue_revision=1,
                   anchor_time=1., action_protocol_id=PROTOCOL, committed_prefix=[])
        raw = np.full((30,76), .03, np.float32)
        raw[:,1] = .6875 if recorded_projection else .8
        raw[:,2] = -.75 if recorded_projection else -.8
        raw[:,64] = -.2
        raw[:,65] = 1.2
        reply = {key: value for key, value in req.items() if key != "committed_prefix"}
        reply["actions"] = raw.tolist()
        queue.install(req, reply, now=1.001)
        _, prefix = queue.snapshot(1.,15)
        action = queue.pop(1.02)
        assert np.array_equal(action, prefix[0])
        pose = hand = None
        deadline = time.monotonic()+2.
        while time.monotonic() < deadline and (pose is None or hand is None):
            io.execute(action)
            time.sleep(.005)
            try:
                pose = poses.recv(zmq.NOBLOCK)
            except zmq.Again:
                pass
            try:
                hand = msgpack.unpackb(hands.recv(zmq.NOBLOCK), raw=False)
            except zmq.Again:
                pass
        assert pose is not None and hand is not None
        token = np.frombuffer(pose[4+HEADER_SIZE:4+HEADER_SIZE+256], dtype="<f4")
        assert np.array_equal(token, prefix[0][:64])
        assert np.array_equal(hand["positions"], prefix[0][64:])
        assert token[:3].tolist() == ([0., .6875, -.75] if recorded_projection else [0., .625, -.625])
        assert hand["positions"][:2] == [0., 1.]
    finally:
        io.close()
        hands.close(linger=0)
        poses.close(linger=0)
        for pub in pubs:
            pub.close(linger=0)
        context.term()


@pytest.mark.parametrize("bundle_format", [None, "chenyz_composer_v1"])
def test_collector_local_zmq_bundle_uses_aliases_and_actual_per_camera_receipt_times(bundle_format):
    from test_real_sonic_collector_client import contract
    context = zmq.Context()
    pubs, endpoints = [], []
    for _ in range(3):
        pub = context.socket(zmq.PUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        pubs.append(pub)
        endpoints.append(f"tcp://127.0.0.1:{port}")
    io = SonicZmqIO(SonicIOConfig(body_endpoint=endpoints[0],camera_endpoint=endpoints[1],
                                hand_state_endpoint=endpoints[2],state_history_count=1,
                                camera_bundle_format=bundle_format))
    io.configure_observation_contract(contract())
    try:
        rgb = np.full((480,640,3),125,np.uint8)
        ok,jpeg = cv2.imencode(".jpg",rgb)
        assert ok
        observed = None
        deadline = time.monotonic()+2.
        while time.monotonic()<deadline:
            stamp = time.monotonic()
            sources = {"ego_view":stamp-.03,"video6":stamp-.018,"video8":stamp-.035}
            packet = dict(images={name:jpeg.tobytes() for name in sources},
                timing=dict(receipt_clock="client_monotonic",
                            cameras={name:dict(received_monotonic=value) for name,value in sources.items()}))
            if bundle_format is not None:
                packet = dict(images=packet["images"], timestamps={name:1700000000. for name in sources},
                    sources={name:dict(received_monotonic=value, frame_id=f"{name}:{stamp}")
                             for name,value in sources.items()}, sequence=1, composer_session="collector",
                    sync=dict(method="workstation_receive_time", hardware_synchronized=False,
                              anchor_monotonic=sources["ego_view"], emitted_monotonic=stamp,
                              contract=contract()["collector"]))
            hand = dict(format="zeta0.brainco12.state.v1",session_id="hands",sequence=[1,1],
                        positions=[.1]*6+[.9]*6,age_s=[.05,.05])
            pubs[0].send(b"g1_debug"+msgpack.packb(dict(body_q=[.5]*29),use_bin_type=True))
            pubs[1].send(msgpack.packb(packet,use_bin_type=True))
            pubs[2].send(msgpack.packb(hand,use_bin_type=True))
            time.sleep(.002)
            io.poll(time.monotonic())
            decision = time.monotonic()
            try:
                io.check_feedback(decision)
                observed = io.observation(decision)
                break
            except ValueError:
                pass
        assert observed is not None
        assert set(observed["images"]) == set(CAMERAS)
        assert np.allclose(observed["state"]["values"],[ [.5]*29+[.1]*6+[.9]*6 ])
        assert (observed["images"][CAMERAS[1]]["time"]-observed["images"][CAMERAS[0]]["time"]) == pytest.approx(.012)
        assert max(observed["snapshot_timing"]["hand_times"]) < -.04
        assert io.body_history.maxlen == 128 and io.camera_bundles.maxlen == 128
        assert not io.supports_hold
    finally:
        io.close()
        for pub in pubs:
            pub.close(linger=0)
        context.term()


@pytest.mark.parametrize("bad_packet", [b"\xc1", msgpack.packb(["not", "a", "camera", "bundle"])])
def test_bad_collector_camera_packet_does_not_interrupt_feedback(bad_packet):
    from test_real_sonic_collector_client import contract
    context = zmq.Context()
    publications = []
    endpoints = []
    for _ in range(3):
        publication = context.socket(zmq.XPUB)
        publication.setsockopt(zmq.LINGER, 0)
        publication.setsockopt(zmq.RCVTIMEO, 2000)
        port = publication.bind_to_random_port("tcp://127.0.0.1")
        publications.append(publication)
        endpoints.append(f"tcp://127.0.0.1:{port}")
    io = SonicZmqIO(SonicIOConfig(body_endpoint=endpoints[0], camera_endpoint=endpoints[1],
                                hand_state_endpoint=endpoints[2], state_history_count=1))
    io.configure_observation_contract(contract())
    try:
        for publication in publications:
            assert publication.recv()[0] == 1
        publications[0].send(b"g1_debug"+msgpack.packb(dict(body_q=[.5]*29)))
        publications[1].send(bad_packet)
        publications[2].send(msgpack.packb(dict(format="zeta0.brainco12.state.v1",
            session_id="hands", sequence=[1,1], positions=[.2]*12, age_s=[0.,0.])))
        deadline = time.monotonic()+2.
        while time.monotonic() < deadline:
            io.poll(time.monotonic())
            if io.camera_bundle_error and io.body_sample and all(sample is not None for sample in io.hand_samples):
                break
        io.check_feedback(time.monotonic())
        assert io.camera_bundle_error is not None
        with pytest.raises(ValueError):
            io.observation(time.monotonic())
    finally:
        io.close()
        for publication in publications:
            publication.close(linger=0)
        context.term()
