"""Independent SONIC wire streams paired using true client-clock receipts."""
import base64
import copy
import math
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np
import pytest
import zmq

from psi.deploy.real_sonic_camera import (
    CameraSnapshotAssembler, IndependentCameraReceiver, normalize_collector_camera_bundle,
)
from psi.deploy.real_sonic_io import SonicZmqIO, decode_camera_image
from psi.real_observation_contract import CAMERA_KEYS


CONTRACT = dict(version=1, camera_max_skew_ms=25., camera_wait_ms=20., max_age_ms=200.,
                body_max_offset_ms=40., hand_max_offset_ms=50., history_size=128, image_shape=[480, 640, 3])


def packet(source, identity=0, *, timestamp=1_791_548_000., image=b"deferred-jpeg"):
    return dict(images={source: image}, timestamps={source: timestamp}, frame_ids={source: identity})


def ingest(assembler, source, received, identity=0):
    assembler.ingest(source, packet(source, identity), received)


@pytest.mark.parametrize("version", [2, 3])
def test_latest_camera_contracts_preserve_independent_receipts_without_pairing(version):
    contract = dict(version=version, selection="latest_available_v1", max_age_ms=200.,
        history_size=128, image_shape=[480, 640, 3], cameras=["ego_view", "left_wrist", "right_wrist"],
        timestamp_kind="workstation_source_receive_time")
    if version == 3:
        contract.update(max_age_ms=None, age_policy="record_latest_available")
    assembler = CameraSnapshotAssembler(contract)
    for source, received in (("ego_view", 1.), ("video6", .9), ("video8", .95)):
        ingest(assembler, source, received)
    bundle = assembler.poll_bundle(1.01)
    assert bundle["timing"]["pairing_rule"] == "latest_available_v1"
    assert [bundle["timing"]["cameras"][name]["received_monotonic"] for name in CAMERA_KEYS] == [1., .9, .95]
    assert assembler.poll_bundle(1.01) is None
    ingest(assembler, "ego_view", 1.3, 1)
    assert (assembler.poll_bundle(1.31) is not None) == (version == 3)


def test_preserves_receipt_clock_aliases_and_existing_bundle_decode_boundary():
    assembler = CameraSnapshotAssembler(CONTRACT)
    for source, received in (("ego_view", 1.), ("video6", 1.005), ("video8", 1.009)):
        ingest(assembler, source, received)
    bundle = assembler.poll_bundle(1.01)
    assert bundle["timing"]["receipt_clock"] == "client_monotonic"
    assert bundle["timing"]["camera_anchor_time"] == 1.
    assert list(bundle["images"]) == list(CAMERA_KEYS)
    assert bundle["timing"]["cameras"][CAMERA_KEYS[1]]["source_key"] == "video6"
    assert all(value == 1_791_548_000. for value in bundle["timestamps"].values())
    io = object.__new__(SonicZmqIO)
    frames, times = io.decode_camera_bundle(bundle, 1.01)
    assert list(frames) == list(CAMERA_KEYS)
    assert list(times.values()) == [1., 1.005, 1.009]


def test_repeated_source_frames_do_not_refresh_receipts_or_emit_twice():
    assembler = CameraSnapshotAssembler(CONTRACT)
    ingest(assembler, "ego_view", 1.)
    assert not assembler.ingest("ego_view", packet("ego_view"), 1.15)
    assert assembler.histories[CAMERA_KEYS[0]][-1].received == 1.
    ingest(assembler, "video6", 1.007)
    ingest(assembler, "video8", 1.009)
    assert assembler.poll_bundle(1.16) is not None
    assert assembler.poll_bundle(1.16) is None
    ingest(assembler, "ego_view", 1.17, 1)
    assert not assembler.ingest("video6", packet("video6"), 1.171)
    assert not assembler.ingest("video8", packet("video8"), 1.172)
    assert assembler.poll_bundle(1.175) is None
    assert assembler.histories[CAMERA_KEYS[1]][-1].received == 1.007


def test_source_timestamp_alone_is_an_identity_and_not_a_client_clock():
    assembler = CameraSnapshotAssembler(CONTRACT)
    data = dict(images={"ego_view": b"jpeg"}, timestamps={"ego_view": 9_999_999_999.})
    assert assembler.ingest("ego_view", data, 2.)
    assert not assembler.ingest("ego_view", data, 2.1)
    assert assembler.histories[CAMERA_KEYS[0]][-1].received == 2.


def test_missing_stream_waits_nonblocking_then_late_frame_uses_next_ego():
    assembler = CameraSnapshotAssembler(CONTRACT)
    ingest(assembler, "ego_view", 1.)
    ingest(assembler, "video6", 1.012)
    assert assembler.poll_bundle(1.015) is None
    # 21ms is still within the 25ms pairing spread, but beyond the wait deadline.
    ingest(assembler, "video8", 1.021)
    assert assembler.poll_bundle(1.021) is None
    ingest(assembler, "ego_view", 1.03, 1)
    bundle = assembler.poll_bundle(1.03)
    assert bundle is not None
    assert bundle["timing"]["camera_anchor_time"] == 1.03
    assert bundle["timing"]["cameras"][CAMERA_KEYS[2]]["received_monotonic"] == 1.021


def test_triple_spread_must_fit_25ms_and_pair_selection_is_joint():
    assembler = CameraSnapshotAssembler(CONTRACT)
    ingest(assembler, "ego_view", 1.)
    ingest(assembler, "video6", .99)
    ingest(assembler, "video8", 1.02)
    assert assembler.poll_bundle(1.019) is None
    # A closer left frame at -10ms would span 30ms with right at +20ms.
    # A new left frame at +20ms makes a valid pair despite being farther from ego.
    ingest(assembler, "video6", 1.02, 1)
    bundle = assembler.poll_bundle(1.02)
    assert bundle is not None
    assert bundle["timing"]["cameras"][CAMERA_KEYS[1]]["received_monotonic"] == 1.02
    rejected = CameraSnapshotAssembler(CONTRACT)
    ingest(rejected, "ego_view", 1.)
    ingest(rejected, "video6", .99)
    ingest(rejected, "video8", 1.02)
    assert rejected.poll_bundle(1.02) is None


def test_equal_distance_prefers_older_wrist_deterministically():
    assembler = CameraSnapshotAssembler(CONTRACT)
    ingest(assembler, "ego_view", 1.)
    ingest(assembler, "video6", .99)
    ingest(assembler, "video6", 1.01, 1)
    ingest(assembler, "video8", 1.)
    bundle = assembler.poll_bundle(1.011)
    assert bundle["timing"]["cameras"][CAMERA_KEYS[1]]["received_monotonic"] == .99


@pytest.mark.parametrize("decision,accepted", [(1.19, True), (1.191, False)])
def test_every_camera_receipt_obeys_200ms_age(decision, accepted):
    assembler = CameraSnapshotAssembler(CONTRACT)
    ingest(assembler, "ego_view", 1.)
    ingest(assembler, "video6", .99)
    ingest(assembler, "video8", 1.005)
    assert (assembler.poll_bundle(decision) is not None) == accepted


def test_history_is_bounded_and_future_receipts_are_never_selected():
    assembler = CameraSnapshotAssembler(CONTRACT)
    for index in range(200):
        ingest(assembler, "ego_view", 1. + index / 1000, index)
    assert len(assembler.histories[CAMERA_KEYS[0]]) == 128
    assert assembler.histories[CAMERA_KEYS[0]][0].identity == 72
    ingest(assembler, "video6", 1.201)
    ingest(assembler, "video8", 1.202)
    assert assembler.poll_bundle(1.199) is None
    assert assembler.poll_bundle(1.203) is not None


@pytest.mark.parametrize("received", [float("nan"), float("inf"), -1.])
def test_rejects_invalid_client_receipt_clock(received):
    with pytest.raises(ValueError, match="client monotonic"):
        CameraSnapshotAssembler(CONTRACT).ingest("ego_view", packet("ego_view"), received)


def test_rejects_regression_missing_identity_and_ambiguous_aliases():
    assembler = CameraSnapshotAssembler(CONTRACT)
    ingest(assembler, "ego_view", 2.)
    with pytest.raises(ValueError, match="regressed"):
        ingest(assembler, "ego_view", 1.99, 1)
    with pytest.raises(ValueError, match="frame identity"):
        assembler.ingest("video6", dict(images={"video6": b"jpeg"}), 2.)
    with pytest.raises(ValueError, match="exactly one"):
        assembler.ingest("video6", dict(images={"video6": b"a", "left_wrist": b"b"}), 2.)
    with pytest.raises(ValueError, match="finite"):
        assembler.poll_bundle(math.nan)


def test_real_zmq_independent_streams_match_source_schema_and_codec():
    class Clock:
        now = 10.
        def __call__(self):
            return self.now

    clock = Clock()
    context = zmq.Context()
    publishers = {}
    receiver = None
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:] = [24, 68, 194]
    encoded = cv2.imencode(".jpg", rgb[..., ::-1])[1].tobytes()
    try:
        endpoints = {}
        for source in ("ego_view", "video6", "video8"):
            pub = context.socket(zmq.XPUB)
            pub.setsockopt(zmq.LINGER, 0)
            port = pub.bind_to_random_port("tcp://127.0.0.1")
            endpoints[source] = f"tcp://127.0.0.1:{port}"
            publishers[source] = pub
        receiver = IndependentCameraReceiver(endpoints, CONTRACT, context=context, clock=clock)
        # Subscription handshake avoids slow-joiner sleeps and uses actual ZMQ sockets.
        for publisher in publishers.values():
            assert publisher.poll(1000, zmq.POLLIN)
            assert publisher.recv() == b"\x01"
        bundle = None
        for source, receipt, canonical in zip(publishers, (10., 10.008, 10.016), CAMERA_KEYS):
            publishers[source].send(msgpack.packb(packet(source, image=encoded), use_bin_type=True))
            assert receiver.sockets[canonical][1].poll(1000, zmq.POLLIN)
            clock.now = receipt
            bundle = receiver.poll()
        assert bundle is not None
        frames, times = object.__new__(SonicZmqIO).decode_camera_bundle(bundle, clock.now)
        assert list(times.values()) == [10., 10.008, 10.016]
        assert all(np.max(np.abs(decode_camera_image(frame).astype(int)-rgb.astype(int))) <= 2
                   for frame in frames.values())
        assert receiver.poll() is None
        receiver.close()
        receiver = None
        assert not context.closed
    finally:
        if receiver is not None:
            receiver.close()
        for publisher in publishers.values():
            publisher.close(linger=0)
        context.term()


def composer_packet():
    names = ("ego_view", "left_wrist", "right_wrist")
    receipts = (10., 9.99, 10.012)
    return dict(
        images={name: b"encoded-jpeg" for name in names},
        timestamps={name: 1_791_548_000.+index for index, name in enumerate(names)},
        sources={name: dict(received_monotonic=stamp, frame_id=f"session:{name}:0",
                            original_timestamp=100.+index, original_clock="robot_monotonic")
                 for index, (name, stamp) in enumerate(zip(names, receipts))},
        sequence=1, composer_session="session",
        sync=dict(method="workstation_receive_time", hardware_synchronized=False,
                  contract=CONTRACT.copy(), anchor_monotonic=10., emitted_monotonic=10.021),
    )


def test_actual_local_composer_preserves_upstream_receipts_not_bundle_arrival_or_robot_clock():
    packet = composer_packet()
    bundle = normalize_collector_camera_bundle(packet, 10.03, CONTRACT, endpoint="tcp://127.0.0.1:5560")
    frames, times = object.__new__(SonicZmqIO).decode_camera_bundle(bundle, 10.03)
    assert list(times.values()) == [10., 9.99, 10.012]
    assert all(value == b"encoded-jpeg" for value in frames.values())
    assert bundle["timing"]["camera_anchor_time"] == 10.
    assert bundle["timing"]["cameras"][CAMERA_KEYS[1]]["original_timestamp"] == 101.
    assert bundle["timing"]["source_clock"] == "same_workstation_monotonic"
    assert "boot_id" not in bundle["timing"]  # Actual producer supplies no boot identity.
    io = object.__new__(SonicZmqIO)
    io.config = SimpleNamespace(camera_bundle_format="chenyz_composer_v1", camera_endpoint="tcp://127.0.0.1:5560")
    io.observation_contract = dict(collector=CONTRACT)
    assert io.decode_camera_bundle(packet, 10.03)[1] == times


@pytest.mark.parametrize("endpoint", ["tcp://192.0.2.222:5560", "tcp://robot:5560", "tcp://0.0.0.0:5560"])
def test_composer_requires_explicit_direct_same_workstation_endpoint(endpoint):
    with pytest.raises(ValueError, match="same-workstation"):
        normalize_collector_camera_bundle(composer_packet(), 10.03, CONTRACT, endpoint=endpoint)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["sync"]["contract"].update(max_age_ms=201.), "differs"),
    (lambda p: p["sources"]["left_wrist"].pop("received_monotonic"), "per-camera"),
    (lambda p: p["sources"]["left_wrist"].update(received_monotonic=9.8), "stale/future"),
    (lambda p: p["sources"]["left_wrist"].update(received_monotonic=10.04), "stale/future"),
    (lambda p: p["sources"]["left_wrist"].update(received_monotonic=9.98), "spread"),
    (lambda p: p["sync"].update(anchor_monotonic=10.001), "anchor/emission"),
    (lambda p: p["sync"].update(emitted_monotonic=10.031), "anchor/emission"),
    (lambda p: p["sources"]["left_wrist"].update(frame_id=""), "frame identity"),
])
def test_composer_rejects_invalid_timing_without_inventing_freshness(mutation, match):
    packet = copy.deepcopy(composer_packet())
    mutation(packet)
    with pytest.raises(ValueError, match=match):
        normalize_collector_camera_bundle(packet, 10.03, CONTRACT, endpoint="tcp://127.0.0.1:5560")


def test_actual_base64_and_binary_jpeg_codecs_recover_same_rgb_channels():
    rgb = np.full((480, 640, 3), [24, 68, 194], dtype=np.uint8)
    binary = cv2.imencode(".jpg", rgb[..., ::-1])[1].tobytes()
    # The supplied legacy ImageMessageSchema encodes its RGB array directly
    # with cv2, then decodes without the binary-MJPEG channel conversion.
    legacy = base64.b64encode(cv2.imencode(".jpg", rgb)[1]).decode("ascii")
    for payload in (binary, legacy, rgb):
        assert np.max(np.abs(decode_camera_image(payload).astype(int)-rgb.astype(int))) <= 2


def test_camera_restart_namespaces_sequence_and_ignores_unrelated_depth_payload():
    assembler = CameraSnapshotAssembler(CONTRACT)
    packet = dict(images={"observation.images.video6": b"jpeg", "depth": b"other"},
                  sequence=0, source_session="first")
    assert assembler.ingest("video6", packet, 1.)
    assert not assembler.ingest("video6", packet, 1.01)
    packet["source_session"] = "restart"
    assert assembler.ingest("video6", packet, 1.02)
    assert len(assembler.histories[CAMERA_KEYS[1]]) == 2
