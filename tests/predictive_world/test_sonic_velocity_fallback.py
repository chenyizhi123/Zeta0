"""CPU checks of actual feedback boundaries and the unapplied native ACK patch."""

import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest

from psi.deploy.sonic_velocity_fallback import (
    ControllerStatus,
    VelocityFallbackHandshake,
    idle_messages,
)
from psi.deploy.sonic_wire import HEADER_SIZE


def packet(clock=100., *, mode=0, sequence=0, frame=-1, valid=0, age=-1.,
           encoder=1, ready=0, pid=123, legacy=False, **overrides):
    fields = dict(controller_pid=pid, controller_time_monotonic=clock,
                  input_control_mode=mode, encoder_active=encoder, planner_ready=1,
                  velocity_fallback_enabled=1, zero_velocity_requested=int(mode == 0))
    if not legacy:
        fields.update(velocity_fallback_ack_version=1, control_active=1,
                      token_stream_ready=ready, decoder_token_input_valid=valid,
                      decoder_token_commit_sequence=sequence,
                      decoder_token_frame_index=frame, decoder_token_receive_age_s=age)
    fields.update(overrides)
    return {name: [value] for name, value in fields.items()}


def idle():
    handshake = VelocityFallbackHandshake()
    handshake.observe(packet(), 1.)
    handshake.request_idle(1.01)
    handshake.observe(packet(100.02), 1.02)
    assert handshake.idle_confirmed(1.02)
    return handshake


def pending():
    handshake = idle()
    handshake.request_stream(1.03)
    return handshake


def test_wire_stages_real_zero_velocity_before_start_not_stop():
    planner, command = idle_messages(upper_body_position=np.arange(17))
    def unpack(raw, topic):
        offset = len(topic)
        header = json.loads(raw[offset:offset+HEADER_SIZE].rstrip(b"\0"))
        return header, raw[offset+HEADER_SIZE:]
    header, payload = unpack(planner, b"planner")
    assert [field["name"] for field in header["fields"]][:5] == ["mode", "movement", "facing", "speed", "height"]
    assert struct.unpack("<i", payload[:4]) == (0,)
    assert struct.unpack("<3f", payload[4:16]) == (0., 0., 0.)
    assert struct.unpack("<2f", payload[28:36]) == (0., -1.)
    _, flags = unpack(command, b"command")
    assert flags == bytes([1, 0, 1])
    with pytest.raises(ValueError, match="17 finite"):
        idle_messages(upper_body_position=np.zeros(29))


def test_idle_ack_requires_new_same_pid_progress_not_absolute_clock_equality():
    handshake = VelocityFallbackHandshake()
    handshake.request_idle(1.)
    handshake.observe(packet(987654.), 1.01)
    assert not handshake.idle_confirmed(1.01)
    assert not handshake.observe(packet(987654.), 1.02)  # Duplicate is not a new heartbeat.
    assert handshake.status.received == 1.01
    handshake.observe(packet(987654.02), 1.03)
    assert handshake.idle_confirmed(1.03)
    assert not handshake.idle_confirmed(1.131)


@pytest.mark.parametrize("field", ["planner_ready", "zero_velocity_requested", "velocity_fallback_enabled", "control_active"])
def test_idle_requires_all_real_native_conditions(field):
    handshake = VelocityFallbackHandshake()
    handshake.observe(packet(), 1.)
    handshake.request_idle(1.01)
    handshake.observe(packet(100.02, **{field: 0}), 1.02)
    assert not handshake.idle_confirmed(1.02)


def test_legacy_can_prove_idle_but_cannot_enable_token_resume():
    handshake = VelocityFallbackHandshake()
    handshake.observe(packet(legacy=True), 1.)
    handshake.request_idle(1.01)
    handshake.observe(packet(100.02, legacy=True), 1.02)
    assert handshake.idle_confirmed(1.02)
    assert not handshake.strict_capable(1.02)
    with pytest.raises(ValueError, match="token ACK v1"):
        handshake.request_stream(1.03)


def test_new_slots_may_send_before_mode_ack_and_any_sent_commit_confirms():
    handshake = pending()
    assert handshake.stream_accepting(1.031)  # Native reset can discard this first packet.
    handshake.register_sent_frame(40, 1.031)
    handshake.observe(packet(100.04, mode=1, ready=1), 1.04)
    assert not handshake.stream_confirmed(1.04)  # Encoder may remain active until a real token.
    handshake.register_sent_frame(41, 1.05)
    handshake.observe(packet(100.06, mode=1, ready=1, encoder=0,
                             sequence=2, frame=41, valid=1, age=.01), 1.06)
    assert handshake.phase == "stream"
    assert handshake.stream_confirmed(1.06)
    assert handshake.stream_alive(1.06)


@pytest.mark.parametrize("override", [
    {"decoder_token_input_valid": 0}, {"decoder_token_frame_index": 42},
    {"decoder_token_commit_sequence": 0}, {"encoder_active": 1},
    {"decoder_token_receive_age_s": .101}, {"decoder_token_receive_age_s": -.01},
    {"token_stream_ready": 0}, {"input_control_mode": 0}, {"control_active": 0},
])
def test_mode_flags_encoder_tokens_and_unconsumed_or_stale_frames_are_not_ack(override):
    handshake = pending()
    handshake.register_sent_frame(40, 1.031)
    response = packet(100.04, mode=1, ready=1, encoder=0, sequence=1, frame=40, valid=1, age=.01)
    response.update({name: [value] for name, value in override.items()})
    response["token_state"] = [0.] * 64  # Existing model/encoder output proves nothing.
    handshake.observe(response, 1.04)
    assert not handshake.stream_confirmed(1.04)


def test_resends_or_clock_progress_do_not_extend_native_100ms_window():
    handshake = pending()
    handshake.register_sent_frame(40, 1.031)
    handshake.observe(packet(100.06, mode=1, ready=1), 1.06)
    handshake.register_sent_frame(41, 1.07)
    assert not handshake.stream_accepting(1.1301)
    assert not handshake.stream_confirmed(1.1301)
    handshake.request_idle(1.131)
    handshake.observe(packet(100.14), 1.14)
    assert handshake.idle_confirmed(1.14)
    handshake.request_stream(1.15)
    handshake.register_sent_frame(42, 1.151)
    # A late ACK from the discarded stream epoch must not confirm this one.
    handshake.observe(packet(100.16, mode=1, ready=1, encoder=0,
                             sequence=1, frame=41, valid=1, age=.01), 1.16)
    assert not handshake.stream_confirmed(1.16)


def test_cached_heartbeat_never_refreshes_receipt_or_commit_identity():
    handshake = pending()
    handshake.register_sent_frame(40, 1.031)
    handshake.observe(packet(100.04, mode=1, ready=1, encoder=0,
                             sequence=1, frame=40, valid=1, age=.01), 1.04)
    assert handshake.stream_alive(1.04)
    assert not handshake.observe(packet(100.04, mode=1, ready=1, encoder=0,
                                       sequence=1, frame=40, valid=1, age=.01), 1.13)
    assert not handshake.stream_alive(1.141)
    with pytest.raises(ValueError, match="new increasing"):
        handshake.register_sent_frame(40, 1.131)


@pytest.mark.parametrize("alteration", [
    {"controller_pid": [124]}, {"controller_time_monotonic": [99.]},
    {"controller_pid": [float("nan")]}, {"planner_ready": [2]},
    {"velocity_fallback_ack_version": [2]}, {"decoder_token_commit_sequence": [-1]},
])
def test_native_identity_schema_and_counter_errors_fail_closed(alteration):
    handshake = idle()
    response = packet(100.04)
    response.update(alteration)
    with pytest.raises(ValueError):
        handshake.observe(response, 1.04)
    assert handshake.phase == "failed"
    assert not handshake.strict_capable(1.04)
    with pytest.raises(ValueError):
        handshake.request_idle(1.05)


def test_flat_schema_partial_patch_and_future_receipts_are_rejected():
    with pytest.raises(ValueError, match="controller_pid"):
        ControllerStatus.from_packet({"state": packet()}, 1.)
    response = packet()
    del response["decoder_token_frame_index"]
    with pytest.raises(ValueError, match="Incomplete"):
        ControllerStatus.from_packet(response, 1.)
    handshake = VelocityFallbackHandshake()
    with pytest.raises(ValueError, match="future"):
        handshake.observe(packet(), 1.1, now=1.)


def test_idle_retry_does_not_hide_native_initialization_timeout():
    handshake = VelocityFallbackHandshake()
    handshake.request_idle(1.)
    handshake.request_idle(5.9)
    with pytest.raises(ValueError, match="timed out"):
        handshake.idle_confirmed(6.001)


def test_patch_hashes_apply_offline_and_snapshot_has_real_commit_identity(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    provenance = json.loads((repo/"real/patches/sonic_velocity_fallback_ack_v1.provenance.json").read_text())
    patch = repo/"real/patches"/provenance["patch_file"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == provenance["patch_sha256"]
    source = repo/"diagnostics/place_doll_lerobot_flow_review_20261014/collection_source_recovered_partial/chenyz/GR00T-WholeBodyControl"
    if not source.is_dir():
        pytest.skip("CRC-recovered user source is available only in the source audit workspace")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A CPU C++ compiler is required for the native snapshot boundary")
    for entry in provenance["files"]:
        original = source/entry["path"]
        assert hashlib.sha256(original.read_bytes()).hexdigest() == entry["source_sha256"]
        target = tmp_path/entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True, capture_output=True)
    for entry in provenance["files"]:
        assert hashlib.sha256((tmp_path/entry["path"]).read_bytes()).hexdigest() == entry["patched_sha256"]
    header = tmp_path/"gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/token_timing_snapshot.hpp"
    program = tmp_path/"snapshot.cpp"
    program.write_text('''#include "token_timing_snapshot.hpp"
#include <cassert>
int main() {
  TokenTimingBuffer buffer;
  auto now=TokenTimingSnapshot::Clock::now();
  std::vector<double> token(64,.125);
  buffer.Commit(token,now,7);
  const auto consumed=buffer.Snapshot();
  assert(consumed.commit_sequence==1 && consumed.frame_index==7);
  assert(buffer.Snapshot().commit_sequence==1); // Cached policy reuse is not a commit.
  buffer.Commit(token,now+std::chrono::milliseconds(20),9);
  assert(consumed.commit_sequence==1 && consumed.frame_index==7);
  assert(consumed.received_at==now); // New packet cannot relabel the consumed snapshot.
  assert(buffer.Snapshot().commit_sequence==2 && buffer.Snapshot().frame_index==9);
  buffer.Clear();
  auto reset=buffer.Snapshot();
  assert(reset.commit_sequence==2 && reset.frame_index==-1 && !reset.received_at);
  buffer.Commit(token,now+std::chrono::milliseconds(40),11);
  assert(buffer.Snapshot().commit_sequence==3 && buffer.Snapshot().frame_index==11);
}''')
    binary = tmp_path/"snapshot"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(header.parent), str(program), "-o", str(binary)],
                   check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True)
