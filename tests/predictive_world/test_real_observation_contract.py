"""Recorded camera-anchored pairing survives ordinary decision-time latency."""
import copy

import pytest

from psi.real_observation_contract import (
    CAMERA_KEYS, build_recorded_action_projection, make_deployment_observation_contract, snapshot_timing_violations,
    validate_action_projection,
    validate_deployment_observation_contract, validate_observation_contract, validate_snapshot_timing,
)


@pytest.fixture
def collector():
    return dict(version=1, camera_max_skew_ms=25., camera_wait_ms=20., max_age_ms=200.,
                body_max_offset_ms=40., hand_max_offset_ms=50., history_size=128, image_shape=[480, 640, 3])


@pytest.fixture
def snapshot(collector):
    # Recorded frame 12's relevant shape: pairing is good, while the decision
    # occurs ~58ms after the camera anchor. A 40ms decision-age rule rejects it.
    return dict(decision_time=100., camera_anchor_time=99.942,
                camera_times=dict(zip(CAMERA_KEYS, [99.942, 99.9467, 99.9293])),
                body_time=99.9393, hand_times=[99.940, 99.940], contract=collector)


def test_normal_capture_delay_does_not_break_camera_pairing(snapshot):
    assert snapshot["decision_time"]-snapshot["body_time"] > .04
    assert snapshot_timing_violations(**snapshot) == []
    validate_snapshot_timing(**snapshot)


def test_pairing_can_use_a_later_feedback_that_arrived_before_decision(snapshot):
    snapshot.update(body_time=99.970, hand_times=[99.980, 99.985])
    validate_snapshot_timing(**snapshot)


@pytest.mark.parametrize("change,code", [
    ({"body_time": 100.01}, "snapshot_future_input"),
    ({"decision_time": 100.2}, "snapshot_stale"),
    ({"body_time": 99.89}, "snapshot_body_offset"),
    ({"hand_times": [99.880, 99.940]}, "snapshot_hand_offset"),
    ({"camera_anchor_time": 99.945}, "snapshot_anchor_mismatch"),
])
def test_real_boundary_violations_are_explicit(snapshot, change, code):
    snapshot.update(change)
    codes = {entry["code"] for entry in snapshot_timing_violations(**snapshot)}
    assert code in codes
    with pytest.raises(ValueError, match=code):
        validate_snapshot_timing(**snapshot)


def test_mixing_cameras_from_different_bundles_is_rejected(snapshot):
    snapshot["camera_times"][CAMERA_KEYS[2]] = 99.990
    assert "snapshot_camera_skew" in {entry["code"] for entry in snapshot_timing_violations(**snapshot)}


def test_nonfinite_timestamp_does_not_publish_nonfinite_json_evidence(snapshot):
    snapshot["body_time"] = float("nan")
    assert snapshot_timing_violations(**snapshot) == [dict(code="snapshot_nonfinite_time", evidence={})]


def test_short_camera_names_preserve_the_same_contract(snapshot):
    snapshot["camera_times"] = {key.removeprefix("observation.images."): value
                                for key, value in snapshot["camera_times"].items()}
    validate_snapshot_timing(**snapshot)


def test_source_configuration_binds_the_advertised_limits(collector):
    data = dict(real_observation_mode="collector_snapshot_v1", real_observation_contract=collector,
                state_history_count=1, state_max_age_s=.2, visual_max_age_s=.2)
    wire = make_deployment_observation_contract(data)
    assert wire["collector"] == collector
    assert validate_deployment_observation_contract(wire) == wire
    wire["state_max_age_s"] = .5
    with pytest.raises(ValueError, match="match"):
        validate_deployment_observation_contract(wire)


def test_legacy_contract_stays_absent():
    assert make_deployment_observation_contract(dict(state_history_count=21)) is None


def test_latest_available_v2_checks_causality_and_age_without_pairing_limits(snapshot):
    contract = dict(version=2, selection="latest_available_v1", max_age_ms=200.,
                    history_size=128, image_shape=[480, 640, 3],
                    cameras=["ego_view", "left_wrist", "right_wrist"],
                    timestamp_kind="workstation_source_receive_time")
    assert validate_observation_contract(contract) == contract
    snapshot.update(contract=contract, body_time=99.81, hand_times=[99.82, 99.83])
    snapshot["camera_times"][CAMERA_KEYS[2]] = 99.81
    assert snapshot_timing_violations(**snapshot) == []
    snapshot["body_time"] = 100.01
    assert {r["code"] for r in snapshot_timing_violations(**snapshot)} == {"snapshot_future_input"}
    snapshot["body_time"] = 99.79
    assert {r["code"] for r in snapshot_timing_violations(**snapshot)} == {"snapshot_stale"}
    data = dict(real_observation_mode="collector_snapshot_v1", real_observation_contract=contract,
                state_history_count=1, state_max_age_s=.2, visual_max_age_s=.2)
    wire = make_deployment_observation_contract(data)
    assert validate_deployment_observation_contract(wire) == wire


def latest_v3():
    return dict(version=3, selection="latest_available_v1", max_age_ms=None,
                age_policy="record_latest_available", history_size=128, image_shape=[480, 640, 3],
                cameras=["ego_view", "left_wrist", "right_wrist"],
                timestamp_kind="workstation_source_receive_time")


def test_v3_preserves_unbounded_source_and_binds_explicit_training_cutoff(snapshot):
    contract = latest_v3()
    before = copy.deepcopy(contract)
    snapshot.update(contract=contract, body_time=99., hand_times=[98., 99.9])
    assert snapshot_timing_violations(**snapshot) == []
    assert {r["code"] for r in snapshot_timing_violations(**snapshot, snapshot_max_age_s=.2)} == {"snapshot_stale"}
    data = dict(real_observation_mode="collector_snapshot_v1", real_observation_contract=contract,
                state_history_count=1, state_max_age_s=.2, visual_max_age_s=.2, real_snapshot_max_age_s=.2)
    wire = make_deployment_observation_contract(data)
    assert wire["collector"]["max_age_ms"] is None
    assert wire["real_snapshot_max_age_s"] == .2
    assert validate_deployment_observation_contract(wire) == wire
    assert contract == before
    snapshot["hand_times"][0] = 100.01
    assert "snapshot_future_input" in {r["code"] for r in snapshot_timing_violations(**snapshot)}


@pytest.mark.parametrize("limit", [None, 0, -.1, float("inf"), float("nan"), True])
def test_v3_requires_explicit_valid_training_cutoff(limit):
    data = dict(real_observation_mode="collector_snapshot_v1", real_observation_contract=latest_v3(),
                state_history_count=1, state_max_age_s=.2, visual_max_age_s=.2, real_snapshot_max_age_s=limit)
    with pytest.raises(ValueError, match="explicit finite positive"):
        make_deployment_observation_contract(data)


@pytest.mark.parametrize("change", [{"max_age_ms": 200.}, {"age_policy": "assume_fresh"}])
def test_v3_rejects_relabeling_its_source_contract(change):
    with pytest.raises(ValueError, match="Collector v3"):
        validate_observation_contract({**latest_v3(), **change})


def test_recorded_projection_preserves_rare_training_tokens_and_legacy_bounds():
    import numpy as np

    from psi.sonic_actions import controller_command_values

    profile = build_recorded_action_projection(-.75, .6875, "a"*64)
    actions = np.zeros((2, 76), dtype=np.float32)
    actions[0, 0], actions[1, 0] = -.75, .6875
    expected = actions.copy()
    assert np.array_equal(controller_command_values(actions, projection=profile), expected)
    assert np.array_equal(actions, expected)
    assert np.array_equal(controller_command_values(actions)[:, 0], [-.625, .625])
    actions[0, 0], actions[1, 0] = -1., 1.
    assert np.array_equal(controller_command_values(actions, projection=profile)[:, 0], [-.75, .6875])


def test_projection_is_carried_from_training_config_to_wire(collector):
    profile = build_recorded_action_projection(-.75, .6875, "b"*64)
    data = dict(real_observation_mode="collector_snapshot_v1", real_observation_contract=collector,
                state_history_count=1, state_max_age_s=.2, visual_max_age_s=.2, real_action_projection=profile)
    wire = make_deployment_observation_contract(data)
    assert wire["action_projection"] == profile
    assert validate_deployment_observation_contract(wire) == wire


@pytest.mark.parametrize("change", [
    {"source_sha256": "unbound"}, {"motion_token_step": .125}, {"motion_token_max": .69},
    {"motion_token_min": float("nan")}, {"hand_max": 1000.}, {"range_source": "guessed_hardware_limit"},
])
def test_projection_must_be_a_bound_recorded_grid(change):
    profile = build_recorded_action_projection(-.75, .6875, "c"*64)
    profile.update(change)
    with pytest.raises(ValueError):
        validate_action_projection(profile)


@pytest.mark.parametrize("change", [
    {"version": True}, {"max_age_ms": float("inf")}, {"max_age_ms": 0},
    {"history_size": 0}, {"image_shape": None}, {"unexpected": 1},
])
def test_malformed_source_contract_cannot_configure_a_client(collector, change):
    invalid = copy.deepcopy(collector)
    invalid.update(change)
    with pytest.raises(ValueError):
        validate_observation_contract(invalid)
