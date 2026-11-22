import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.config.predictive_world import PredictiveWorldDataConfig
from psi.models.predictive_world.observation_contract_transition import (
    REAL_TEMPORAL_FIELDS, observation_contract_transition, verify_observation_contract_transition,
)
from psi.models.predictive_world.state_history_transition import current_state_transition


def collector_contract():
    return dict(version=1, camera_max_skew_ms=25., camera_wait_ms=20., max_age_ms=200.,
        body_max_offset_ms=40., hand_max_offset_ms=50., history_size=128, image_shape=[480, 640, 3])


def target_data(**changes):
    return PredictiveWorldDataConfig(observed_visual_mode="current_multiview", state_history_count=1,
        real_observation_mode="collector_snapshot_v1", real_observation_contract=collector_contract(),
        real_action_label_mode="held_command_v1", state_max_age_s=.2, visual_max_age_s=.2,
        future_measurement_max_age_s=.04, future_visual_max_age_s=.08, **changes)


def contracts():
    payload = dict(data=dict(state_history_count=21, state_max_age_s=.04, visual_max_age_s=.08))
    temporal = dict(state_history_count=21, state_query_hz=50., action_hz=50., action_horizon=30,
        history_visual_offsets_s=[-.4, -.2, 0], future_group_ends_s=[.2, .4, .6], observed_visual_mode="current_multiview")
    source = dict(weights_sha256="a" * 64, metadata=dict(temporal_abi=temporal, resolved_config_sha256=canonical_sha256(payload)))
    data = target_data()
    target = {**temporal, **data.observation_contract, "state_history_count": 1}
    return source, payload, data, target


def test_v3_transition_binds_training_cutoff_without_relabeling_collector():
    source, payload, data, temporal = contracts()
    collector = dict(version=3, selection="latest_available_v1", max_age_ms=None,
        age_policy="record_latest_available", history_size=128, image_shape=[480, 640, 3],
        cameras=["ego_view", "left_wrist", "right_wrist"], timestamp_kind="workstation_source_receive_time")
    data = replace(data, real_observation_contract=collector, real_snapshot_max_age_s=.2)
    target = {**temporal, **data.observation_contract}
    record = observation_contract_transition(source, payload, target)
    assert record["target_observation_contract"]["collector"]["max_age_ms"] is None
    assert record["target_observation_contract"]["real_snapshot_max_age_s"] == .2
    assert verify_observation_contract_transition(record, source, payload, target, target_data=data) == record
    changed = replace(data, real_snapshot_max_age_s=.3, state_max_age_s=.3, visual_max_age_s=.3)
    with pytest.raises(ValueError, match="disagrees"):
        verify_observation_contract_transition(record, source, payload, target, target_data=changed)


def test_observation_transition_binds_source_target_and_preserves_teacher_limits():
    source, payload, data, target = contracts()
    record = observation_contract_transition(source, payload, target)
    assert record["preserved_future_limits"] == dict(future_measurement_max_age_s=.04, future_visual_max_age_s=.08,
        action_max_age_s=.04)
    assert record["target_observation_contract"]["state_max_age_s"] == .2
    assert record["allowed_temporal_fields"] == [
        "real_action_label_mode", "real_observation_contract", "real_observation_mode",
    ]
    assert "action_projection" not in record["target_observation_contract"]
    assert verify_observation_contract_transition(record, source, payload, target, target_data=data) == record
    changed = copy.deepcopy(source)
    changed["weights_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="exact source and target"):
        verify_observation_contract_transition(record, changed, payload, target)


def test_recorded_projection_is_an_explicit_fourth_temporal_field_without_changing_v4():
    from psi.real_observation_contract import build_recorded_action_projection

    source, payload, data, temporal = contracts()
    v4_record = observation_contract_transition(source, payload, temporal)
    projection = build_recorded_action_projection(-.75, .6875, "b" * 64)
    data = replace(data, real_action_projection=projection)
    target = {**temporal, "real_action_projection": projection}
    record = observation_contract_transition(source, payload, target)
    assert record["target_observation_contract"]["action_projection"] == projection
    assert record["allowed_temporal_fields"] == sorted(set(target) & REAL_TEMPORAL_FIELDS)
    assert len(record["allowed_temporal_fields"]) == 4
    assert verify_observation_contract_transition(record, source, payload, target, target_data=data) == record
    # Re-reading a published v4 record after adding v5 support is exact.
    assert observation_contract_transition(source, payload, temporal) == v4_record
    altered = copy.deepcopy(target)
    altered["real_action_projection"]["motion_token_min"] = -.625
    with pytest.raises(ValueError, match="exact source and target"):
        verify_observation_contract_transition(record, source, payload, altered)


@pytest.mark.parametrize("change", ["horizon", "rate", "groups", "observed_mode", "full_index_abi"])
def test_observation_transition_rejects_undeclared_temporal_changes(change):
    source, payload, _, target = contracts()
    if change == "horizon":
        target["action_horizon"] = 50
    elif change == "rate":
        target["action_hz"] = 30
    elif change == "groups":
        target["future_group_ends_s"] = [.3, .4, .6]
    elif change == "observed_mode":
        target["observed_visual_mode"] = "history"
    else:
        target["selection"] = dict(future_measurement=dict(max_age_s=.04))
    with pytest.raises(ValueError, match="existing model/teacher temporal"):
        observation_contract_transition(source, payload, target)


def test_observation_transition_does_not_relax_future_supervision_or_state_resize():
    source, payload, data, target = contracts()
    record = observation_contract_transition(source, payload, target)
    changed = asdict(data)
    changed["future_visual_max_age_s"] = .2
    with pytest.raises(ValueError, match="future teacher/action age"):
        verify_observation_contract_transition(record, source, payload, target, target_data=changed)
    changed = copy.deepcopy(target)
    changed["state_history_count"] = 2
    with pytest.raises(ValueError, match="one measured state"):
        observation_contract_transition(source, payload, changed)
    changed = copy.deepcopy(payload)
    changed["data"]["state_max_age_s"] = .1
    with pytest.raises(ValueError, match="source config checksum"):
        observation_contract_transition(source, changed, target)


def test_joint_initialization_combines_current_state_and_observation_records(tmp_path):
    from torch import nn
    from test_checkpoint import _joint_metadata
    from test_current_state import small_modules
    from psi.models.predictive_world.checkpoint import save_predictive_world_bundle
    from psi.models.predictive_world.initialization import initialize_joint
    from psi.models.predictive_world.latent_transition import LatentNormalizer
    from psi.models.predictive_world.training import module_state_sha256

    source_modules, protocols, states = small_modules(21)
    destination, _, _ = small_modules(1)
    for modules in (source_modules, destination):
        modules["normalizer"] = LatentNormalizer((3, 8, 4))
        modules["normalizer"].fitted.fill_(True)
        modules["encoder"] = nn.Linear(2, 2)
        modules["encoder"].register_buffer("normalizer_frozen", torch.tensor(True))
        modules["encoder"].register_buffer("last_target_update", torch.tensor(14999))
    source, payload, data, temporal = contracts()
    payload["data"].update(index_path=str(tmp_path / "index.jsonl"))
    for name, value in (("action_protocol_registry", protocols), ("state_schema_registry", states),
                        ("measurement_registry", {"measured": dict(width=41)})):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        payload["data"][name] = str(path)
    (tmp_path / "index.jsonl.meta.json").write_text(json.dumps(dict(provenance=dict(registry_binding=dict(
        state_schema_registry_sha256=file_sha256(tmp_path / "state_schema_registry.json"))))))
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps(payload))
    metadata = {**_joint_metadata(), "architecture": {"test": "tiny"}, "temporal_abi": source["metadata"]["temporal_abi"],
        "resolved_config_sha256": canonical_sha256(payload), "optimization_contract": dict(target_encoder_ema_decay=.99),
        "protocol_registry_sha256": file_sha256(tmp_path / "action_protocol_registry.json"),
        "measurement_registry_sha256": file_sha256(tmp_path / "measurement_registry.json"),
        "normalization_id": module_state_sha256(source_modules["normalizer"]), "loss_contract": {"same": True}}
    checkpoint = tmp_path / "joint"
    sidecar_path = save_predictive_world_bundle(checkpoint, phase="joint", modules=source_modules,
        completed_updates=15000, metadata=metadata)
    source = json.loads(sidecar_path.read_text())
    observation = observation_contract_transition(source, payload, temporal)
    training = SimpleNamespace(init_joint=str(checkpoint), init_source_config=str(config_path),
        init_processor_diagnostic=None, init_processor_migration=None, init_objective_migration=None, init_goal_intent=None,
        init_state_history_migration=current_state_transition(source, payload), init_observation_contract_migration=observation,
        init_body_target_migration=None,
        target_encoder_ema_decay=.99)
    data = replace(data, **{name: payload["data"][name] for name in (
        "action_protocol_registry", "state_schema_registry", "measurement_registry")})
    config = SimpleNamespace(training=training, data=data, runtime=SimpleNamespace(asset_path_map={}))
    expected = {**metadata, "temporal_abi": temporal}
    # A source-bound observation change never implicitly permits shape resizing.
    current = training.init_state_history_migration
    training.init_state_history_migration = None
    with pytest.raises(ValueError, match="explicit current-state position"):
        initialize_joint(config, destination, expected)
    training.init_state_history_migration = current
    report, _ = initialize_joint(config, destination, expected)
    assert report["observation_contract_migration"] == observation
    for name, key in (("history_encoder", "state_history_position"), ("teacher", "state_position"), ("student", "state_position")):
        torch.testing.assert_close(destination[name].state_dict()[key], source_modules[name].state_dict()[key][-1:], rtol=0, atol=0)
    assert {key: value for key, value in temporal.items() if key not in REAL_TEMPORAL_FIELDS} == {
        **source["metadata"]["temporal_abi"], "state_history_count": 1}
