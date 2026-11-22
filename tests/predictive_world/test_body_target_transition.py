"""Whole-body coverage changes coordinate meaning explicitly, even at width29."""
import copy

import pytest
import torch

from psi.data.predictive_world.cache import canonical_sha256
from psi.models.predictive_world.body_target_transition import (
    ARM_NAMES, OLD_SLOTS, EXTRA_NAMES, SCHEMA,
    whole_body_schema, make_body_target_transition, verify_body_target_transition,
    migrate_body_target_tensors,
)


def schema_fixture():
    names = [f"other_{i}" for i in range(64)]
    # Test a deliberately interleaved native layout; resolve names, not slices.
    for index, name in zip(range(6, 35), EXTRA_NAMES + ARM_NAMES):
        names[index] = name
    indices = [names.index(n) for n in ARM_NAMES]
    mean = [i / 100 for i in range(64)]
    std = [1. + i / 100 for i in range(64)]
    old = dict(width=64, world_action_eligible=True,
        fields=[dict(name="measured_q29", start=6, stop=35, kind="measurement",
                     unit="rad", coordinate_frame="robot_native", time_key="timestamp", statistics_id="state")],
        canonical_body=dict(layout_id="humanoid_upper_body_and_height_29_v1", slots=list(OLD_SLOTS),
            raw_indices=indices + [None]*15, mean=[mean[i] for i in indices]+[0.]*15,
            std=[std[i] for i in indices]+[1.]*15, normalization_space="per_source_registry_zscore_v1"))
    return old, names, mean, std


def test_whole_body_gathers_measured_joints_and_preserves_arm_semantics():
    old, names, mean, std = schema_fixture()
    before = copy.deepcopy(old)
    target = whole_body_schema(old, names, mean, std)["canonical_body"]
    assert set(target["raw_indices"]) == set(range(6, 35))
    for field in ("slots", "raw_indices", "mean", "std"):
        assert target[field][:14] == old["canonical_body"][field][:14]
    assert [names[i] for i in target["raw_indices"]] == list(ARM_NAMES + EXTRA_NAMES)
    assert old == before
    names[names.index(EXTRA_NAMES[0])] = "action.reference_joint"
    with pytest.raises(ValueError):
        whole_body_schema(old, names, mean, std)


@pytest.mark.parametrize("damage", ["record", "source", "registry", "old_schema"])
def test_body_transition_rejects_changed_bindings(damage):
    old, names, mean, std = schema_fixture()
    registry = {"old": old, SCHEMA: whole_body_schema(old, names, mean, std)}
    payload = {"training": {"seed": 0}}
    source = dict(phase="joint", weights_sha256="a"*64,
                  metadata=dict(resolved_config_sha256=canonical_sha256(payload), architecture=dict(measurement_dim=29)))
    record = make_body_target_transition(source, payload, registry)
    assert verify_body_target_transition(record, source, payload, registry, {"old": old}) == record
    if damage == "record":
        record["reset_slots"] = []
    elif damage == "source":
        source["weights_sha256"] = "b"*64
    elif damage == "registry":
        registry[SCHEMA]["canonical_body"]["mean"][20] += 1
    else:
        registry["old"] = {**old, "width": 65}
    with pytest.raises(ValueError):
        verify_body_target_transition(record, source, payload, registry, {"old": old})


def test_tensor_transition_preserves_old_function_for_arm_inputs_and_resets_only_declared_coordinates():
    torch.manual_seed(7)
    tensors = {f"encoder.{mode}.measurement_projection.0.weight": torch.randn(8, 58)
               for mode in ("online_encoder", "target_encoder")}
    tensors.update({"readout.body_output.1.weight": torch.randn(29, 8),
                    "readout.body_output.1.bias": torch.randn(29), "unrelated": torch.randn(7)})
    original = {k: v.clone() for k, v in tensors.items()}
    migrated = migrate_body_target_tensors(tensors)
    arm_only = torch.randn(4, 58)
    arm_only[:, 14:29] = 0
    arm_only[:, 43:58] = 0
    whole_body = torch.randn_like(arm_only)
    whole_body[:, :14] = arm_only[:, :14]
    whole_body[:, 29:43] = arm_only[:, 29:43]
    for mode in ("online_encoder", "target_encoder"):
        key = f"encoder.{mode}.measurement_projection.0.weight"
        torch.testing.assert_close(arm_only @ tensors[key].T, whole_body @ migrated[key].T)
    for key in ("readout.body_output.1.weight", "readout.body_output.1.bias"):
        assert torch.equal(migrated[key][:14], tensors[key][:14])
        assert torch.count_nonzero(migrated[key][14:]) == 0
    assert migrated["unrelated"] is tensors["unrelated"]
    assert all(torch.equal(tensors[k], v) for k, v in original.items())
    # Reset columns remain trainable and can learn newly observed joint inputs.
    weight = migrated["encoder.online_encoder.measurement_projection.0.weight"].requires_grad_()
    (whole_body @ weight.T).square().sum().backward()
    assert weight.grad[:, 14:29].abs().sum() > 0


def test_strict_joint_initialization_composes_body_objective_and_current_state_transitions(tmp_path):
    import json
    from dataclasses import asdict, replace

    from test_checkpoint import _joint_metadata
    from test_current_state import small_modules

    from psi.config.predictive_world import (
        PredictiveWorldConfig, PredictiveWorldDataConfig, PredictiveWorldRuntimeConfig,
        PredictiveWorldTrainingConfig, predictive_world_loss_contract,
    )
    from psi.data.predictive_world.cache import file_sha256
    from psi.data.predictive_world.sonic_lerobot_post_training import STATE_NAMES
    from psi.models.predictive_world.body_target_transition import REAL_SCHEMA, real_whole_body_schema
    from psi.models.predictive_world.checkpoint import save_predictive_world_bundle, read_predictive_world_bundle
    from psi.models.predictive_world.initialization import initialize_joint
    from psi.models.predictive_world.latent_transition import LatentNormalizer
    from psi.models.predictive_world.objective_transition import full_ert_training, make_objective_transition
    from psi.models.predictive_world.state_history_transition import current_state_transition
    from psi.models.predictive_world.representation import FutureTrajectoryEncoder, EMAFutureTrajectoryEncoder, WorldReadout
    from psi.models.predictive_world.training import module_state_sha256

    source_modules, protocols, states = small_modules(21)
    target, _, _ = small_modules(1)
    _, _, mean, std = schema_fixture()
    old = dict(width=41, world_action_eligible=True,
        fields=[dict(name="measured_q29", start=0, stop=29, kind="measurement",
                     unit="rad", coordinate_frame="robot_native", time_key="timestamp", statistics_id="state")],
        canonical_body=dict(layout_id="humanoid_upper_body_and_height_29_v1", slots=list(OLD_SLOTS),
            raw_indices=list(range(15, 29)) + [None] * 15, mean=mean[15:29] + [0.] * 15,
            std=std[15:29] + [1.] * 15, normalization_space="per_source_registry_zscore_v1"))
    source_registry = {"old": old}
    for modules in (source_modules, target):
        modules["normalizer"] = LatentNormalizer((3, 8, 4))
        modules["normalizer"].fitted.fill_(True)
        modules["encoder"] = EMAFutureTrajectoryEncoder(FutureTrajectoryEncoder(
            visual_dim=8, measurement_dim=29, latent_shape=(3, 8, 4), hidden_dim=16,
            num_blocks=1, num_heads=4, ff_dim=32))
        modules["encoder"].normalizer_frozen.fill_(True)
        modules["encoder"].last_target_update.fill_(14999)
        modules["readout"] = WorldReadout(visual_dim=8, measurement_dim=29,
            latent_shape=(3, 8, 4), hidden_dim=16, num_heads=4, ff_dim=32,
            num_blocks=1, max_scene_patches=3, body_points_per_group=2)
    registry_paths = {}
    for name, content in (("action_protocol_registry", protocols), ("state_schema_registry", states),
                          ("measurement_registry", source_registry)):
        registry_path = tmp_path / f"{name}.json"
        registry_path.write_text(json.dumps(content))
        registry_paths[name] = registry_path
    (tmp_path / "index.jsonl.meta.json").write_text(json.dumps({
        "provenance": {"registry_binding": {
            "state_schema_registry_sha256": file_sha256(registry_paths["state_schema_registry"]),
        }},
    }))
    data = PredictiveWorldDataConfig(**{name: str(path) for name, path in registry_paths.items()},
        index_path=str(tmp_path / "index.jsonl"), state_history_count=21,
        observed_visual_mode="current_multiview")
    source_training = PredictiveWorldTrainingConfig(phase="joint", world_model="world", max_updates=15000,
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0,
        rtc_training_max_delay=6)
    payload = {"data": asdict(data), "training": asdict(source_training)}
    source_config = tmp_path / "source_config.json"
    source_config.write_text(json.dumps(payload))
    temporal = {"state_history_count": 21, "state_query_hz": 50, "action_hz": 50, "action_horizon": 30,
                "future_group_ends_s": [.2, .4, .6], "history_visual_offsets_s": [-.4, -.2, 0],
                "observed_visual_mode": "current_multiview"}
    metadata = dict(_joint_metadata(), architecture={"fixture": "tiny", "measurement_dim": 29},
        temporal_abi=temporal, resolved_config_sha256=canonical_sha256(payload),
        protocol_registry_sha256=file_sha256(registry_paths["action_protocol_registry"]),
        measurement_registry_sha256=file_sha256(registry_paths["measurement_registry"]),
        optimization_contract={"target_encoder_ema_decay": source_training.target_encoder_ema_decay},
        normalization_id=module_state_sha256(source_modules["normalizer"]),
        loss_contract=predictive_world_loss_contract(source_training))
    checkpoint = tmp_path / "body_source"
    save_predictive_world_bundle(checkpoint, phase="joint", modules=source_modules,
                                  completed_updates=15000, metadata=metadata)
    source = read_predictive_world_bundle(checkpoint)
    target_registry = {**source_registry, REAL_SCHEMA: real_whole_body_schema(old, STATE_NAMES, mean[:41], std[:41])}
    target_path = tmp_path / "body_target_registry.json"
    target_path.write_text(json.dumps(target_registry))
    training = full_ert_training(source_training)
    training = replace(training, world_model=None, init_joint=str(checkpoint),
        init_source_config=str(source_config), max_updates=50000,
        init_body_target_migration=make_body_target_transition(source, payload, target_registry, schema=REAL_SCHEMA),
        init_objective_migration=make_objective_transition(source, payload, training, "FULL+ERT"),
        init_state_history_migration=current_state_transition(source, payload))
    config = PredictiveWorldConfig(data=replace(data, state_history_count=1, measurement_registry=str(target_path)),
        training=training, runtime=PredictiveWorldRuntimeConfig(output_dir=str(tmp_path / "new")))
    expected = dict(metadata, temporal_abi={**temporal, "state_history_count": 1},
        measurement_registry_sha256=file_sha256(target_path), loss_contract=predictive_world_loss_contract(training))
    report, _ = initialize_joint(config, target, expected)
    assert report["body_target_migration"] == training.init_body_target_migration
    assert report["objective_migration"] == training.init_objective_migration
    assert report["state_history_migration"] == training.init_state_history_migration
    assert report["optimizer_restored"] is False
    source_tensors = {f"{name}.{key}": value for name, module in source_modules.items()
                      for key, value in module.state_dict().items()}
    body_migrated = migrate_body_target_tensors(source_tensors)
    for name, module in target.items():
        for key, actual in module.state_dict().items():
            wanted = body_migrated[f"{name}.{key}"]
            if key in {"state_history_position", "state_position"}:
                wanted = wanted[-1:]
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
