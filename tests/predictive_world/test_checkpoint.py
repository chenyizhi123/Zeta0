import json

import pytest
import torch
from torch import nn

from psi.config.predictive_world import PredictiveWorldTrainingConfig, predictive_world_loss_contract

from psi.models.predictive_world.checkpoint import (
    complete_predictive_world_bundle,
    load_predictive_world_bundle,
    predictive_world_bundle_identity,
    read_predictive_world_bundle,
    save_predictive_world_bundle,
)
from psi.models.predictive_world.representation import (
    EMAFutureTrajectoryEncoder,
    FutureTrajectoryEncoder,
)


def _metadata():
    return {
        "architecture": {"latent_shape": [3, 8, 256]},
        "temporal_abi": {"future_group_ends_s": [0.2, 0.4, 0.6]},
        "encoder_id": "encoder-hash",
        "processor_id": "processor-hash",
        "protocol_registry_sha256": "a" * 64,
        "measurement_registry_sha256": "b" * 64,
        "normalization_id": "normalizer-hash",
        "reference_id": "reference-hash",
        "data_fingerprint": "data-hash",
        "resolved_config_sha256": "config-hash",
        "code_revision": "revision",
        "code_fingerprint": "code-hash",
        "environment_fingerprint": "environment-hash",
    }


def _joint_metadata():
    return {
        **_metadata(),
        "parent_world_model": {
            "phase": "world_model",
            "completed_updates": 15_000,
            "weights_sha256": "c" * 64,
            "reference_id": "parent-reference-hash",
            "normalization_id": "parent-normalizer-hash",
        },
    }


def test_strict_forecast_bundle_roundtrip(tmp_path):
    source = {
        "history_state_adapters": nn.Linear(1, 2),
        "history_encoder": nn.Linear(2, 3),
        "latent_transition": nn.Linear(3, 4),
    }
    path = save_predictive_world_bundle(
        tmp_path / "forecast", phase="forecast", modules=source,
        completed_updates=17, metadata=_metadata(),
    )
    target = {
        "history_state_adapters": nn.Linear(1, 2),
        "history_encoder": nn.Linear(2, 3),
        "latent_transition": nn.Linear(3, 4),
    }
    sidecar = load_predictive_world_bundle(
        path.parent,
        phase="forecast",
        modules=target,
        expected_metadata={"reference_id": "reference-hash"},
    )
    assert sidecar["completed_updates"] == 17
    for name in source:
        for left, right in zip(source[name].parameters(), target[name].parameters()):
            torch.testing.assert_close(left, right)


def test_bundle_is_fail_closed_for_hash_metadata_and_phase(tmp_path):
    modules = {"student": nn.Linear(2, 3)}
    path = save_predictive_world_bundle(
        tmp_path / "policy", phase="policy", modules=modules,
        completed_updates=1, metadata=_metadata(),
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_predictive_world_bundle(
            path.parent, phase="policy", modules={"student": nn.Linear(2, 3)},
            expected_metadata={"reference_id": "wrong"},
        )
    payload = json.loads(path.read_text())
    payload["weights_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_predictive_world_bundle(path.parent)


@pytest.mark.parametrize("saved_scope", [None, "both", "prior"])
def test_objective_semantics_must_match_before_resume(tmp_path, saved_scope):
    contract = predictive_world_loss_contract(PredictiveWorldTrainingConfig())
    metadata = _metadata()
    if saved_scope is not None:
        metadata["loss_contract"] = {**contract, "forecast_control_scope": saved_scope}
    path = save_predictive_world_bundle(
        tmp_path / "objective", phase="policy", modules={"student": nn.Linear(2, 3)},
        completed_updates=1, metadata=metadata,
    )
    target = {"student": nn.Linear(2, 3)}
    before = target["student"].weight.detach().clone()
    if saved_scope == "prior":
        loaded = load_predictive_world_bundle(
            path.parent, phase="policy", modules=target, expected_metadata={"loss_contract": contract},
        )
        assert loaded["metadata"]["loss_contract"] == contract
    else:
        with pytest.raises(ValueError, match="loss_contract"):
            load_predictive_world_bundle(
                path.parent, phase="policy", modules=target, expected_metadata={"loss_contract": contract},
            )
        torch.testing.assert_close(target["student"].weight, before)


@pytest.mark.parametrize("legacy_format", [
    "zeta0.predictive_world", "zeta0.predictive_world.direct_transition",
])
def test_legacy_flow_bundle_is_rejected_before_weight_loading(tmp_path, legacy_format):
    path = save_predictive_world_bundle(
        tmp_path / "legacy", phase="policy", modules={"student": nn.Linear(2, 3)},
        completed_updates=1, metadata=_metadata(),
    )
    payload = json.loads(path.read_text())
    payload["format_id"] = legacy_format
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="format/schema mismatch"):
        read_predictive_world_bundle(path.parent)


def test_deploy_bundle_cannot_contain_training_teacher(tmp_path):
    with pytest.raises(ValueError, match="requires modules"):
        save_predictive_world_bundle(
            tmp_path / "bad-deploy",
            phase="deploy",
            modules={
                "history_encoder": nn.Linear(1, 1),
                "latent_transition": nn.Linear(1, 1),
                "latent_normalizer": nn.Linear(1, 1),
                "history_state_adapters": nn.Linear(1, 1),
                "student": nn.Linear(1, 1),
                "teacher": nn.Linear(1, 1),
            },
            completed_updates=0,
            metadata=_metadata(),
        )


def test_deferred_bundle_is_unreadable_until_explicit_completion(tmp_path):
    directory = tmp_path / "deferred"
    save_predictive_world_bundle(
        directory, phase="policy", modules={"student": nn.Linear(2, 3)},
        completed_updates=2, metadata=_metadata(), publish_complete=False,
    )
    with pytest.raises(FileNotFoundError, match="incomplete"):
        read_predictive_world_bundle(directory)
    complete_predictive_world_bundle(directory)
    assert read_predictive_world_bundle(directory)["completed_updates"] == 2


def test_world_model_bundle_requires_both_representation_and_dynamics(tmp_path):
    modules = {
        "encoder": nn.Linear(1, 1), "readout": nn.Linear(1, 1),
        "teacher": nn.Linear(1, 1), "normalizer": nn.Linear(1, 1),
        "history_state_adapters": nn.Linear(1, 1),
        "history_encoder": nn.Linear(1, 1), "latent_transition": nn.Linear(1, 1),
    }
    path = save_predictive_world_bundle(
        tmp_path / "world", phase="world_model", modules=modules,
        completed_updates=3, metadata=_metadata(),
    )
    assert read_predictive_world_bundle(path.parent)["phase"] == "world_model"


def test_joint_bundle_owns_updated_world_and_student(tmp_path):
    modules = {
        "encoder": nn.Linear(1, 1), "readout": nn.Linear(1, 1),
        "teacher": nn.Linear(1, 1), "normalizer": nn.Linear(1, 1),
        "history_state_adapters": nn.Linear(1, 1),
        "history_encoder": nn.Linear(1, 1), "latent_transition": nn.Linear(1, 1),
        "student": nn.Linear(1, 1),
    }
    path = save_predictive_world_bundle(
        tmp_path / "joint", phase="joint", modules=modules,
        completed_updates=20_000, metadata=_joint_metadata(),
    )
    sidecar = read_predictive_world_bundle(path.parent)
    assert sidecar["phase"] == "joint"
    assert sidecar["completed_updates"] == 20_000
    assert sidecar["metadata"]["parent_world_model"]["completed_updates"] == 15_000


def test_joint_bundle_rejects_missing_parent_provenance(tmp_path):
    modules = {
        "encoder": nn.Linear(1, 1), "readout": nn.Linear(1, 1),
        "teacher": nn.Linear(1, 1), "normalizer": nn.Linear(1, 1),
        "history_state_adapters": nn.Linear(1, 1),
        "history_encoder": nn.Linear(1, 1), "latent_transition": nn.Linear(1, 1),
        "student": nn.Linear(1, 1),
    }
    with pytest.raises(ValueError, match="parent_world_model"):
        save_predictive_world_bundle(
            tmp_path / "joint-without-parent", phase="joint", modules=modules,
            completed_updates=1, metadata=_metadata(),
        )


def test_bundle_identity_is_stable_and_covers_exact_weight_bytes(tmp_path):
    first = save_predictive_world_bundle(
        tmp_path / "first", phase="policy", modules={"student": nn.Linear(2, 3)},
        completed_updates=1, metadata=_metadata(),
    ).parent
    second_module = nn.Linear(2, 3)
    second_module.weight.data.add_(1)
    second = save_predictive_world_bundle(
        tmp_path / "second", phase="policy", modules={"student": second_module},
        completed_updates=1, metadata=_metadata(),
    ).parent
    assert predictive_world_bundle_identity(first) == predictive_world_bundle_identity(first)
    assert predictive_world_bundle_identity(first) != predictive_world_bundle_identity(second)


def test_world_bundle_roundtrip_preserves_ema_and_normalizer_stage(tmp_path):
    def encoder():
        return EMAFutureTrajectoryEncoder(FutureTrajectoryEncoder(
            visual_dim=6,
            measurement_dim=4,
            latent_shape=(3, 8, 4),
            hidden_dim=16,
            num_blocks=1,
            num_heads=4,
            ff_dim=32,
        ))

    source_encoder = encoder()
    source_encoder.update_target(12)
    source_encoder.mark_normalizer_update(12)
    source_encoder.freeze_normalizer_stage()
    source = {
        "encoder": source_encoder,
        "readout": nn.Linear(1, 1),
        "teacher": nn.Linear(1, 1),
        "normalizer": nn.Linear(1, 1),
        "history_state_adapters": nn.Linear(1, 1),
        "history_encoder": nn.Linear(1, 1),
        "latent_transition": nn.Linear(1, 1),
    }
    directory = save_predictive_world_bundle(
        tmp_path / "world-ema",
        phase="world_model",
        modules=source,
        completed_updates=13,
        metadata=_metadata(),
    ).parent
    target_encoder = encoder()
    target = {
        "encoder": target_encoder,
        **{name: nn.Linear(1, 1) for name in source if name != "encoder"},
    }
    load_predictive_world_bundle(directory, phase="world_model", modules=target)
    assert int(target_encoder.last_target_update) == 12
    assert int(target_encoder.last_normalizer_update) == 12
    assert bool(target_encoder.normalizer_frozen)


@pytest.mark.parametrize('saved_mode', ['history', 'current_multiview'])
def test_observation_mode_rejects_incompatible_bundle_before_loading(tmp_path, saved_mode):
    from dataclasses import replace
    from psi.config.predictive_world import PredictiveWorldDataConfig
    data = replace(PredictiveWorldDataConfig(), observed_visual_mode=saved_mode)
    metadata = _metadata()
    metadata['temporal_abi'].update(data.observation_contract)
    path = save_predictive_world_bundle(
        tmp_path / 'observation', phase='policy', modules={'student': nn.Linear(2, 3)},
        completed_updates=1, metadata=metadata,
    )
    other = replace(data, observed_visual_mode='current_multiview' if saved_mode == 'history' else 'history')
    expected = {**_metadata()['temporal_abi'], **other.observation_contract}
    target = {'student': nn.Linear(2, 3)}
    before = target['student'].weight.detach().clone()
    with pytest.raises(ValueError, match='metadata mismatch'):
        load_predictive_world_bundle(path.parent, phase='policy', modules=target,
                                     expected_metadata={'temporal_abi': expected})
    torch.testing.assert_close(before, target['student'].weight)


@pytest.mark.parametrize("revision", [1, 2])
def test_pre_rename_parallel_bundle_compatibility(tmp_path, revision):
    source = {"student": nn.Linear(2, 3)}
    path = save_predictive_world_bundle(
        tmp_path / "policy", phase="policy", modules=source,
        completed_updates=1, metadata=_metadata(),
    )
    payload = json.loads(path.read_text())
    payload["format_id"] = "oraclevla.predictive_world.parallel_process"
    payload["schema_revision"] = revision
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    target = {"student": nn.Linear(2, 3)}
    if revision == 1:
        load_predictive_world_bundle(path.parent, phase="policy", modules=target)
        torch.testing.assert_close(source["student"].weight, target["student"].weight)
        torch.testing.assert_close(source["student"].bias, target["student"].bias)
        assert path.read_bytes() == original
    else:
        with pytest.raises(ValueError, match="format/schema mismatch"):
            read_predictive_world_bundle(path.parent)
