import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.models.predictive_world.vlm_action_transition import (
    make_vlm_action_transition, verify_vlm_action_transition, migrate_action_position_tensors,
)


def contracts():
    temporal = dict(history_visual_offsets_s=[-.4, -.2, 0], state_history_count=21, state_query_hz=50,
                    action_horizon=30, action_hz=50, future_group_ends_s=[.2, .4, .6])
    payload = {"data": {"index_path": "source"}}
    source = dict(phase="joint", weights_sha256="source", metadata=dict(
        resolved_config_sha256=canonical_sha256(payload), temporal_abi=temporal,
        encoder_id="qwen3vl:original", processor_id="qwen3vl-processor:original",
        architecture=dict(qwen_model="original", test="tiny")))
    expected = dict(temporal_abi={**temporal, "state_history_count": 1, "action_horizon": 50, "action_hz": 30},
        encoder_id="qwen3vl:psi0", processor_id="qwen3vl-processor:psi0",
        architecture=dict(qwen_model="psi0", test="tiny"))
    return source, payload, expected


def record(source, payload, expected):
    return make_vlm_action_transition(source, payload, encoder_id=expected["encoder_id"],
        processor_id=expected["processor_id"], temporal=expected["temporal_abi"],
        qwen_model=expected["architecture"]["qwen_model"])


@pytest.mark.parametrize("damage", ["source", "encoder", "processor", "groups", "architecture", "normalizer"])
def test_vlm_action_transfer_rejects_unbound_changes(damage):
    source, payload, expected = contracts()
    migration = record(source, payload, expected)
    if damage == "source": source["weights_sha256"] = "other"
    elif damage == "encoder": expected["encoder_id"] = "qwen3vl:other"
    elif damage == "processor": expected["processor_id"] = "qwen3vl-processor:other"
    elif damage == "groups": expected["temporal_abi"]["future_group_ends_s"] = [.3, .6, .9]
    elif damage == "architecture": expected["architecture"]["test"] = "big"
    else: migration["preserve_pretrained_latent_normalizer"] = False
    with pytest.raises(ValueError): verify_vlm_action_transition(migration, source, payload, expected)


def test_position_resize_changes_only_two_tables_preserves_endpoints_and_source():
    source, payload, expected = contracts()
    migration = record(source, payload, expected)
    assert verify_vlm_action_transition(migration, source, payload, expected) == migration
    tensors = {name+".action_position": torch.arange(30.).reshape(30, 1).expand(30, 4).clone()
               for name in ("teacher", "student")}
    tensors["normalizer.mean"] = torch.randn(3, 8, 4)
    before = copy.deepcopy(tensors)
    result = migrate_action_position_tensors(tensors, migration)
    for name in ("teacher", "student"):
        value = result[name+".action_position"]
        assert value.shape == (50, 4)
        torch.testing.assert_close(value[:, 0], torch.linspace(0, 29, 50))
    assert result["normalizer.mean"] is tensors["normalizer.mean"]
    for key in tensors: torch.testing.assert_close(tensors[key], before[key], rtol=0, atol=0)


@pytest.mark.parametrize("same_vlm", [False, True])
def test_full_strict_joint_initialization_composes_vlm_grid_and_current_state(tmp_path, same_vlm):
    from test_current_state import small_modules
    from test_checkpoint import _joint_metadata
    from psi.models.predictive_world.checkpoint import save_predictive_world_bundle
    from psi.models.predictive_world.initialization import initialize_joint
    from psi.models.predictive_world.latent_transition import LatentNormalizer
    from psi.models.predictive_world.state_history_transition import current_state_transition
    from psi.models.predictive_world.training import module_state_sha256

    source_modules, protocols, states = small_modules(21)
    destination, _, _ = small_modules(1)
    for name in ("teacher", "student"):
        destination[name].action_position = nn.Parameter(torch.randn(50, 16))
        destination[name].action_steps = 50
    for modules in (source_modules, destination):
        modules["normalizer"] = LatentNormalizer((3, 8, 4))
        modules["normalizer"].fitted.fill_(True)
        modules["encoder"] = nn.Linear(2, 2)
        modules["encoder"].register_buffer("normalizer_frozen", torch.tensor(True))
        modules["encoder"].register_buffer("last_target_update", torch.tensor(14999))
    source, payload, expected = contracts()
    if same_vlm:
        expected.update(encoder_id=source["metadata"]["encoder_id"],
                        processor_id=source["metadata"]["processor_id"],
                        architecture=source["metadata"]["architecture"])
        expected["temporal_abi"]["action_hz"] = 50
    for name, value in (("action_protocol_registry", protocols), ("state_schema_registry", states),
                        ("measurement_registry", {"measured": dict(width=41)})):
        path = tmp_path / (name+".json"); path.write_text(json.dumps(value)); payload["data"][name] = str(path)
    payload["data"]["index_path"] = str(tmp_path / "index.jsonl")
    (tmp_path / "index.jsonl.meta.json").write_text(json.dumps(dict(provenance=dict(registry_binding=dict(
        state_schema_registry_sha256=file_sha256(tmp_path / "state_schema_registry.json"))))))
    config_path = tmp_path / "source.json"; config_path.write_text(json.dumps(payload))
    metadata = {**_joint_metadata(), **source["metadata"], "resolved_config_sha256": canonical_sha256(payload),
        "optimization_contract": dict(target_encoder_ema_decay=.99),
        "protocol_registry_sha256": file_sha256(tmp_path / "action_protocol_registry.json"),
        "measurement_registry_sha256": file_sha256(tmp_path / "measurement_registry.json"),
        "normalization_id": module_state_sha256(source_modules["normalizer"]), "loss_contract": {"same": True}}
    checkpoint = tmp_path / "joint"
    sidecar = save_predictive_world_bundle(checkpoint, phase="joint", modules=source_modules,
        completed_updates=15000, metadata=metadata)
    source = json.loads(sidecar.read_text())
    migration = record(source, payload, expected)
    training = SimpleNamespace(init_joint=str(checkpoint), init_source_config=str(config_path),
        init_processor_diagnostic=None, init_processor_migration=None, init_objective_migration=None,
        init_state_history_migration=current_state_transition(source, payload), init_observation_contract_migration=None,
        init_body_target_migration=None, init_vlm_action_migration=migration, target_encoder_ema_decay=.99)
    data = SimpleNamespace(**{k:payload["data"][k] for k in ("action_protocol_registry", "state_schema_registry", "measurement_registry")})
    config = SimpleNamespace(training=training, data=data, runtime=SimpleNamespace(asset_path_map={}))
    expected["loss_contract"] = metadata["loss_contract"]
    report, _ = initialize_joint(config, destination, expected)
    assert report["vlm_action_migration"] == migration and report["optimizer_restored"] is False
    resized = migrate_action_position_tensors({f"{name}.{k}":v for name,m in source_modules.items() for k,v in m.state_dict().items()}, migration)
    for name, module in destination.items():
        for key, value in module.state_dict().items():
            original = resized[f"{name}.{key}"]
            if key in {"state_position", "state_history_position"}: original = original[-1:]
            torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_50_step_policy_sampling_preserves_15_step_rtc_prefix():
    from test_action_expert import _expert, _inputs
    from psi.data.predictive_world.contracts import NativeActionBatch

    expert = _expert()
    expert.action_steps = 50
    expert.action_position = nn.Parameter(torch.randn(50, expert.hidden_dim))
    inputs = _inputs(batch=1)
    # Binary-grid native commands round-trip exactly through toy affine stats.
    values = torch.randint(-8, 9, (1, 50, 2)).float() / 8
    valid = torch.zeros_like(values, dtype=torch.bool)
    valid[:, :15] = True
    prefix = NativeActionBatch(values=values, valid=valid, times=torch.arange(1, 51)[None]/30,
        protocol_ids=("toy",), native_dims=torch.tensor([2]), normalization_spaces=("native",))
    result = expert.sample(**inputs, noise=torch.randn_like(values), committed_prefix=prefix, steps=10)
    assert result.shape == (1, 50, 2) and torch.isfinite(result).all()
    torch.testing.assert_close(result[:, :15], values[:, :15], rtol=0, atol=0)


def test_same_vlm_action_grid_migration_is_explicit_and_bound():
    source, payload, expected = contracts()
    expected.update(encoder_id=source["metadata"]["encoder_id"],
                    processor_id=source["metadata"]["processor_id"],
                    architecture=source["metadata"]["architecture"])
    expected["temporal_abi"]["action_hz"] = 50
    migration = record(source, payload, expected)
    assert migration["format"] == "zeta0.predictive_world.action_grid_initialization.v1"
    assert verify_vlm_action_transition(migration, source, payload, expected) == migration
    expected["processor_id"] = "qwen3vl-processor:changed"
    with pytest.raises(ValueError, match="unchanged VLM"):
        record(source, payload, expected)


@pytest.mark.parametrize('damage', [None, 'missing', 'report', 'target', 'encoder', 'source_config'])
def test_grid_change_composes_verified_processor_evidence_and_relocation(tmp_path, damage):
    from test_processor_diagnostic import fixture
    from psi.data.predictive_world.processor_migration import make_processor_migration

    report_path, source_config, source, target_processor = fixture(tmp_path)
    template, _, _ = contracts()
    payload = json.loads(source_config.read_text())
    source['phase'] = 'joint'
    source['metadata'].update(
        resolved_config_sha256=canonical_sha256(payload),
        temporal_abi=template['metadata']['temporal_abi'],
        architecture=dict(qwen_model='/old/qwen', test='tiny'))
    report = json.loads(report_path.read_text())
    report['format'] = 'oraclevla.qwen_gpu_diagnostic.v1'
    report['source_checkpoint_metadata'] = source
    report_path.write_text(json.dumps(report))
    migration = make_processor_migration(report_path, accepted_by='test user', acceptance_record='test approval')
    expected = dict(encoder_id=source['metadata']['encoder_id'], processor_id=target_processor,
        temporal_abi={**source['metadata']['temporal_abi'], 'state_history_count': 1,
                      'action_horizon': 50, 'action_hz': 30},
        architecture=dict(qwen_model='/relocated/qwen', test='tiny'))
    args = dict(encoder_id=expected['encoder_id'], processor_id=target_processor,
        temporal=expected['temporal_abi'], qwen_model='/relocated/qwen',
        processor_migration=migration, source_config=source_config)
    record = make_vlm_action_transition(source, payload, **args)
    assert record['source_encoder_id'] == record['target_encoder_id']
    assert record['processor_migration'] == migration
    if damage == 'missing': migration = None
    elif damage == 'report': report_path.write_text(report_path.read_text()+'\n')
    elif damage == 'target': expected['processor_id'] = 'qwen3vl-processor:wrong'
    elif damage == 'encoder': expected['encoder_id'] = 'qwen3vl:other'
    elif damage == 'source_config': source_config.write_text('{"changed":true}')
    if damage is not None:
        with pytest.raises(ValueError):
            verify_vlm_action_transition(record, source, payload, expected,
                processor_migration=migration, source_config=source_config)
    else:
        assert verify_vlm_action_transition(record, source, payload, expected,
            processor_migration=migration, source_config=source_config) == record


def test_config_accepts_only_source_bound_processor_and_grid_composition(tmp_path):
    from dataclasses import replace
    from psi.config.predictive_world import PredictiveWorldTrainingConfig
    migration = {"format": "evidence-is-verified-at-initialization"}
    transition = dict(format="zeta0.predictive_world.action_grid_initialization.v1",
        source_encoder_id="qwen3vl:original", target_encoder_id="qwen3vl:original",
        processor_migration=migration)
    training = PredictiveWorldTrainingConfig(phase="joint", init_joint="source",
        init_source_config="source.json", init_processor_migration=migration,
        init_vlm_action_migration=transition, normalizer_calibration_updates=0,
        forecast_warmup_updates=0, forecast_ramp_updates=0, micro_batch=64,
        effective_global_batch=256, max_updates=20000)
    training.validate(world_size=4)
    for field,value in [("target_encoder_id", "qwen3vl:other"), ("processor_migration", {}),
                        ("format", "zeta0.predictive_world.vlm_action_initialization.v1")]:
        with pytest.raises(ValueError, match="bound processor evidence"):
            replace(training, init_vlm_action_migration={**transition, field:value}).validate(world_size=4)
