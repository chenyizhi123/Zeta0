import copy

import pytest
import torch
from torch import nn

from psi.data.predictor_pretrain.state_adapter import StateAdapterBank
from psi.models.predictive_world.action_expert import TemporalNativeActionExpert
from psi.models.predictive_world.latent_transition import TemporalLatentTransition
from psi.models.predictive_world.initialization import transfer_joint_tensors
from psi.models.predictive_world.checkpoint import prune_completed_checkpoints


def test_portable_export_binds_state_registry_config_and_weights_without_original_index(tmp_path):
    from psi.data.predictive_world.cache import file_sha256
    from psi.models.predictive_world.initialization import exported_state_registry_digest

    config = tmp_path / "config/train_config.json"
    state = tmp_path / "config/state_schema_registry.json"
    checkpoint = tmp_path / "checkpoint-00029000"
    config.parent.mkdir()
    checkpoint.mkdir()
    config.write_text('{"source":"pretraining"}')
    state.write_text('{"state":{"dim":3}}')
    sidecar = checkpoint / "predictive_world_checkpoint.json"
    sidecar.write_text('{"phase":"joint"}')
    source = dict(weights_file="predictive_world.safetensors", weights_sha256="a" * 64)
    assert exported_state_registry_digest(checkpoint, config, state, source) is None
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text("".join(f"{file_sha256(p)}  {p.relative_to(tmp_path)}\n" for p in (config, state, sidecar))
        + "a" * 64 + "  checkpoint-00029000/predictive_world.safetensors\n")
    assert exported_state_registry_digest(checkpoint, config, state, source) == file_sha256(state)
    for path in (state, config, sidecar):
        original = path.read_text()
        path.write_text(original + " ")
        with pytest.raises(ValueError, match="identity mismatch"):
            exported_state_registry_digest(checkpoint, config, state, source)
        path.write_text(original)
    with pytest.raises(ValueError, match="checkpoint weights"):
        exported_state_registry_digest(checkpoint, config, state, {**source, "weights_sha256": "b" * 64})


def modules(protocols, states):
    def expert():
        return TemporalNativeActionExpert(protocols, {k: v['dim'] for k, v in states.items()},
            qwen_dim=6, context_tokens=2, latent_dim=4, latent_groups=3, latent_slots=8,
            state_history_count=5, action_steps=4, hidden_dim=16, num_blocks=1, num_heads=4, ff_dim=32)
    return dict(encoder=nn.Linear(2, 2), normalizer=nn.Linear(2, 2),
                readout=nn.Linear(2, 2), history_encoder=nn.Linear(2, 2),
                history_state_adapters=StateAdapterBank({k: v['dim'] for k, v in states.items()}),
                teacher=expert(), student=expert(),
                latent_transition=TemporalLatentTransition(protocols, context_dim=16,
                    latent_shape=(3,8,4), hidden_dim=16, num_blocks=1, num_heads=4, ff_dim=32))


def setup_transfer():
    source_protocols = {'z_old': {'dim': 2, 'mean': [0,0], 'std': [1,1]}}
    target_protocols = {**source_protocols, 'a_new': {'dim': 3, 'mean': [0]*3, 'std': [1]*3}}
    source_states = {'old_state': {'dim': 3}}
    target_states = {**source_states, 'new_state': {'dim': 4}}
    source, target = modules(source_protocols, source_states), modules(target_protocols, target_states)
    tensors = {f'{name}.{key}': value.clone() for name, module in source.items() for key, value in module.state_dict().items()}
    kwargs = dict(source_protocols=source_protocols, target_protocols=target_protocols,
                  source_states=source_states, target_states=target_states)
    return source, target, tensors, kwargs


def test_joint_transfer_preserves_student_and_remaps_embedding_by_protocol_name():
    source, target, tensors, kwargs = setup_transfer()
    new_student = target['student'].adapters['a_new'].input[0].weight.clone()
    new_embedding = target['student'].protocol_embedding.weight[0].clone()
    report = transfer_joint_tensors(tensors, target, **kwargs)
    assert report['new_protocols'] == ['a_new']
    assert report['new_state_schemas'] == ['new_state']
    for name in ('teacher', 'student'):
        torch.testing.assert_close(target[name].protocol_embedding.weight[1], source[name].protocol_embedding.weight[0])
        torch.testing.assert_close(target[name].blocks[0].linear1.weight, source[name].blocks[0].linear1.weight)
    torch.testing.assert_close(target['latent_transition'].protocol_embedding.weight[1],
                               source['latent_transition'].protocol_embedding.weight[0])
    assert not torch.equal(target['teacher'].blocks[0].linear1.weight, target['student'].blocks[0].linear1.weight)
    torch.testing.assert_close(target['student'].adapters['a_new'].input[0].weight, new_student)
    torch.testing.assert_close(target['student'].protocol_embedding.weight[0], new_embedding)
    for name in ('encoder', 'normalizer', 'readout', 'history_encoder'):
        torch.testing.assert_close(target[name].weight, source[name].weight)


def test_new_adapter_initialization_hash_pairs_same_seed_and_keeps_shared_tensors():
    reports = []
    for seed in (17, 17, 18):
        torch.manual_seed(seed)
        source, target, tensors, kwargs = setup_transfer()
        report = transfer_joint_tensors(tensors, target, **kwargs)
        reports.append(report["new_adapter_initialization_sha256"])
        for name in ("teacher", "student"):
            torch.testing.assert_close(target[name].blocks[0].linear1.weight,
                                       source[name].blocks[0].linear1.weight, rtol=0, atol=0)
    assert reports[0] == reports[1]
    assert reports[0] != reports[2]


@pytest.mark.parametrize('damage', ['missing', 'shape', 'registry', 'student'])
def test_invalid_joint_transfer_fails_before_changing_weights(damage):
    _, target, tensors, kwargs = setup_transfer()
    before = copy.deepcopy(target['encoder'].state_dict())
    if damage == 'missing':
        tensors.pop('encoder.weight')
    elif damage == 'shape':
        tensors['student.protocol_embedding.weight'] = torch.zeros(2, 16)
    elif damage == 'registry':
        kwargs['target_protocols']['z_old'] = {'dim': 1}
    else:
        target.pop('student')
    with pytest.raises(ValueError):
        transfer_joint_tensors(tensors, target, **kwargs)
    for key, value in target['encoder'].state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_retention_preserves_partial_and_external_checkpoints(tmp_path):
    for step in (20, 5000, 10000):
        path = tmp_path / f'checkpoint-{step:08d}'
        path.mkdir()
        (path/'.complete').touch()
        (path/'.trainer_complete').touch()
    partial = tmp_path / 'checkpoint-00015000'
    partial.mkdir()
    external = tmp_path / 'source'
    external.mkdir()
    (tmp_path/'checkpoint-00000001').symlink_to(external, target_is_directory=True)
    removed = prune_completed_checkpoints(tmp_path, keep=2)
    assert removed == [str(tmp_path/'checkpoint-00000020')]
    assert partial.exists() and external.exists()


def test_relocated_source_keeps_config_and_registry_identity(tmp_path):
    import json
    from psi.data.predictive_world.cache import canonical_sha256, file_sha256
    from psi.models.predictive_world.checkpoint import PHASE_MODULES, save_predictive_world_bundle
    from psi.models.predictive_world.initialization import read_initialization_source
    from test_checkpoint import _joint_metadata

    source = {'data': {}}
    mapping = {}
    hashes = {}
    for key in ('action_protocol_registry', 'state_schema_registry', 'measurement_registry'):
        original = '/unavailable/source/'+key+'.json'
        actual = tmp_path/(key+'.json')
        actual.write_text(json.dumps({'old': {'dim': 3}}))
        source['data'][key] = original
        mapping[original] = str(actual)
        hashes[key] = file_sha256(actual)
    source['data']['index_path'] = '/unavailable/source/index.jsonl'
    meta = tmp_path/'index.meta.json'
    meta.write_text(json.dumps({'provenance': {'registry_binding': {
        'state_schema_registry_sha256': hashes['state_schema_registry']}}}))
    mapping[source['data']['index_path']+'.meta.json'] = str(meta)
    config = tmp_path/'source-config.json'
    config.write_text(json.dumps(source))
    before = config.read_bytes()
    metadata = _joint_metadata()
    metadata.update(resolved_config_sha256=canonical_sha256(source),
                    protocol_registry_sha256=hashes['action_protocol_registry'],
                    measurement_registry_sha256=hashes['measurement_registry'])
    checkpoint = tmp_path/'checkpoint'
    save_predictive_world_bundle(checkpoint, phase='joint', completed_updates=1,
                                modules={name: nn.Linear(1, 1) for name in PHASE_MODULES['joint']},
                                metadata=metadata)
    with pytest.raises(FileNotFoundError):
        read_initialization_source(checkpoint, config)
    _, loaded, registries = read_initialization_source(checkpoint, config, asset_path_map=mapping)
    assert loaded == source and config.read_bytes() == before
    assert registries['state_schema_registry'] == {'old': {'dim': 3}}
    (tmp_path/'state_schema_registry.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='state registry identity mismatch'):
        read_initialization_source(checkpoint, config, asset_path_map=mapping)
