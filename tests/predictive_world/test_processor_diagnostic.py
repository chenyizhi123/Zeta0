import copy
import json
from types import SimpleNamespace

import pytest
import torch

from psi.data.predictive_world.cache import canonical_sha256, file_sha256
from psi.data.predictive_world.processor_diagnostic import (
    processor_diagnostic_contract, verify_processor_diagnostic,
    require_diagnostic_execution, diagnostic_anchor_selection,
)
from psi.models.predictive_world.factory import build_phase, build_deploy_policy


def fixture(tmp_path):
    fp = {'method': 'qwen_image_processor_json+image_runtime_v2',
          'processor_class': 'Qwen3VLProcessor', 'image_processor_class': 'Qwen2VLImageProcessorFast',
          'transformers_version': '4.57.3', 'image_processor_config_sha256': 'fixture',
          'image_runtime': {'torch': 'fixture', 'pillow': 'fixture', 'numpy': 'fixture'}}
    old = {k: v for k, v in fp.items() if k != 'image_runtime'}
    old['method'] = 'qwen_image_processor_json_sha256_v1'
    source_id = 'qwen3vl-processor:' + canonical_sha256(old)
    target_id = 'qwen3vl-processor:' + canonical_sha256(fp)
    config = tmp_path/'source.json'
    config.write_text('{}')
    bundle = {'metadata': {'processor_id': source_id, 'encoder_id': 'qwen3vl:fixture'},
              'weights_sha256': 'weights'}
    report = {'format': 'zeta0.qwen_gpu_diagnostic.v1', 'historical_teacher_equivalence': False,
              'authorizes_cache_reuse': False, 'optimizer_steps': 0,
              'latent': {'normalizer_unchanged': True}, 'source_checkpoint_metadata': bundle,
              'source_config_sha256': file_sha256(config),
              'actual_teacher': {'processor_fingerprint': fp, 'processor_id': target_id,
                                 'encoder_id': 'qwen3vl:fixture'}}
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report))
    return path, config, bundle, target_id


def test_exact_diagnostic_pair_is_explicit_and_not_production_approval(tmp_path):
    path, config, bundle, target = fixture(tmp_path)
    contract = processor_diagnostic_contract(path)
    result = verify_processor_diagnostic(contract, bundle, target, config)
    assert result == contract
    assert result['production_accepted'] is False
    assert result['historical_equivalence'] is False


@pytest.mark.parametrize('damage', ['report', 'source', 'target', 'config', 'contract'])
def test_diagnostic_rejects_changed_bindings(tmp_path, damage):
    path, config, bundle, target = fixture(tmp_path)
    contract = processor_diagnostic_contract(path)
    if damage == 'report':
        path.write_text(path.read_text() + '\n')
    elif damage == 'source':
        bundle = copy.deepcopy(bundle)
        bundle['weights_sha256'] = 'other'
    elif damage == 'target':
        target = 'other'
    elif damage == 'config':
        config.write_text('{"changed":true}')
    else:
        contract['production_accepted'] = True
    with pytest.raises(ValueError):
        verify_processor_diagnostic(contract, bundle, target, config)


@pytest.mark.parametrize('field', ['transformers_version', 'image_processor_config_sha256'])
def test_diagnostic_rejects_changed_recorded_legacy_fields(tmp_path, field):
    path, *_ = fixture(tmp_path)
    report = json.loads(path.read_text())
    report['actual_teacher']['processor_fingerprint'][field] = 'changed'
    report['actual_teacher']['processor_id'] = 'qwen3vl-processor:' + canonical_sha256(
        report['actual_teacher']['processor_fingerprint'])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        processor_diagnostic_contract(path)


def config_fixture():
    return SimpleNamespace(training=SimpleNamespace(init_processor_diagnostic={}, micro_batch=2,
                                                    gradient_accumulation_steps=1),
                           runtime=SimpleNamespace(world_size=1))


@pytest.mark.parametrize('damage', ['normal_train', 'too_long', 'resume', 'batch', 'world', 'accumulation'])
def test_diagnostic_execution_is_bounded(damage):
    config = config_fixture()
    args = dict(preflight_only=True, updates=20)
    if damage == 'normal_train':
        args['preflight_only'] = False
    elif damage == 'too_long':
        args['updates'] = 51
    elif damage == 'resume':
        args['resume'] = 'checkpoint'
    elif damage == 'batch':
        config.training.micro_batch = 3
    elif damage == 'world':
        config.runtime.world_size = 2
    else:
        config.training.gradient_accumulation_steps = 2
    with pytest.raises(ValueError):
        require_diagnostic_execution(config, **args)


def test_default_factory_and_deploy_reject_diagnostic_before_allocation():
    config = config_fixture()
    with pytest.raises(ValueError, match='diagnostic'):
        build_phase(config, device=torch.device('cpu'))
    with pytest.raises(ValueError, match='diagnostic'):
        build_deploy_policy(config, 'nonexistent', device=torch.device('cpu'))
    require_diagnostic_execution(config, preflight_only=True, updates=20)
    require_diagnostic_execution(config, preflight_only=True, updates=50)


def test_diagnostic_anchors_keep_split_and_fixed_selection():
    episodes = [dict(episode=2, split='train', times=range(120)),
                dict(episode=1, split='train', times=range(120)),
                dict(episode=3, split='validation', times=range(120))]
    selection = diagnostic_anchor_selection(episodes, 8)
    assert selection['episode_frames'] == {'1': list(range(20, 60, 5)), '3': list(range(20, 60, 5))}
    assert selection['statistics'] == 'unchanged_full_train_split'
    with pytest.raises(ValueError):
        diagnostic_anchor_selection(episodes[:2], 8)
    with pytest.raises(ValueError):
        diagnostic_anchor_selection(episodes, 0)


def test_multiple_diagnostic_episodes_keep_original_splits():
    episodes = [dict(episode=i, split='train' if i < 4 else 'validation', times=range(120))
                for i in range(8)]
    selection = diagnostic_anchor_selection(episodes, 8, episodes_per_split=4)
    assert selection['count_per_split'] == 32
    assert len(selection['episode_frames']) == 8
    assert selection['version'] == 'diagnostic_multi_episode_complete_windows_v2'
    with pytest.raises(ValueError):
        diagnostic_anchor_selection(episodes[:-1], 8, episodes_per_split=4)


def test_unique_validation_is_bounded_and_does_not_count_duplicate_draws():
    from psi.data.predictive_world.processor_diagnostic import unique_diagnostic_rows
    rows = [dict(sample_id=name) for name in ('a', 'a', 'b')]
    assert unique_diagnostic_rows(iter(rows), 2) == [rows[0], rows[2]]
    with pytest.raises(ValueError):
        unique_diagnostic_rows(iter(rows), 2, max_draws=2)
