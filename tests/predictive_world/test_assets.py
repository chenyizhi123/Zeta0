import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from psi.data.predictive_world.assets import load_asset_path_map, resolve_asset_path, validate_asset_path_map
from psi.data.predictive_world.cache import canonical_sha256
from psi.data.predictive_world.provenance import verify_qwen_provenance
from psi.data.lerobot.action_aligned_cache import build_action_aligned_vlm_processor_fingerprint
from psi.data.predictor_pretrain.runtime import fingerprint_model_directory


def test_asset_map_is_exact_portable_and_does_not_modify_source(tmp_path):
    target = tmp_path/'assets/weights'
    target.parent.mkdir()
    target.write_bytes(b'original')
    path = tmp_path/'map.json'
    path.write_text(json.dumps({'/source/weights': 'assets/weights'}))
    before = path.read_bytes()
    mapping = load_asset_path_map(path)
    assert mapping == {'/source/weights': str(target)}
    assert resolve_asset_path('/source/weights', mapping).read_bytes() == b'original'
    assert resolve_asset_path('/source/weights/other', mapping) == Path('/source/weights/other')
    assert path.read_bytes() == before
    assert load_asset_path_map(None) == {}


@pytest.mark.parametrize('mapping', [[], None, {'relative': '/target'}, {'/source': 'relative'},
                                     {'/source': 1}, {'/source/../escape': '/target'}])
def test_invalid_asset_maps_are_rejected(mapping):
    with pytest.raises(ValueError, match='asset_path_map'):
        validate_asset_path_map(mapping)


def identity_fixture(tmp_path):
    save_file({'weight': torch.arange(4)}, str(tmp_path/'model.safetensors'))
    (tmp_path/'config.json').write_text('{"model_type":"unit-test"}')
    # Test actual canonical fingerprinting/hash checks, not full Qwen execution.
    processor = SimpleNamespace(image_processor=SimpleNamespace(to_json_string=lambda: '{"size":32}'))
    fp = build_action_aligned_vlm_processor_fingerprint(processor)
    provenance = dict(encoder_id='qwen3vl:'+fingerprint_model_directory(tmp_path),
                      processor_id='qwen3vl-processor:'+canonical_sha256(fp),
                      feature_layer='post_merger_content_area_overlap_4x4_v1',
                      image_preprocessing='aspect_v1', dtype='float16', feature_shape=[16, 2048])
    return processor, provenance


def test_qwen_identity_checks_actual_weight_bytes_and_runtime(tmp_path, monkeypatch):
    processor, provenance = identity_fixture(tmp_path)
    report = verify_qwen_provenance(tmp_path, provenance, processor=processor)
    assert report['encoder_id'] == provenance['encoder_id']
    assert report['processor_id'] == provenance['processor_id']
    with monkeypatch.context() as changed:
        changed.setattr(torch, '__version__', 'different-build')
        with pytest.raises(ValueError, match='processor identity mismatch'):
            verify_qwen_provenance(tmp_path, provenance, processor=processor)
    save_file({'weight': torch.arange(4)+1}, str(tmp_path/'model.safetensors'))
    with pytest.raises(ValueError, match='encoder identity mismatch'):
        verify_qwen_provenance(tmp_path, provenance, processor=processor)


def test_old_processor_label_is_not_accepted_as_current_runtime(tmp_path):
    processor, provenance = identity_fixture(tmp_path)
    fp = build_action_aligned_vlm_processor_fingerprint(processor)
    fp.pop('image_runtime')
    fp['method'] = 'qwen_image_processor_json_sha256_v1'
    provenance['processor_id'] = 'qwen3vl-processor:'+canonical_sha256(fp)
    with pytest.raises(ValueError, match='do not relabel old artifacts'):
        verify_qwen_provenance(tmp_path, provenance, processor=processor)


def test_cache_cli_rejects_invalid_abi_before_index_or_output(tmp_path, monkeypatch):
    manifest = tmp_path/'bad.json'
    manifest.write_text(json.dumps({'feature_layer': 'wrong'}))
    output = tmp_path/'no-output'
    monkeypatch.setattr(sys, 'argv', ['cache', '--index', str(tmp_path/'absent-index'),
                                     '--provenance', str(manifest), '--model', str(tmp_path/'absent-model'),
                                     '--output', str(output)])
    script = Path(__file__).resolve().parents[2]/'scripts/data/cache_predictive_world_features.py'
    with pytest.raises(ValueError, match='implemented feature ABI'):
        runpy.run_path(str(script))['main']()
    assert not output.exists()


@pytest.mark.parametrize('deploy', [False, True])
def test_factory_checks_provenance_before_model_allocation(tmp_path, deploy):
    from psi.models.predictive_world.factory import build_phase, build_deploy_policy
    manifest = tmp_path/'bad.json'
    manifest.write_text(json.dumps({'provenance': {'feature_layer': 'wrong'}}))
    config = SimpleNamespace(model=SimpleNamespace(qwen_model='/absent/model', host_architecture='zeta_native'),
                             training=SimpleNamespace(init_processor_diagnostic=None, init_processor_migration=None),
                             runtime=SimpleNamespace(asset_path_map={}),
                             data=SimpleNamespace(feature_cache=str(manifest)))
    with pytest.raises(ValueError, match='implemented feature ABI'):
        if deploy:
            build_deploy_policy(config, tmp_path/'absent-bundle', device=torch.device('cpu'))
        else:
            build_phase(config, device=torch.device('cpu'))
