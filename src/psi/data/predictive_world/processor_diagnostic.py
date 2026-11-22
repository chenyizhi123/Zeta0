"""Explicit, bounded processor-transition trials; never production acceptance."""
from __future__ import annotations

import json
from pathlib import Path

from psi.data.predictive_world.cache import canonical_sha256, file_sha256


def processor_diagnostic_contract(report_path):
    path = Path(report_path).resolve()
    report = json.loads(path.read_text())
    if (report.get('format') not in {'zeta0.qwen_gpu_diagnostic.v1', 'oraclevla.qwen_gpu_diagnostic.v1'}
            or report.get('historical_teacher_equivalence') is not False
            or report.get('authorizes_cache_reuse') is not False
            or report.get('optimizer_steps') != 0
            or report.get('latent', {}).get('normalizer_unchanged') is not True):
        raise ValueError('processor diagnostic requires the completed read-only GPU report')
    actual = report['actual_teacher']
    fingerprint = actual['processor_fingerprint']
    if fingerprint.get('method') != 'qwen_image_processor_json+image_runtime_v2':
        raise ValueError('diagnostic target must identify the actual v2 processor')
    target = 'qwen3vl-processor:' + canonical_sha256(fingerprint)
    legacy = {name: fingerprint[name] for name in (
        'processor_class', 'image_processor_class', 'transformers_version',
        'image_processor_config_sha256')}
    legacy['method'] = 'qwen_image_processor_json_sha256_v1'
    source = 'qwen3vl-processor:' + canonical_sha256(legacy)
    bundle = report['source_checkpoint_metadata']
    if (actual['processor_id'] != target
            or bundle['metadata']['processor_id'] != source
            or bundle['metadata']['encoder_id'] != actual['encoder_id']):
        raise ValueError('diagnostic source/target identities disagree with the report')
    return {
        'format': 'zeta0.processor_transition_preflight.v1',
        'report_path': str(path), 'report_sha256': file_sha256(path),
        'source_processor_id': source, 'target_processor_id': target,
        'source_weights_sha256': bundle['weights_sha256'],
        'source_config_sha256': report['source_config_sha256'],
        'target_processor_fingerprint': fingerprint,
        'historical_equivalence': False, 'production_accepted': False,
    }


def verify_processor_diagnostic(contract, source_bundle, target_processor_id, source_config):
    if not isinstance(contract, dict):
        raise ValueError('processor diagnostic contract must be an object')
    actual = processor_diagnostic_contract(contract['report_path'])
    if actual != contract:
        raise ValueError('processor diagnostic contract/report identity changed')
    report = json.loads(Path(contract['report_path']).read_text())
    if (source_bundle != report['source_checkpoint_metadata']
            or target_processor_id != contract['target_processor_id']
            or file_sha256(Path(source_config)) != contract['source_config_sha256']):
        raise ValueError('processor diagnostic is bound to a different source or target')
    return dict(contract)


def require_diagnostic_execution(config, *, preflight_only, updates, resume=None):
    if config.training.init_processor_diagnostic is None:
        return
    if (not preflight_only or resume or not 3 <= updates <= 50
            or config.runtime.world_size != 1
            or config.training.micro_batch > 2
            or config.training.gradient_accumulation_steps != 1):
        raise ValueError('processor transition is diagnostic-only: fresh preflight, 3..50 updates, single GPU, batch <= 2')


def diagnostic_anchor_selection(episodes, count, *, episodes_per_split=1):
    """Prespecified small complete windows, keeping the original episode splits."""
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError('diagnostic anchors per split must be in 1..32')
    if type(episodes_per_split) is not int or not 1 <= episodes_per_split <= 4:
        raise ValueError('diagnostic episodes per split must be in 1..4')
    frames = list(range(20, 20 + count * 5, 5))
    selected = {}
    for split in ('train', 'validation'):
        eligible = sorted((ep for ep in episodes if ep['split'] == split
                           and len(ep['times']) > frames[-1] + 30), key=lambda ep: ep['episode'])
        if len(eligible) < episodes_per_split:
            raise ValueError(f'not enough {split} episodes support the fixed complete-window diagnostic')
        for ep in eligible[:episodes_per_split]:
            selected[str(ep['episode'])] = frames
    if len(selected) != 2 * episodes_per_split:
        raise ValueError('diagnostic splits must use distinct episodes')
    return {'version': ('diagnostic_fixed_complete_windows_v1' if episodes_per_split == 1
                        else 'diagnostic_multi_episode_complete_windows_v2'),
            'first_anchor': 20, 'stride': 5, 'count_per_split': count * episodes_per_split,
            'episode_frames': selected, 'statistics': 'unchanged_full_train_split'}


def unique_diagnostic_rows(stream, count, *, max_draws=256):
    """Bounded, explicitly unique validation coverage, not occurrence weighting."""
    from itertools import islice
    if not 1 <= count <= max_draws:
        raise ValueError('invalid unique diagnostic sample budget')
    selected = {}
    for row in islice(stream, max_draws):
        selected.setdefault(row['sample_id'], row)
        if len(selected) == count:
            return list(selected.values())
    raise ValueError('diagnostic sampler did not provide enough unique validation windows')
