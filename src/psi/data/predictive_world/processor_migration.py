"""Explicit research post-training approval, without historical relabeling."""
from __future__ import annotations

import json
from pathlib import Path

from psi.data.predictive_world.processor_diagnostic import (
    processor_diagnostic_contract, verify_processor_diagnostic,
)


FORMAT = 'zeta0.processor_transition_post_training.v1'


def make_processor_migration(
    report_path,
    *,
    accepted_by,
    acceptance_record,
    max_effective_batch=256,
):
    if not accepted_by.strip() or not acceptance_record.strip():
        raise ValueError('migration requires an explicit approval attribution and record')
    max_effective_batch = int(max_effective_batch)
    if max_effective_batch <= 0:
        raise ValueError('migration max_effective_batch must be positive')
    return {
        'format': FORMAT,
        'diagnostic': processor_diagnostic_contract(report_path),
        'accepted_by': accepted_by, 'acceptance_record': acceptance_record,
        'scope': 'research_post_training',
        'historical_equivalence': False, 'production_accepted': False,
        'reuse_source_cache': False, 'max_updates': 100000,
        'max_effective_batch': max_effective_batch,
    }


def validate_processor_migration(contract):
    if not isinstance(contract, dict) or contract.get('format') != FORMAT:
        raise ValueError('invalid processor post-training migration format')
    expected = make_processor_migration(
        contract['diagnostic']['report_path'], accepted_by=contract['accepted_by'],
        acceptance_record=contract['acceptance_record'],
        max_effective_batch=contract.get('max_effective_batch', 256))
    if contract != expected:
        raise ValueError('processor migration contract/evidence changed')
    return dict(contract)


def load_processor_migration(path):
    return validate_processor_migration(json.loads(Path(path).read_text()))


def verify_processor_migration(contract, source_bundle, target_processor_id, source_config):
    contract = validate_processor_migration(contract)
    verify_processor_diagnostic(contract['diagnostic'], source_bundle,
                                target_processor_id, source_config)
    return contract


def require_migration_budget(config):
    contract = config.training.init_processor_migration
    if contract is None:
        return
    contract = validate_processor_migration(contract)
    if (config.training.max_updates > contract['max_updates']
            or config.training.effective_global_batch > contract['max_effective_batch']):
        raise ValueError('training budget exceeds the approved processor migration')


def migration_target_provenance(provenance, contract, source_bundle, source_config):
    """Specify a NEW cache; do not mutate or reuse the source manifest/cache."""
    target = contract['diagnostic']['target_processor_id']
    verify_processor_migration(contract, source_bundle, target, source_config)
    if (provenance.get('processor_id') != contract['diagnostic']['source_processor_id']
            or provenance.get('encoder_id') != source_bundle['metadata']['encoder_id']):
        raise ValueError('source cache identity differs from the reviewed migration')
    result = {key: value for key, value in provenance.items() if key in {
        'encoder_id', 'processor_id', 'image_preprocessing', 'feature_layer', 'dtype', 'feature_shape'}}
    result.update(processor_id=target, processor_migration=contract, video_decoder='torchcodec')
    return result
