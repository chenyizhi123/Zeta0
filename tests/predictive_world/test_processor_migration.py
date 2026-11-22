import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from psi.config.predictive_world import PredictiveWorldTrainingConfig
from psi.data.predictive_world.processor_migration import (
    make_processor_migration, verify_processor_migration, validate_processor_migration,
    migration_target_provenance, require_migration_budget,
)
from psi.models.predictive_world.factory import build_deploy_policy
from test_processor_diagnostic import fixture


def contract_fixture(tmp_path):
    path, config, bundle, target = fixture(tmp_path)
    contract = make_processor_migration(path, accepted_by='test user', acceptance_record='test approval')
    return contract, config, bundle, target


def test_research_migration_preserves_source_and_forces_fresh_cache(tmp_path):
    contract, config, bundle, target = contract_fixture(tmp_path)
    assert verify_processor_migration(contract, bundle, target, config) == contract
    old = dict(bundle['metadata'], dtype='float16', image_preprocessing='aspect_v1',
               feature_layer='post_merger_content_area_overlap_4x4_v1', cache_identity='old-cache')
    before = copy.deepcopy(old)
    new = migration_target_provenance(old, contract, bundle, config)
    assert old == before
    assert new['processor_id'] == target
    assert new['processor_migration'] == contract
    assert 'cache_identity' not in new
    assert contract['historical_equivalence'] is False
    assert contract['production_accepted'] is False


@pytest.mark.parametrize('damage', ['report', 'source', 'target', 'config', 'budget', 'production', 'reuse'])
def test_migration_rejects_changed_evidence_or_scope(tmp_path, damage):
    contract, config, bundle, target = contract_fixture(tmp_path)
    if damage == 'report':
        from pathlib import Path
        path = Path(contract['diagnostic']['report_path'])
        path.write_text(path.read_text() + '\n')
    elif damage == 'source':
        bundle['weights_sha256'] = 'changed'
    elif damage == 'target':
        target = 'changed'
    elif damage == 'config':
        config.write_text('{"changed":true}')
    elif damage == 'budget':
        contract['max_updates'] = 200000
    elif damage == 'production':
        contract['production_accepted'] = True
    else:
        contract['reuse_source_cache'] = True
    with pytest.raises(ValueError):
        verify_processor_migration(contract, bundle, target, config)


def test_migration_budget_and_deployment_limits(tmp_path):
    contract, *_ = contract_fixture(tmp_path)
    training = PredictiveWorldTrainingConfig(
        phase='joint', init_joint='source', init_source_config='source.json',
        init_processor_migration=contract, max_updates=100000,
        normalizer_calibration_updates=0, forecast_warmup_updates=0, forecast_ramp_updates=0,
        micro_batch=256, effective_global_batch=256)
    training.validate(world_size=1)
    config = SimpleNamespace(training=training)
    require_migration_budget(config)
    for altered in (replace(training, max_updates=100001),
                    replace(training, effective_global_batch=512)):
        with pytest.raises(ValueError, match='budget'):
            require_migration_budget(SimpleNamespace(training=altered))
    with pytest.raises(ValueError, match='deployment acceptance'):
        build_deploy_policy(config, 'absent', device=torch.device('cpu'))
    with pytest.raises(ValueError, match='without diagnostic'):
        replace(training, init_processor_diagnostic={}).validate(world_size=1)
    with pytest.raises(ValueError):
        validate_processor_migration({'format': 'diagnostic'})
