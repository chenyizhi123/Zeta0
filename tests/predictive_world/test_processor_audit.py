"""Audit equality is bounded evidence, never permission to relabel old artifacts."""
from copy import deepcopy

import pytest
import torch

from psi.data.predictive_world.processor_audit import (
    compare_processor_reports, legacy_processor_diagnostic, tensor_record,
)


def fingerprints():
    source = {'method': 'qwen_image_processor_json_sha256_v1',
              'transformers_version': '4.57.3', 'processor_class': 'Qwen3VLProcessor',
              'image_processor_class': 'Qwen2VLImageProcessorFast',
              'image_processor_config_sha256': 'fixture'}
    actual = dict(source, method='qwen_image_processor_json+image_runtime_v2',
                  image_runtime={'pillow': '12.3.0', 'numpy': '1.26.4', 'torch': '2.7.0+cu126'})
    return source, actual


def test_legacy_field_match_never_conflates_runtime_identity():
    source, actual = fingerprints()
    result = legacy_processor_diagnostic(source, actual)
    assert result['recorded_v1_fields_match']
    assert result['source_processor_id'] != result['actual_processor_id']
    assert not result['authorizes_reuse']
    assert result['source_runtime_unrecorded'] == ['pillow', 'numpy', 'torch']


def test_legacy_difference_is_explained_without_changing_source():
    source, actual = fingerprints()
    before = deepcopy(source)
    actual['transformers_version'] = '4.57.1'
    result = legacy_processor_diagnostic(source, actual)
    assert not result['recorded_v1_fields_match']
    assert result['differences'] == {'transformers_version': {'source': '4.57.3', 'actual': '4.57.1'}}
    assert source == before


def test_missing_source_fields_rejected():
    source, actual = fingerprints()
    del source['image_processor_config_sha256']
    with pytest.raises(ValueError, match='complete'):
        legacy_processor_diagnostic(source, actual)


def report():
    return dict(format='qwen_processor_cpu_ab_v1', source_files={'stage': 'hash'},
                model_sha256='model', code_sha256={'producer': 'hash'},
                fixture_spec={'tasks': ['fixture']},
                fixtures={'image': tensor_record(torch.zeros(2, 3))})


def test_equal_report_still_does_not_authorize_migration():
    result = compare_processor_reports(report(), report())
    assert result['byte_equal']
    assert not result['full_teacher_equivalence']
    assert not result['authorizes_reuse']


@pytest.mark.parametrize('field', ['format', 'source_files', 'model_sha256',
                                  'code_sha256', 'fixture_spec', 'fixtures'])
def test_changed_evidence_rejected(field):
    baseline, candidate = report(), report()
    candidate[field] = {}
    with pytest.raises(ValueError, match='differs|differ'):
        compare_processor_reports(baseline, candidate)


def test_no_samples_is_not_success():
    baseline = report()
    baseline['fixtures'] = {}
    with pytest.raises(ValueError):
        compare_processor_reports(baseline, baseline)


def test_nonfinite_tensor_rejected_and_dtype_included():
    with pytest.raises(ValueError, match='non-finite'):
        tensor_record(torch.tensor([float('nan')]))
    assert tensor_record(torch.zeros(2)) != tensor_record(torch.zeros(2, dtype=torch.int32))
