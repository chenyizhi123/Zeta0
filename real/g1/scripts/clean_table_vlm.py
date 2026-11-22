"""Require the trained PSI0 identity and measured CUDA preprocessing equivalence."""
import copy
import hashlib
import json
from pathlib import Path
from local_vla_compat import processor_identity, require_preprocessing_runtime

ROOT = Path(__file__).resolve().parents[1]


def verify(provenance, fingerprint, encoder_id, kit, *, evidence_directory=None):
    if encoder_id != provenance['encoder_id']:
        raise ValueError('clean_table requires its exact trained PSI0 VLM')
    if processor_identity(fingerprint) == provenance['processor_id']:
        return dict(mode='exact', encoder_id=encoder_id, processor_id=provenance['processor_id'])
    directory = Path(evidence_directory or ROOT/'reports/clean_table_trash-preparation')
    record = json.loads((directory/'preprocessing-equivalence.json').read_text())
    evidence = []
    for filename in ('preprocessing-cu126.json', 'preprocessing-cu128.json'):
        raw = (directory/filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256'][filename]:
            raise ValueError('Preprocessing evidence changed')
        evidence.append(json.loads(raw))
    source, target = evidence
    if processor_identity(source['fingerprint']) != provenance['processor_id']:
        raise ValueError('Reference processor differs from training')
    expected = copy.deepcopy(source['fingerprint'])
    if expected['image_runtime']['torch'] != '2.7.0+cu126':
        raise ValueError('Unexpected reference Torch build')
    expected['image_runtime']['torch'] = '2.7.0+cu128'
    if fingerprint != expected or target['fingerprint'] != fingerprint:
        raise ValueError('Local preprocessing differs beyond tested CUDA build')
    digest = source.get('runtime_sha256')
    require_preprocessing_runtime(kit, digest)
    for report in evidence:
        if report['encoder_id'] != encoder_id or report['runtime_sha256'] != digest:
            raise ValueError('Preprocessing evidence belongs to different encoder/runtime')
    if len(source['cases']) != 3 or source['cases'] != target['cases'] or not record['exactly_equal']:
        raise ValueError('Recorded input tensors are not identical')
    mode = 'verified_recorded_cuda_build_migration' if evidence_directory else 'verified_clean_table_cuda_build_migration'
    return dict(mode=mode, encoder_id=encoder_id,
        processor_id=processor_identity(fingerprint), source_processor_id=provenance['processor_id'],
        exact_preprocessing_cases=3, original_task_provenance_modified=False)
