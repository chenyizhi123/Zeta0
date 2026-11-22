"""Validate the recorded CUDA-build migration without relabeling model files."""
import copy
import hashlib
import json


def processor_identity(fingerprint):
    payload = json.dumps(fingerprint, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return 'qwen3vl-processor:' + hashlib.sha256(payload.encode()).hexdigest()


def require_preprocessing_runtime(kit, recorded_digest):
    manifest = (kit / 'COMPATIBILITY.json').read_bytes()
    current_digest = hashlib.sha256(manifest).hexdigest()
    # Release copies can retain evidence from an earlier manifest when the
    # recorded preprocessing path is unchanged.
    reference_digest = json.loads(manifest).get('preprocessing_reference_compatibility_sha256')
    if not isinstance(recorded_digest, str) or recorded_digest not in (current_digest, reference_digest):
        raise ValueError('Preprocessing evidence belongs to a different runtime')


def validate_preprocessing_migration(provenance, fingerprint, encoder_id, kit_sha256, report, source, target):
    if encoder_id != provenance['encoder_id']:
        raise ValueError('The local Qwen model files differ from the trained encoder')
    actual_id = processor_identity(fingerprint)
    if actual_id == provenance['processor_id']:
        return dict(mode='exact', encoder_id=encoder_id, processor_id=actual_id)
    if report.get('format') != 'local_zeta0_cuda_build_migration.v1':
        raise ValueError('A recorded preprocessing comparison is required for this runtime')
    if report.get('artifact_provenance_modified') is not False:
        raise ValueError('Model provenance must remain unchanged')
    source_fp = source.get('fingerprint')
    if processor_identity(source_fp) != provenance['processor_id']:
        raise ValueError('Reference preprocessing does not match the trained processor')
    candidate = copy.deepcopy(source_fp)
    if (candidate['image_runtime']['torch'] != '2.7.0+cu126'
            or fingerprint['image_runtime']['torch'] != '2.7.0+cu128'):
        raise ValueError('Only the tested PyTorch 2.7.0 cu126-to-cu128 build migration is supported')
    candidate['image_runtime']['torch'] = '2.7.0+cu128'
    if candidate != fingerprint or target.get('fingerprint') != fingerprint:
        raise ValueError('The local preprocessing runtime differs from the tested build migration')
    for evidence in (source, target, report):
        if evidence.get('encoder_id') != encoder_id or evidence.get('kit_compatibility_sha256') != kit_sha256:
            raise ValueError('Migration evidence belongs to different model/runtime files')
    cases = source.get('cases', {})
    if (len(cases) != 6 or cases != target.get('cases') or cases != report.get('cases')
            or report.get('all_preprocessed_tensors_exactly_equal') is not True):
        raise ValueError('Reference and local preprocessing tensors are not exactly equal')
    for task in ('flatten_box', 'tidy_table_trash'):
        if sum(key.startswith(task + '/') for key in cases) != 3:
            raise ValueError('Migration evidence needs three real frames from each task')
    if report.get('source_fingerprint') != source_fp or report.get('target_fingerprint') != fingerprint:
        raise ValueError('Migration fingerprints were modified')
    return dict(mode='verified_local_cuda_build_migration', encoder_id=encoder_id,
                source_processor_id=provenance['processor_id'], processor_id=actual_id,
                exact_preprocessing_cases=len(cases))
