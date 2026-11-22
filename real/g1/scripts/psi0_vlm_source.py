"""Verify the explicitly selected official PSI0 VLM without rewriting task provenance."""
import hashlib
import json
from pathlib import Path

from local_vla_compat import processor_identity


ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'psi0_egodex200k_he30k'
REPO = 'USC-PSI-Lab/psi-model'
REVISION = 'cb1480f50d9764ab2f6261bc2e651a67ad54da17'
REMOTE_DIRECTORY = 'psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k'
MANIFEST = ROOT / '.local/models/psi0/official-vlm-manifest.json'
MANIFEST_SHA256 = '662ef53f38adf5e3ccae72c85e3d64833f4ee94db40aab15316fd48cdfbd4c9e'


def verify_psi0_vlm(model_dir, selection, provenance, fingerprint, encoder_id, *, manifest_path=None):
    if selection != PROFILE:
        raise ValueError('Unknown explicit VLM selection: ' + str(selection))
    manifest_path = Path(manifest_path or MANIFEST)
    content = manifest_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Official PSI0 file manifest checksum mismatch')
    manifest = json.loads(content)
    if (manifest.get('format') != 'local_psi0_official_vlm_manifest.v1'
            or manifest.get('repo_id') != REPO or manifest.get('revision') != REVISION
            or manifest.get('remote_directory') != REMOTE_DIRECTORY):
        raise ValueError('PSI0 source does not match the selected official checkpoint')
    root = Path(model_dir).resolve()
    expected = manifest['files']
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != set(expected):
        raise ValueError('PSI0 model directory has missing or unexpected files')
    verified = {}
    for name, entry in expected.items():
        path = root / name
        if path.resolve().parent != root or path.name != name:
            raise ValueError('PSI0 model file escaped its verified directory')
        size = path.stat().st_size
        if size != entry['size']:
            raise ValueError('PSI0 file size mismatch: ' + name)
        sha256, git_digest = hashlib.sha256(), hashlib.sha1()
        git_digest.update(('blob ' + str(size) + '\0').encode())
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                sha256.update(chunk)
                git_digest.update(chunk)
        if ('sha256' in entry and sha256.hexdigest() != entry['sha256']
                or 'sha256' not in entry and git_digest.hexdigest() != entry['git_oid']):
            raise ValueError('Official PSI0 file checksum mismatch: ' + name)
        verified[name] = sha256.hexdigest()
    return dict(mode='user_selected_official_psi0_vlm', profile=PROFILE,
                repo_id=REPO, revision=REVISION, remote_directory=REMOTE_DIRECTORY,
                encoder_id=encoder_id, processor_id=processor_identity(fingerprint),
                packaged_encoder_id=provenance['encoder_id'],
                packaged_processor_id=provenance['processor_id'],
                matches_packaged_encoder=encoder_id == provenance['encoder_id'],
                user_reported_training_backbone='PSI0 VLM',
                training_metadata_conflict=encoder_id != provenance['encoder_id'],
                original_task_provenance_modified=False, verified_file_sha256=verified)
