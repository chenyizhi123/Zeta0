"""Crash-recoverable publication beside LeRobot v2.1 and official v3 datasets.

Recording happens in a private dataset snapshot. Existing media are hard-linked
read-only, metadata are copied, and only the new episode is published. The last
write is an atomic commit manifest. Training readers must use those manifests;
the standard LeRobot indexes alone do not establish a complete commit.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

FORMAT_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_mkdir(path):
    """Persist newly created directory links as well as the leaf directory."""
    path = Path(path)
    missing = []
    ancestor = path
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created.parent)
        _fsync_directory(created)


def atomic_json(path, value):
    path = Path(path)
    _durable_mkdir(path.parent)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source, destination):
    _durable_mkdir(destination.parent)
    temporary = destination.with_name('.'+destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with Path(source).open('rb') as reader, temporary.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or str(path) in ('', '.'):
        raise ValueError('Unsafe transaction artifact path')
    return path


def _transaction_parent(destination):
    return destination.parent/'.collection-transactions'/destination.name


def _quarantine_parent(destination):
    return destination.parent/'.collection-quarantine'/destination.name


def commit_manifest_path(destination, episode):
    return Path(destination)/'meta/commits'/f'episode_{episode:06d}.json'


def verify_commit_manifest(destination, manifest):
    """Read-only integrity check; old files never acquire eligibility implicitly."""
    destination = Path(destination)
    if (manifest.get('format_version') != FORMAT_VERSION or
            manifest.get('committed') is not True or not manifest.get('artifacts')):
        raise ValueError('Invalid episode commit manifest')
    for item in manifest['artifacts']:
        relative = _safe_relative(item.get('relative', item.get('path', '')))
        path = destination/relative
        if (not path.is_file() or path.is_symlink() or
                path.stat().st_size != item['size'] or file_sha256(path) != item['sha256']):
            raise ValueError('Committed artifact is missing or changed: '+str(relative))
    return True


class EpisodeTransaction:
    """Create under the collector's exclusive lock; finalize with commit or abort."""
    def __init__(self, destination, episode):
        self.destination = Path(destination).resolve()
        if type(episode) is not int or episode < 0:
            raise ValueError('Invalid episode index')
        self.episode = episode
        self.transaction_id = uuid.uuid4().hex
        self.directory = _transaction_parent(self.destination)/self.transaction_id
        if self.directory.exists():
            raise FileExistsError(self.directory)
        _durable_mkdir(self.directory)
        self.staging_root = self.directory/'dataset'
        self.journal_path = self.directory/'journal.json'
        self.closed = False
        self.journal = dict(format_version=FORMAT_VERSION, transaction_id=self.transaction_id,
            destination=str(self.destination), episode=episode, phase='recording',
            started_at=time.time(), destination_existed=self.destination.exists(),
            new_files=[], metadata_files=[], baseline_files=[])
        atomic_json(self.journal_path, self.journal)
        if self.destination.exists():
            if not (self.destination/'meta/info.json').is_file():
                raise ValueError('Existing destination is not a complete dataset')
            info = json.loads((self.destination/'meta/info.json').read_text())
            if info.get('total_episodes') != episode:
                raise ValueError('Destination episode count changed before transaction')
            _durable_mkdir(self.staging_root)
            for source in self.destination.rglob('*'):
                if not source.is_file():
                    continue
                if source.is_symlink():
                    raise ValueError('Dataset symlinks are not accepted in staging')
                relative = source.relative_to(self.destination)
                if relative.parts[0] == 'images':
                    raise ValueError('Dataset contains uncommitted image placeholders')
                target = self.staging_root/relative
                _durable_mkdir(target.parent)
                if relative.parts[0] in ('videos', 'data'):
                    os.link(source, target)
                else:
                    shutil.copy2(source, target)  # Never hard-link mutable metadata.
                self.journal['baseline_files'].append(str(relative))
        # For a fresh dataset, staging_root MUST NOT exist: upstream create()
        # chooses create vs resume solely from the existence of the directory.
        atomic_json(self.journal_path, self.journal)

    def _publish_hook(self, phase, relative=None):
        """An overridable offline fault-injection seam, with no production action."""

    def commit(self, result):
        if self.closed:
            raise RuntimeError('Transaction is already closed')
        index = self.episode
        if int(result.get('episode', index)) != index or int(result.get('frames', 0)) < 1:
            raise ValueError('Episode result does not match transaction')
        if not result.get('storage_valid', result.get('verified', False)):
            raise ValueError('Episode storage has not passed validation')
        info_path = self.staging_root/'meta/info.json'
        info = json.loads(info_path.read_text())
        if info.get('total_episodes') != index+1:
            raise ValueError('Staged dataset counters do not include exactly one new episode')
        from episode_session import dataset_episode_media_paths
        data_path, video_paths = dataset_episode_media_paths(info, index, self.staging_root)
        baseline = set(self.journal['baseline_files'])
        media = [p for p in self.staging_root.rglob(f'episode_{index:06d}.*')
                 if p.suffix in ('.mp4', '.parquet')]
        expected_media = {self.staging_root/data_path,
                          *(self.staging_root/path for path in video_paths.values())}
        if info.get('codebase_version') == 'v3.0':
            media = [p for p in self.staging_root.rglob('*') if p.is_file()
                and p.relative_to(self.staging_root).parts[0] in ('data','videos')
                and str(p.relative_to(self.staging_root)) not in baseline]
        sidecars = [self.staging_root/'meta'/f'chenyz_{kind}_{index:06d}.{suffix}'
                    for kind, suffix in (('timing','jsonl'),('grasp','jsonl'),('episode','json'))]
        if (set(media) != expected_media or
                any(not p.is_file() or p.is_symlink() for p in expected_media) or
                any(not p.is_file() for p in sidecars)):
            raise ValueError('Commit requires exactly the declared Parquet/RGB videos and timing/grasp/episode metadata')
        if 'videos' in result and set(result['videos']) != set(video_paths):
            raise ValueError('Validated camera fields do not match the staged dataset schema')
        for kind in ('timing', 'grasp'):
            rows = [json.loads(line) for line in
                    (self.staging_root/'meta'/f'chenyz_{kind}_{index:06d}.jsonl').read_text().splitlines()
                    if line.strip()]
            if len(rows) != result['frames'] or [r.get('frame_index') for r in rows] != list(range(result['frames'])):
                raise ValueError(kind+' sidecar does not match every saved frame')
        episode_result = json.loads(sidecars[-1].read_text())
        if episode_result.get('episode') != index or episode_result.get('frames') != result['frames']:
            raise ValueError('Staged episode metadata does not match result')
        for key in ('quality', 'runtime_capture'):
            if episode_result.get(key) != result.get(key):
                raise ValueError('Staged episode metadata differs from commit '+key)
        artifacts, metadata = [], []
        for source in self.staging_root.rglob('*'):
            if not source.is_file():
                continue
            relative = str(source.relative_to(self.staging_root))
            parts = Path(relative).parts
            if parts[0] == 'images':
                raise ValueError('Placeholder images must be closed/cleared before commit')
            if parts[:2] == ('meta', 'commits'):
                continue
            if relative not in baseline:
                # Stable standard indexes are publication metadata, not episode artifacts.
                if source in media+sidecars or (parts[0] == 'meta' and parts[:2] != ('meta','episodes') and
                                               source.name not in ('info.json','episodes.jsonl',
                                                   'tasks.jsonl','tasks.parquet','stats.json',
                                                   'episodes_stats.jsonl','modality.json','chenyz_config.json')):
                    artifacts.append(source)
                elif parts[0] in ('videos', 'data'):
                    raise ValueError('Unexpected unsaved episode artifact: '+relative)
                else:
                    metadata.append(source)
            elif parts[0] not in ('videos', 'data') and file_sha256(source) != file_sha256(self.destination/relative):
                if source in sidecars:
                    raise ValueError('Refusing to replace an old episode sidecar')
                metadata.append(source)
        # An index already present in the baseline can never be overwritten.
        if any(str(p.relative_to(self.staging_root)) in baseline for p in media+sidecars):
            raise ValueError('Episode artifacts already exist; refusing overwrite')
        manifest_path = commit_manifest_path(self.destination, index)
        if manifest_path.exists():
            raise ValueError('Episode is already committed')
        artifacts = sorted(set(artifacts), key=lambda p: str(p))
        metadata = sorted(set(metadata), key=lambda p: str(p))
        manifest = dict(format_version=FORMAT_VERSION, committed=True,
            transaction_id=self.transaction_id, episode=index, frames=result['frames'],
            committed_at=time.time(), storage_valid=True,
            quality=result.get('quality', {key:result.get(key) for key in
                ('task_success','training_eligible','end_reason','truncated_by_data_interruption')}),
            runtime=result.get('runtime_capture', {}),
            artifacts=[dict(relative=str(p.relative_to(self.staging_root)),
                sha256=file_sha256(p), size=p.stat().st_size) for p in artifacts])
        self.journal['new_files'] = [item['relative'] for item in manifest['artifacts']]
        self.journal['metadata_files'] = [str(p.relative_to(self.staging_root)) for p in metadata]
        self.journal['metadata_before'] = []
        for source in metadata:
            relative = source.relative_to(self.staging_root)
            original = self.destination/relative
            existed = original.exists()
            self.journal['metadata_before'].append(dict(relative=str(relative), existed=existed,
                sha256=file_sha256(original) if existed else None))
            if existed:
                backup = self.directory/'before'/relative
                _durable_mkdir(backup.parent)
                _atomic_copy(original, backup)
        self.journal['phase'] = 'publishing'
        self.journal['manifest'] = manifest
        atomic_json(self.journal_path, self.journal)
        self._publish_hook('journal_prepared')
        _durable_mkdir(self.destination)
        for source in artifacts:
            relative = source.relative_to(self.staging_root)
            target = self.destination/relative
            _durable_mkdir(target.parent)
            # Link publishes without clobbering an existing destination. The stage
            # remains intact until the commit marker is durable.
            with source.open('rb') as durable:
                os.fsync(durable.fileno())
            os.link(source, target)
            _fsync_directory(target.parent)
            self._publish_hook('artifact_published', str(relative))
        for source in metadata:
            relative = source.relative_to(self.staging_root)
            _atomic_copy(source, self.destination/relative)
            self._publish_hook('metadata_published', str(relative))
        verify_commit_manifest(self.destination, manifest)
        self._publish_hook('before_commit_manifest')
        atomic_json(manifest_path, manifest)
        self._publish_hook('after_commit_manifest')
        self.journal['phase'] = 'committed'
        atomic_json(self.journal_path, self.journal)
        self.closed = True
        shutil.rmtree(self.directory)  # Only private links/copies, never destination.
        _fsync_directory(self.directory.parent)
        return manifest

    def abort(self, reason='operator_discard'):
        if self.closed:
            return None
        outcome = _recover_transaction(self.directory, str(reason))
        self.closed = True
        return outcome


def _recover_transaction(directory, reason='interrupted_transaction'):
    directory = Path(directory)
    journal_path = directory/'journal.json'
    journal = json.loads(journal_path.read_text())
    destination = Path(journal['destination']).resolve()
    if directory.parent != _transaction_parent(destination):
        raise ValueError('Transaction journal destination does not match its directory')
    index = journal['episode']
    commit_path = commit_manifest_path(destination, index)
    if commit_path.exists():
        manifest = json.loads(commit_path.read_text())
        if manifest.get('transaction_id') != journal['transaction_id']:
            raise ValueError('Unexpected commit marker; refusing automatic rollback')
        verify_commit_manifest(destination, manifest)
        shutil.rmtree(directory)
        _fsync_directory(directory.parent)
        return dict(episode=index, status='committed', recovered=True)
    if journal.get('phase') == 'publishing':
        manifest = journal.get('manifest', {})
        expected = {item['relative']:item for item in manifest.get('artifacts', [])}
        for name in journal['new_files']:
            relative = _safe_relative(name)
            original = destination/relative
            if not original.exists():
                continue
            item = expected[name]
            if original.stat().st_size != item['size'] or file_sha256(original) != item['sha256']:
                raise ValueError('Uncommitted artifact changed externally; refusing rollback: '+name)
            archive = directory/'partial-publication'/relative
            _durable_mkdir(archive.parent)
            os.replace(original, archive)
            _fsync_directory(original.parent)
        for item in journal.get('metadata_before', []):
            relative = _safe_relative(item['relative'])
            original = destination/relative
            staged = directory/'dataset'/relative
            if original.is_file():
                current_hash = file_sha256(original)
                allowed = {file_sha256(staged)} if staged.is_file() else set()
                if item['existed']:
                    allowed.add(item['sha256'])
                if current_hash not in allowed:
                    raise ValueError('Metadata changed externally; refusing rollback: '+str(relative))
            if item['existed']:
                backup = directory/'before'/relative
                if not backup.is_file() or file_sha256(backup) != item['sha256']:
                    raise ValueError('Metadata rollback backup is invalid: '+str(relative))
                _atomic_copy(backup, original)
            elif original.exists():
                archive = directory/'partial-publication'/relative
                _durable_mkdir(archive.parent)
                os.replace(original, archive)
                _fsync_directory(original.parent)
    journal.update(phase='quarantined', recovery_reason=reason, recovered_at=time.time())
    atomic_json(journal_path, journal)
    target = _quarantine_parent(destination)/journal['transaction_id']
    _durable_mkdir(target.parent)
    previous_parent = directory.parent
    os.replace(directory, target)
    _fsync_directory(target.parent)
    _fsync_directory(previous_parent)
    # A killed first-episode publication can leave empty directories. These have
    # no dataset metadata or files; removing only the empty shell permits create().
    if not journal.get('destination_existed') and destination.exists():
        for path in sorted(destination.rglob('*'), key=lambda p:len(p.parts), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            destination.rmdir()
            _fsync_directory(destination.parent)
        except OSError:
            pass
    return dict(episode=index, status='quarantined', path=str(target), reason=reason)


def recover_dataset(destination):
    """Recover private journals under the append lock, preserving all old data."""
    destination = Path(destination).resolve()
    parent = _transaction_parent(destination)
    outcomes = []
    if parent.exists():
        for directory in sorted(parent.iterdir()):
            if not directory.is_dir() or not (directory/'journal.json').is_file():
                raise ValueError('Incomplete transaction journal requires manual inspection: '+str(directory))
            outcomes.append(_recover_transaction(directory))
    for path in sorted((destination/'meta/commits').glob('episode_*.json')):
        verify_commit_manifest(destination, json.loads(path.read_text()))
    return outcomes


def dataset_transaction_status(destination, verify_artifacts=False):
    """Read-only dry-run status; do not repair or classify legacy as successful."""
    destination = Path(destination).resolve()
    pending = []
    parent = _transaction_parent(destination)
    if parent.exists():
        pending = [str(path) for path in parent.iterdir() if path.is_dir()]
    commits = []
    for path in sorted((destination/'meta/commits').glob('episode_*.json')):
        manifest = json.loads(path.read_text())
        if verify_artifacts:
            verify_commit_manifest(destination, manifest)
        elif manifest.get('format_version') != FORMAT_VERSION or manifest.get('committed') is not True:
            raise ValueError('Invalid episode commit manifest: '+str(path))
        commits.append(manifest['episode'])
    info_path = destination/'meta/info.json'
    total = json.loads(info_path.read_text()).get('total_episodes', 0) if info_path.exists() else 0
    return dict(pending_transactions=pending, committed_episodes=commits,
                legacy_unreviewed_episodes=[i for i in range(total) if i not in commits])
