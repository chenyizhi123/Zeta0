import json
from pathlib import Path
import runpy

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi.data.predictive_world.sonic_post_training import (
    inspect_dataset, extend_registries, episode_windows, MEASUREMENT, PROTOCOL,
    canonical_frame_timestamps, TIME_ALIGNMENT,
)
from psi.data.predictive_world.alignment import action_prefix_visibility
from psi.data.predictive_world.measurement_schema import MeasurementSchema, SourceCapability
from psi.data.predictive_world.index import predictive_world_temporal_abi, verify_predictive_world_index


def dataset(root, *, v3):
    (root/'meta').mkdir(parents=True)
    names = ['orientation']*6 + [f'other_{i}' for i in range(29)] + ['velocity']*29
    for i, (side, joint) in enumerate((s,j) for s in ('left','right') for j in
            ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')):
        names[6+i] = f'state.dof_pos.{side}_{joint}_joint'
    info = dict(codebase_version='v3.0' if v3 else 'v2.1', fps=50, total_episodes=2, chunks_size=1000,
        data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet' if v3 else 'data/episode_{episode_index:06d}.parquet',
        video_path='videos/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4' if v3 else 'videos/episode_{episode_index:06d}.mp4',
        vla_protocol={'schema':'unitree_g1_gmt_refpose_v3_1'},
        features={'observation.state':{'shape':[64],'names':names}, 'action':{'shape':[40]}})
    (root/'meta/info.json').write_text(json.dumps(info))
    tasks = [{'task_index':0,'task':'Sit down.'}]
    if v3:
        pq.write_table(pa.Table.from_pylist(tasks), root/'meta/tasks.parquet')
    else:
        (root/'meta/tasks.jsonl').write_text(json.dumps(tasks[0])+'\n')
    all_rows, metadata = [], []
    for episode in range(2):
        rows = [dict(episode_index=episode, timestamp=np.float32(i/50), frame_index=i,
                     **{'observation.state': [float(episode+i/100)]*64, 'action':[float(episode+i/100)]*40})
                for i in range(40)]
        fields=dict(episode_index=episode, chunk_index=0, file_index=0)
        parquet, video = root/info['data_path'].format(**fields), root/info['video_path'].format(**fields)
        parquet.parent.mkdir(parents=True,exist_ok=True)
        video.parent.mkdir(parents=True,exist_ok=True)
        video.write_bytes(b'fixture-video')
        if not v3:
            pq.write_table(pa.Table.from_pylist(rows),parquet)
        all_rows.extend(rows)
        metadata.append({'episode_index':episode, 'length':40, 'data/chunk_index':0, 'data/file_index':0,
            'videos/observation.images.front/chunk_index':0,'videos/observation.images.front/file_index':0,
            'videos/observation.images.front/from_timestamp':episode*.8,
            'videos/observation.images.front/to_timestamp':(episode+1)*.8})
    if v3:
        pq.write_table(pa.Table.from_pylist(all_rows),parquet)
        (root/'meta/episodes').mkdir()
        pq.write_table(pa.Table.from_pylist(metadata),root/'meta/episodes/file-000.parquet')
    return info


@pytest.mark.parametrize('v3', [False, True])
def test_release_keeps_episodes_disjoint_and_maps_shared_shards(tmp_path, v3):
    dataset(tmp_path, v3=v3)
    info,instruction,episodes,audit,stats = inspect_dataset(tmp_path,task='HSI_sit_sofa',validation_fraction=.5)
    assert set(audit['train_episodes']).isdisjoint(audit['validation_episodes'])
    assert len(audit['train_episodes']) == len(audit['validation_episodes']) == 1
    assert audit['time_alignment'] == TIME_ALIGNMENT
    assert audit['format'] == 'zeta0.sonic_post_training.release.v2'
    assert episodes[1]['timestamp_max_rounding_error_s'] > 0
    np.testing.assert_array_equal(episodes[1]['times'], np.arange(40)/50)
    assert episodes[1]['row_indices'][0] == (40 if v3 else 0)
    assert episodes[1]['frame_offset'] == (40 if v3 else 0)
    canonical = dict(layout_id='humanoid_upper_body_and_height_29_v1',slots=[f'slot{i}' for i in range(29)],
                     raw_indices=[0]+[None]*28,mean=[0]*29,std=[1]*29,normalization_space='per_source_registry_zscore_v1')
    source={'measurement_registry':{'old':{'canonical_body':canonical}},
            'action_protocol_registry':{},'state_schema_registry':{}}
    registries=extend_registries(source,info,stats,task='HSI_sit_sofa',audit_identity='audit')
    schema=MeasurementSchema.from_mapping(MEASUREMENT,registries['measurement_registry'][MEASUREMENT])
    capability=SourceCapability.from_mapping('HSI_sit_sofa',registries['capability_registry']['HSI_sit_sofa'])
    assert schema.canonical_body.raw_indices[14:] == (None,)*15
    assert len(schema.body_target_indices) == 14
    temporal=predictive_world_temporal_abi(history_visual_offsets_s=[-.4,-.2,0],state_history_count=21,
        state_query_hz=50,future_group_ends_s=[.2,.4,.6],action_horizon=30,action_hz=50,
        visual_max_age_s=.08,state_max_age_s=.04,action_max_age_s=.04)
    windows=episode_windows(episodes[1],task='HSI_sit_sofa',instruction=instruction,
        temporal=temporal,capability=capability,measurement_schema=schema)
    assert windows and windows[0]['materialization']['action_protocol_id'] == PROTOCOL
    assert min(windows[0]['state_indices']) >= (40 if v3 else 0)
    assert min(windows[0]['future_visual_indices']) >= (40 if v3 else 0)
    assert windows[-1]['action_valid'].count(False) > 0
    row_offset = 40 if v3 else 0
    for anchor, window in enumerate(windows[:10]):
        # Include the former float32 failure at anchor 4. Labels follow the
        # source ABI (t, t+0.6], never the current row or a duplicate interior row.
        expected = row_offset + anchor + np.arange(1, 31)
        assert window['action_valid'] == [True]*30
        np.testing.assert_array_equal(window['action_indices'], expected)
        np.testing.assert_array_equal(window['future_measurement_indices'], expected)
        assert window['future_measurement_valid'] == [True]*30
        assert window['future_visual_indices'] == (row_offset+anchor+np.array([10, 20, 30])).tolist()
        assert window['future_visual_valid'] == [True]*3
        assert window['world_action_group_valid'] == [True]*3
        assert all(window['eligibility'].values())
        np.testing.assert_array_equal(window['action_actual_times'], np.arange(1, 31)/50)
        np.testing.assert_array_equal(window['future_measurement_actual_times'], np.arange(1, 31)/50)
        np.testing.assert_array_equal(window['future_visual_actual_times'], [.2, .4, .6])
        visible = action_prefix_visibility(
            np.array([window['action_actual_times']]), np.array([window['action_valid']]), [.2, .4, .6]
        )
        assert visible[0].sum(axis=1).tolist() == [9, 19, 29]
    for anchor, window in enumerate(windows):
        state_valid = np.array(window['state_valid'])
        expected_state = anchor+np.arange(-20, 1)
        np.testing.assert_array_equal(state_valid, expected_state >= 0)
        np.testing.assert_array_equal(
            np.array(window['state_indices'])[state_valid], row_offset+expected_state[state_valid]
        )
        for key in ('state_indices', 'action_indices', 'future_measurement_indices', 'future_visual_indices'):
            assert all(row_offset <= index < row_offset+40 for index in window[key])
    # Preserve the established max-age hold near the end, then mask padding;
    # this fix does not redefine partial-episode supervision.
    assert windows[38]['action_indices'][:3] == [row_offset+39]*3
    assert windows[38]['action_valid'] == [True]*3+[False]*27
    assert not any(windows[39]['action_valid'])
    assert not any(windows[39]['future_visual_valid'])
    audit_script = Path(__file__).resolve().parents[2]/'scripts/data/audit_sonic_frame_alignment.py'
    check_episode = runpy.run_path(str(audit_script))['check_episode']
    assert check_episode(episodes[1], windows, temporal) == 10
    # The read-only real-data diagnostic must catch an interior repeated row.
    windows[4]['action_indices'][0] -= 1
    with pytest.raises(AssertionError, match='action frames'):
        check_episode(episodes[1], windows, temporal)


def test_canonical_frame_clock_accepts_only_float32_rounding():
    frames = np.arange(10001)
    expected = frames/50
    np.testing.assert_array_equal(canonical_frame_timestamps(expected, frames), expected)
    rounded = expected.astype(np.float32)
    np.testing.assert_array_equal(canonical_frame_timestamps(rounded, frames), expected)
    rounded[5] += 1e-5
    with pytest.raises(ValueError, match='beyond float32 rounding'):
        canonical_frame_timestamps(rounded, frames)


@pytest.mark.parametrize('times,frames', [
    ([0., .02, .04], [0, 1, 3]),
    ([0., .02, .04], [0, 1, 1]),
    ([0., .02, .04], [0, 1, 2.1]),
    ([0., float('nan')], [0, 1]),
    ([0., .02], [0]),
    ([], []),
])
def test_canonical_frame_clock_rejects_invalid_frames(times, frames):
    with pytest.raises(ValueError, match='contiguous episode frame indices'):
        canonical_frame_timestamps(times, frames)


def test_dataset_audit_rejects_genuine_timestamp_jitter(tmp_path):
    info = dataset(tmp_path, v3=False)
    path = tmp_path/info['data_path'].format(episode_index=0)
    rows = pq.read_table(path).to_pylist()
    rows[5]['timestamp'] += 0.001
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match='beyond float32 rounding'):
        inspect_dataset(tmp_path, task='HSI_sit_sofa')


def test_data_audit_rejects_nonfinite_values(tmp_path):
    info=dataset(tmp_path,v3=False)
    path=tmp_path/info['data_path'].format(episode_index=0)
    rows=pq.read_table(path).to_pylist()
    rows[0]['action'][0]=float('nan')
    pq.write_table(pa.Table.from_pylist(rows),path)
    with pytest.raises(ValueError,match='non-finite'):
        inspect_dataset(tmp_path,task='HSI_sit_sofa')


@pytest.mark.parametrize('first_checkpoint', [20, 10000])
@pytest.mark.parametrize('relocated', [False, True])
@pytest.mark.parametrize('migrated', [False, True])
def test_prepare_cli_publishes_audited_index_and_immutable_joint_configs(tmp_path, monkeypatch, relocated, first_checkpoint, migrated):
    from psi.data.predictive_world.cache import canonical_sha256
    from psi.config.predictive_world import load_predictive_world_config
    repo=Path(__file__).resolve().parents[2]
    root=tmp_path/'datasets/HSI_sit_sofa/sonic_refpose_v3_1'
    dataset(root,v3=True)
    source=tmp_path/'source'
    source.mkdir()
    canonical=dict(layout_id='humanoid_upper_body_and_height_29_v1',slots=[f'slot{i}' for i in range(29)],
                   raw_indices=[0]+[None]*28,mean=[0]*29,std=[1]*29,normalization_space='per_source_registry_zscore_v1')
    registry={
        'measurement_registry': {'old': {'width':1,'world_action_eligible':True,'canonical_body':canonical,
            'fields':[dict(name='q',start=0,stop=1,kind='measurement',unit='rad',time_key='timestamp',statistics_id='old')]}},
        'action_protocol_registry': {'old':dict(dim=2,mean=[0,0],std=[1,1],normalization_space='native')},
        'state_schema_registry': {'old':dict(dim=3,mean=[0,0,0],std=[1,1,1])},
    }
    payload=json.loads((repo/'configs/predictive_world/joint.template.json').read_text())
    payload['model']['qwen_model']='local-qwen'
    for name,value in registry.items():
        path=source/(name+'.json');path.write_text(json.dumps(value));payload['data'][name]=str(path)
    payload['data']['canonical_future_capabilities']='capabilities.json'
    payload['data']['index_path']='source-index.jsonl'
    manifest=source/'feature_manifest.json'
    manifest.write_text(json.dumps({'provenance':dict(encoder_id='e',processor_id='p',image_preprocessing='aspect_v1',
                                                    feature_layer='post_merger_content_area_overlap_4x4_v1',dtype='float16')}))
    payload['data']['feature_cache']=str(manifest)
    payload['training']['world_model']='parent-world'
    payload['runtime']['output_dir']=str(source/'run')
    asset_map = {}
    if relocated:
        for name in (*registry.keys(), 'feature_cache'):
            original = '/unavailable/source/'+name
            asset_map[original] = payload['data'][name]
            payload['data'][name] = original
    mapping_path = source/'asset-map.json'
    mapping_path.write_text(json.dumps(asset_map))
    config=source/'config.json';config.write_text(json.dumps(payload))
    source_bytes = config.read_bytes()
    checkpoint=source/'checkpoint';checkpoint.mkdir();(checkpoint/'.complete').touch()
    (checkpoint/'predictive_world_checkpoint.json').write_text(json.dumps({'phase':'joint','metadata':{'resolved_config_sha256':canonical_sha256(payload)}}))
    migration_path = None
    if migrated:
        from test_processor_diagnostic import fixture
        from psi.data.predictive_world.processor_migration import make_processor_migration
        from psi.data.predictive_world.cache import file_sha256
        evidence = tmp_path / 'evidence'
        evidence.mkdir()
        report_path, _, old_bundle, target_id = fixture(evidence)
        sidecar_path = checkpoint / 'predictive_world_checkpoint.json'
        sidecar = json.loads(sidecar_path.read_text())
        sidecar['metadata'].update(old_bundle['metadata'])
        sidecar['weights_sha256'] = old_bundle['weights_sha256']
        sidecar_path.write_text(json.dumps(sidecar))
        old_manifest = json.loads(manifest.read_text())
        old_manifest['provenance'].update(old_bundle['metadata'])
        manifest.write_text(json.dumps(old_manifest))
        report = json.loads(report_path.read_text())
        report['source_checkpoint_metadata'] = sidecar
        report['source_config_sha256'] = file_sha256(config)
        report_path.write_text(json.dumps(report))
        migration = make_processor_migration(report_path, accepted_by='test user', acceptance_record='test approval')
        migration_path = tmp_path / 'migration.json'
        migration_path.write_text(json.dumps(migration))
    output=tmp_path/'new-run'
    import sys
    monkeypatch.setenv('PYTHONPATH',str(repo/'src'))
    monkeypatch.setattr(sys,'argv',['prepare','--task','HSI_sit_sofa','--dataset',str(root),
        '--output',str(output),'--init-joint',str(checkpoint),'--source-config',str(config),
        '--first-checkpoint-update', str(first_checkpoint), '--checkpoint-every', '10000'])
    if relocated:
        sys.argv += ['--asset-path-map', str(mapping_path)]
    if migration_path:
        sys.argv += ['--processor-migration', str(migration_path)]
    main=runpy.run_path(str(repo/'scripts/train/predictive_world/prepare_sonic_task.py'))['main']
    main()
    resolved,actual=load_predictive_world_config(output/'config.json')
    assert resolved.training.init_joint == str(checkpoint)
    assert resolved.training.world_model is None
    assert resolved.runtime.world_size == 1
    assert resolved.training.effective_global_batch == 256
    assert resolved.training.checkpoint_retention == 2
    assert resolved.training.first_checkpoint_update == first_checkpoint
    assert resolved.training.checkpoint_every == 10000
    if migrated:
        assert resolved.training.init_processor_migration == migration
        assert resolved.training.init_processor_diagnostic is None
        cache_spec = json.loads((output / 'artifacts/feature_provenance.json').read_text())
        assert cache_spec['processor_id'] == target_id
        assert cache_spec['processor_migration'] == migration
    assert resolved.training.validation_batches == 20
    assert resolved.training.validation_every == 0
    audit_all = json.loads((output/"artifacts/dataset_audit.json").read_text())
    assert audit_all["train_episodes"] == [0, 1]
    assert audit_all["validation_episodes"] == []
    assert resolved.training.rtc_training_max_delay == 6
    assert actual['training']['rtc_training_max_delay'] == 6
    assert resolved.runtime.asset_path_map == asset_map
    assert config.read_bytes() == source_bytes
    preflight,_ = load_predictive_world_config(output/'preflight.json')
    assert preflight.training.rtc_training_max_delay == 6
    assert (output/'artifacts/source_audit.json').exists()
    audit = json.loads((output/'artifacts/dataset_audit.json').read_text())
    assert audit['time_alignment'] == TIME_ALIGNMENT
    metadata = verify_predictive_world_index(output/'artifacts/predictive_world_index.jsonl')
    assert metadata['provenance']['time_alignment'] == TIME_ALIGNMENT
    assert metadata['provenance']['dataset_audit_identity'] == canonical_sha256(audit)
    assert metadata['provenance']['anchor_selection'] == 'every_native_frame_50hz_v2'
    if migrated:
        # The first prepared config models an already-migrated 20k checkpoint:
        # its source sidecar and feature cache have the migration target
        # processor, while its config records the historical migration that got
        # it there. A further weight warm-start must not replay that migration.
        target_cache = output / 'artifacts' / 'feature_cache'
        target_cache.mkdir()
        target_provenance = json.loads((output / 'artifacts/feature_provenance.json').read_text())
        (target_cache / 'feature_manifest.json').write_text(json.dumps({'provenance': target_provenance}))
        target_checkpoint = tmp_path / 'already-migrated-checkpoint'
        target_checkpoint.mkdir()
        (target_checkpoint / '.complete').touch()
        target_sidecar = json.loads((checkpoint / 'predictive_world_checkpoint.json').read_text())
        target_sidecar['metadata']['processor_id'] = target_id
        target_sidecar['metadata']['resolved_config_sha256'] = canonical_sha256(
            json.loads((output / 'config.json').read_text()))
        (target_checkpoint / 'predictive_world_checkpoint.json').write_text(json.dumps(target_sidecar))
        same_processor_output = tmp_path / 'same-processor-warmstart'
        with monkeypatch.context() as same_processor_args:
            same_processor_args.setattr(sys, 'argv', [
                'prepare', '--task', 'HSI_sit_sofa', '--dataset', str(root),
                '--output', str(same_processor_output), '--init-joint', str(target_checkpoint),
                '--source-config', str(output / 'config.json'),
            ])
            main()
        same_processor, _ = load_predictive_world_config(same_processor_output / 'config.json')
        assert same_processor.training.init_processor_migration is None
        assert same_processor.training.init_processor_diagnostic is None
    with monkeypatch.context() as diagnostic_args:
        diagnostic_args.setattr(sys, 'argv', ['audit', '--data-root', str(root.parent.parent),
                                             '--source-config', str(config), '--tasks', 'HSI_sit_sofa',
                                             '--asset-path-map', str(mapping_path)])
        runpy.run_path(str(repo/'scripts/data/audit_sonic_frame_alignment.py'))['main']()
    before=(output/'config.json').read_text()
    main()
    assert (output/'config.json').read_text() == before
    sys.argv += ['--rtc-training-max-delay', '0']
    with pytest.raises(ValueError, match='immutable artifact differs'):
        main()
    assert (output/'config.json').read_text() == before
    # Explicitly publish different sampling experiments to different outputs.
    for maximum in (0, 13):
        other = tmp_path / f'rtc{maximum}-run'
        sys.argv[sys.argv.index('--output') + 1] = str(other)
        sys.argv[sys.argv.index('--rtc-training-max-delay') + 1] = str(maximum)
        main()
        alternate, _ = load_predictive_world_config(other / 'config.json')
        alternate_preflight, _ = load_predictive_world_config(other / 'preflight.json')
        assert alternate.training.rtc_training_max_delay == maximum
        assert alternate_preflight.training.rtc_training_max_delay == maximum
    # A published pre-frame-grid index must not be silently reused, even when
    # all its rows/checksums remain valid under the generic index verifier.
    meta_path = other/'artifacts/predictive_world_index.jsonl.meta.json'
    old_meta = json.loads(meta_path.read_text())
    del old_meta['provenance']['time_alignment']
    old_meta['provenance']['anchor_selection'] = 'every_native_frame_50hz_v1'
    old_meta['semantic_identity'] = canonical_sha256({
        'provenance': old_meta['provenance'], 'temporal_abi': old_meta['temporal_abi'],
    })
    meta_path.write_text(json.dumps(old_meta))
    verify_predictive_world_index(other/'artifacts/predictive_world_index.jsonl')
    with pytest.raises(ValueError, match='existing task index has a different identity'):
        main()


@pytest.mark.parametrize('v3', [False, True])
def test_default_posttraining_uses_all_episodes_and_statistics(tmp_path, v3):
    dataset(tmp_path, v3=v3)
    _, _, episodes, audit, stats = inspect_dataset(tmp_path, task='HSI_sit_sofa')
    assert audit['validation_fraction'] == 0
    assert audit['train_episodes'] == [0, 1]
    assert audit['validation_episodes'] == []
    assert all(ep['split'] == 'train' for ep in episodes)
    np.testing.assert_allclose(stats['body_mean'], [.695] * 64)


@pytest.mark.parametrize('fraction', [-.1, 1., float('nan')])
def test_invalid_posttraining_split_rejected(tmp_path, fraction):
    dataset(tmp_path, v3=False)
    with pytest.raises(ValueError, match='validation_fraction'):
        inspect_dataset(tmp_path, task='HSI_sit_sofa', validation_fraction=fraction)


def test_three_camera_release_preserves_views_and_rejects_missing_wrist(tmp_path):
    info = dataset(tmp_path, v3=False)
    cameras = ['observation.images.front', 'observation.images.left_wrist', 'observation.images.right_wrist']
    info['video_path'] = 'videos/{video_key}/episode_{episode_index:06d}.mp4'
    for key in cameras:
        info['features'][key] = {'dtype': 'video', 'shape': [8, 8, 3]}
        for episode in range(2):
            p = tmp_path / info['video_path'].format(video_key=key, episode_index=episode)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'video')
    (tmp_path/'meta/info.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='synchronization audit'):
        inspect_dataset(tmp_path, task='HOI_double_desk')
    info['visual_synchronization_audit_revision'] = 'test_synchronized_frames_v1'
    (tmp_path/'meta/info.json').write_text(json.dumps(info))
    _, instruction, episodes, audit, stats = inspect_dataset(tmp_path, task='HOI_double_desk', validation_fraction=.5)
    canonical = dict(layout_id='humanoid_upper_body_and_height_29_v1', slots=[f'slot{i}' for i in range(29)],
                     raw_indices=[0]+[None]*28, mean=[0]*29, std=[1]*29, normalization_space='per_source_registry_zscore_v1')
    source = {'measurement_registry': {'old': {'canonical_body': canonical}},
              'action_protocol_registry': {}, 'state_schema_registry': {}}
    registries = extend_registries(source, info, stats, task='HOI_double_desk', audit_identity='audit')
    capability = SourceCapability.from_mapping('HOI_double_desk', registries['capability_registry']['HOI_double_desk'])
    schema = MeasurementSchema.from_mapping(MEASUREMENT, registries['measurement_registry'][MEASUREMENT])
    temporal = predictive_world_temporal_abi(history_visual_offsets_s=[-.4,-.2,0], state_history_count=21,
        state_query_hz=50, future_group_ends_s=[.2,.4,.6], action_horizon=30, action_hz=50,
        visual_max_age_s=.08, state_max_age_s=.04, action_max_age_s=.04)
    row = episode_windows(episodes[0], task='HOI_double_desk', instruction=instruction,
                          temporal=temporal, capability=capability, measurement_schema=schema)[0]
    assert row['source_capability']['visual_keys'] == cameras
    assert set(row['history_visual_indices_by_camera']) == set(cameras)
    assert row['source_capability']['future_visual_keys'] == ['observation.images.front']
    # Exercise the cache-consumer boundary: exactly three front-view groups,
    # not nine groups across all current observation cameras.
    from psi.data.predictive_world.dataset import validate_predictive_world_sample_feature_mapping
    records = [dict(group=i, camera_id=cameras[0], actual_time=float(row['future_visual_actual_times'][i]),
                    frame_id=f"{Path(row['materialization']['video_paths'][cameras[0]]).resolve()}:{row['future_visual_indices'][i]}")
               for i, valid in enumerate(row['future_visual_valid']) if valid]
    validate_predictive_world_sample_feature_mapping(row, records)
    with pytest.raises(ValueError, match='incomplete'):
        validate_predictive_world_sample_feature_mapping(row, records + [dict(records[0], camera_id=cameras[1])])
    assert list(episodes[0]['video_paths']) == cameras
    assert list(audit['episodes'][0]['video_inventory']) == cameras
    (tmp_path / info['video_path'].format(video_key=cameras[-1], episode_index=1)).unlink()
    with pytest.raises(FileNotFoundError, match='right_wrist'):
        inspect_dataset(tmp_path, task='HOI_double_desk', validation_fraction=.5)
