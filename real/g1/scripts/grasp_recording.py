"""Validate and record operator grasp inputs without changing old dataset schemas."""
import numpy as np
from brainco_gripper_data import MissingGripperData
from recording_runtime import decode_runtime_identity, config_hash

PROFILE_KEY = 'teleop.grasp_profile'
CLOSURE_KEY = 'teleop.grasp_closure'


def grasp_features():
    return {
        PROFILE_KEY: dict(dtype='int64', shape=(2,), names=['left_grasp_profile','right_grasp_profile']),
        CLOSURE_KEY: dict(dtype='float32', shape=(2,), names=['left_grasp_closure','right_grasp_closure']),
    }


def use_grasp_features(existing_info=None):
    if existing_info is None:
        return True
    present = [key in existing_info['features'] for key in grasp_features()]
    if any(present) and not all(present):
        raise ValueError('已有数据集抓姿字段不完整，请检查数据集或使用新任务名')
    return all(present)


def grasp_sample(manager, packet, required=False):
    """Log requested inputs separately from actual six-joint DDS commands."""
    profile = np.asarray(manager.get('brainco_grasp_profile',[]))
    stroke = np.asarray(manager.get('brainco_grasp_closure',[]))
    valid = (profile.shape == (2,) and stroke.shape == (2,) and
             np.isin(profile,[0,1]).all() and np.isfinite(stroke).all() and
             ((0 <= stroke) & (stroke <= 1)).all())
    if required and not valid:
        raise MissingGripperData('等待新版 PICO 抓姿输入；退出旧 SONIC 后重新启动，再录制新任务')
    result = dict(profile=profile.astype(int).tolist() if valid else None,
                  closure=stroke.tolist() if valid else None)
    revision = manager.get('brainco_grasp_revision')
    if revision is not None and (len(revision) != 1 or revision[0] not in (0, 1, 2)):
        raise MissingGripperData('抓姿映射版本无效')
    result['mapping_revision'] = int(revision[0]) if revision is not None else None
    identity=decode_runtime_identity(manager)
    result['config_sha256']=identity['grasp_config_sha256'] if identity else None
    result['runtime_session']=identity['session'] if identity else None
    for side in ('left','right'):
        q = np.asarray(manager.get('brainco_'+side+'_q',[]))
        result[side+'_requested_q'] = (q.tolist() if q.shape == (6,) and
            np.isfinite(q).all() and ((0 <= q) & (q <= 1)).all() else None)
    for kind in ('state','action'):
        q = packet.get(kind+'_joints')
        if q is not None:
            q = np.asarray(q)
            if q.shape != (2,6) or not np.isfinite(q).all() or not ((0 <= q) & (q <= 1)).all():
                raise MissingGripperData('抓姿记录的实际手部六通道无效')
            result[kind+'_joints'] = q.tolist()
        else:
            result[kind+'_joints'] = None  # Old read-only sources have only means.
    speeds=packet.get('action_speed_joints')
    if speeds is not None:
        speeds=np.asarray(speeds)
        if speeds.shape!=(2,6) or not np.isfinite(speeds).all() or not ((0<=speeds)&(speeds<=1)).all():
            raise MissingGripperData('Actual hand command speeds are invalid')
        result['action_speed_joints']=speeds.tolist()
    else:
        result['action_speed_joints']=None
    for name in ('source_session','action_receive_monotonic','action_receive_sequence'):
        result[name]=packet.get(name)
    return result


def sample_features(sample):
    return {PROFILE_KEY: np.asarray(sample['profile'],dtype=np.int64),
            CLOSURE_KEY: np.asarray(sample['closure'],dtype=np.float32)}


def save_grasp_sidecar(destination, index, samples, config):
    import json
    path = destination/'meta'/f'chenyz_grasp_{index:06d}.jsonl'
    with path.open('x') as output:
        for frame, sample in enumerate(samples):
            output.write(json.dumps(dict(frame_index=frame,**sample),allow_nan=False)+'\n')
    revisions = sorted({s['mapping_revision'] for s in samples
                        if s.get('mapping_revision') is not None})
    snapshot_matches = (config is not None and bool(samples) and
                        all(s.get('mapping_revision') == config.get('mapping_revision', 1)
                            for s in samples))
    return dict(grasp_input_file=str(path.relative_to(destination)),
                grasp_input_frames=len(samples),grasp_profiles=config,
                grasp_runtime_revisions=revisions,
                grasp_mapping_revision_matches_snapshot=snapshot_matches,
                grasp_config_sha256=config_hash(config) if config is not None else None,
                grasp_config_hash_matches_snapshot=(config is not None and bool(samples) and
                    all(s.get('config_sha256')==config_hash(config) for s in samples)),
                grasp_input_alignment='frame_index aligned to saved episode; receive-time snapshots, not hardware synchronization')
