"""Task naming, exclusive append, and persisted episode validation. No control APIs."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re


HEAD_VIDEO_KEY = 'observation.images.ego_view'
THREE_VIDEO_KEYS = (HEAD_VIDEO_KEY, 'observation.images.left_wrist',
                    'observation.images.right_wrist')


def dataset_video_keys(info):
    """Camera schemas are explicit; old three-camera metadata need no new fields."""
    features = info.get('features')
    if not isinstance(features, dict):
        raise ValueError('数据集缺少相机字段声明 features')
    keys = tuple(key for key, feature in features.items()
                 if isinstance(feature, dict) and feature.get('dtype') == 'video')
    if set(keys) not in ({HEAD_VIDEO_KEY}, set(THREE_VIDEO_KEYS)):
        raise ValueError('仅支持顶部 ego_view 单路或顶部加左右腕三路 RGB 视频字段')
    return keys


def validate_dataset_cameras(path, camera_keys):
    """Reject a camera-mode change before any existing dataset writer is opened."""
    expected = tuple(key if key.startswith('observation.images.') else
                     'observation.images.'+key for key in camera_keys)
    dataset_video_keys(dict(features={key:dict(dtype='video') for key in expected}))
    info_path = Path(path)/'meta/info.json'
    if info_path.exists():
        actual = dataset_video_keys(json.loads(info_path.read_text()))
        if set(actual) != set(expected):
            raise ValueError('已有数据集相机字段不同，请用新任务名：已有 '+
                             ', '.join(actual)+'；本次 '+', '.join(expected))
    return expected


def dataset_episode_rows(root):
    """Read official v3 episode indexes, including its chunk/file coordinates."""
    import pyarrow.parquet as pq
    rows = []
    for path in sorted((Path(root)/'meta/episodes').rglob('*.parquet')):
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows,key=lambda row:row['episode_index'])


def dataset_episode_media_paths(info, episode, root=None):
    """Render the complete Parquet/video inventory from the persisted schema."""
    keys = dataset_video_keys(info)
    chunks = info.get('chunks_size')
    if type(chunks) is not int or chunks <= 0:
        raise ValueError('数据集 chunks_size 必须为正整数')
    arguments = dict(episode_chunk=episode//chunks, episode_index=episode)
    video_arguments = {key:{**arguments,'video_key':key} for key in keys}
    if info.get('codebase_version') == 'v3.0':
        if root is None:
            raise ValueError('LeRobot v3 媒体路径需要 episode 索引所在目录')
        rows = [row for row in dataset_episode_rows(root) if row['episode_index']==episode]
        if len(rows) != 1:
            raise ValueError('LeRobot v3 episode 索引缺失或重复')
        row = rows[0]
        arguments = dict(chunk_index=row['data/chunk_index'],file_index=row['data/file_index'])
        video_arguments = {key:dict(video_key=key,
            chunk_index=row[f'videos/{key}/chunk_index'],file_index=row[f'videos/{key}/file_index'])
            for key in keys}
    try:
        data = Path(info['data_path'].format(**arguments))
        videos = {key:Path(info['video_path'].format(**video_arguments[key]))
                  for key in keys}
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError('数据集媒体路径模板无效') from error
    paths = (data, *videos.values())
    if any(path.is_absolute() or '..' in path.parts or str(path) in ('', '.')
           for path in paths):
        raise ValueError('数据集媒体路径必须在数据集目录内')
    if (data.suffix != '.parquet' or any(path.suffix != '.mp4' for path in videos.values()) or
            len(set(paths)) != len(paths)):
        raise ValueError('数据集媒体路径必须声明独立的 Parquet 和各路 MP4')
    return data, videos


def task_path(root, task):
    if not task or task in ('.', '..') or not re.fullmatch(r'[\w .-]+', task, re.UNICODE):
        raise ValueError('任务名只能包含文字、数字、空格、下划线、点和横线，不能包含路径分隔符')
    if task != task.strip() or len(task) > 100:
        raise ValueError('任务名不可带首尾空格，长度不能超过 100')
    return Path(root).resolve()/task


def dataset_plan(path, task, prompt, requested=None, camera_keys=None, codebase_version=None):
    path = Path(path)
    from episode_transaction import dataset_transaction_status
    status = dataset_transaction_status(path)
    if status['pending_transactions']:
        raise ValueError('发现待恢复的采集事务；正常采集入口会在独占锁内恢复，dry-run不修改数据')
    info_path = path/'meta/info.json'
    manifest_path = path/'task.json'
    if path.exists() and not info_path.exists():
        raise ValueError('目录不是完整 LeRobot 数据集，请检查后换任务名；不会覆盖: '+str(path))
    if (path/'verification_failed.json').exists():
        raise ValueError('上次保存校验失败，须先检查 verification_failed.json')
    if camera_keys is not None:
        validate_dataset_cameras(path, camera_keys)
    next_episode = 0
    if info_path.exists():
        info = json.loads(info_path.read_text())
        version = info.get('codebase_version')
        if version not in ('v2.1','v3.0'):
            raise ValueError('仅支持 LeRobot v2.1/v3.0 数据集')
        if codebase_version is not None and version != codebase_version:
            raise ValueError('已有任务为 '+version+'，本次后端为 '+codebase_version+
                '；官方 v3 采集请用新任务名，续采旧任务需 --backend legacy')
        next_episode = int(info['total_episodes'])
        if not manifest_path.exists():
            raise ValueError('此数据集不是按任务采集入口创建，请使用新任务名')
        manifest = json.loads(manifest_path.read_text())
        if manifest['task'] != task or manifest['prompt'] != prompt:
            raise ValueError('任务名或任务描述与已有数据集不同，请使用新任务名')
        # Reject files from an interrupted/uncommitted episode before opening writers.
        for candidate in list(path.rglob('episode_*.mp4'))+list(path.rglob('episode_*.parquet')):
            if int(candidate.stem.split('_')[-1]) >= next_episode:
                raise ValueError('发现未提交文件，先检查并移走，避免覆盖: '+str(candidate))
    if requested is not None and requested != next_episode:
        raise ValueError(f'下一个 episode 必须为 {next_episode}，不能覆盖或跳号（输入 {requested}）')
    return next_episode


@contextmanager
def collection_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('已有按任务采集进程运行，请先结束它') from error
        yield


def verify_saved_episode(exporter, episode, expected_frames):
    import av
    import numpy as np
    import pyarrow.parquet as pq
    root = Path(exporter.root)
    info = json.loads((root/'meta/info.json').read_text())
    data_path, video_paths = dataset_episode_media_paths(info, episode, root)
    features = getattr(exporter, 'features', {})
    declared_keys = set(video_paths)
    writer_keys = tuple(exporter.meta.video_keys)
    if (set(writer_keys) != declared_keys or len(writer_keys) != len(declared_keys) or
            set(dataset_video_keys(dict(features=features))) != declared_keys):
        raise ValueError('录制相机字段与已保存数据集相机声明不同')
    if Path(exporter.meta.get_data_file_path(episode)) != data_path or any(
            Path(exporter.meta.get_video_file_path(episode, key)) != path
            for key, path in video_paths.items()):
        raise ValueError('录制媒体路径与数据集声明不同')
    expected_media = {root/data_path, *(root/path for path in video_paths.values())}
    if info.get('codebase_version') == 'v3.0':
        # Our per-save official resume starts fresh files. No shared mutable
        # media are admitted into a commit manifest.
        table = pq.read_table(root/data_path)
        actual_media = expected_media
        row = next(r for r in dataset_episode_rows(root) if r['episode_index']==episode)
        if any(abs(row[f'videos/{key}/from_timestamp'])>1e-9 for key in video_paths):
            raise ValueError('本采集入口要求每条 v3 episode 独立视频文件')
        for key in video_paths:
            statistics=exporter.meta.stats.get(key,{})
            for name in ('min','max','mean','std','count'):
                if name not in statistics or not np.isfinite(np.asarray(statistics[name])).all():
                    raise ValueError(key+' 缺少有效的官方图像统计：'+name)
    else:
        actual_media = {path for path in root.rglob(f'episode_{episode:06d}.*')
                        if path.suffix in ('.mp4', '.parquet')}
    if expected_media != actual_media or any(not path.is_file() or path.is_symlink()
                                             for path in expected_media):
        raise ValueError('已保存 Parquet/视频文件与数据集声明不符')
    table = pq.read_table(root/data_path)
    if expected_frames <= 0 or table.num_rows != expected_frames:
        raise ValueError('Parquet 帧数不符')
    for key, dimension in [('observation.state',29), ('action.wbc',29),
                           ('observation.gripper',2), ('action.gripper',2)]:
        values = np.asarray(table[key].to_pylist())
        if values.shape != (expected_frames, dimension) or not np.isfinite(values).all():
            raise ValueError('错误的字段形状或数值: '+key)
        if 'gripper' in key and ((values < 0).any() or (values > 1).any()):
            raise ValueError('手部开合值超出 0..1')
    for key, feature in features.items():
        if feature.get('dtype') not in ('float32','float64','int64','int32','int16','int8',
                                        'uint64','uint32','uint16','uint8','bool'):
            continue
        if key not in table.column_names:
            raise ValueError('Parquet 缺少字段: '+key)
        values = np.asarray(table[key].to_pylist())
        scalar_index = key in ('timestamp','frame_index','episode_index','index','task_index')
        expected = (expected_frames,) if scalar_index else (expected_frames,*feature['shape'])
        allowed_shapes = {expected}
        # HF represents any one-element feature as a scalar Value. Upstream
        # SONIC emits e.g. delta_heading as [value] before Parquet flattens it.
        # Both representations are valid for shape [1]; larger vectors stay strict.
        if tuple(feature['shape']) == (1,):
            allowed_shapes.update(((expected_frames,), (expected_frames,1)))
        if values.shape not in allowed_shapes or not np.isfinite(values).all():
            raise ValueError('错误的字段形状或数值: '+key)
    if set(table['episode_index'].to_pylist()) != {episode}:
        raise ValueError('episode 编号不符')
    if table['frame_index'].to_pylist() != list(range(expected_frames)):
        raise ValueError('frame_index 必须逐行连续')
    videos = {}
    for key, path in video_paths.items():
        with av.open(str(root/path)) as video:
            frames = 0
            times = []
            shape = features.get(key, {}).get('shape')
            for frame in video.decode(video=0):
                if shape and (frame.height,frame.width) != tuple(shape[:2]):
                    raise ValueError(key+' 视频分辨率与数据集不同')
                frames += 1
                if info.get('codebase_version') == 'v3.0':
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError(key+' 视频缺少 PTS/time_base')
                    times.append(float(frame.pts*frame.time_base))
        if frames != expected_frames:
            raise ValueError(f'{key} 视频帧数 {frames} 与 Parquet {expected_frames} 不符')
        if times and not np.allclose(times,np.arange(frames)/info['fps'],atol=1e-4,rtol=0):
            raise ValueError(key+' 视频时间轴与 Parquet 不一致')
        videos[key] = frames
    # This verifies storage only. It does not assert a completed task or approve
    # the episode for training. Those require explicit operator/quality labels.
    return dict(episode=episode, frames=expected_frames, videos=videos,
                verified=True, storage_valid=True)
