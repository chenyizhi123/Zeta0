"""Local LeRobot v2.1 empty-dataset indexes; never reconstruct saved episodes."""
import json
from pathlib import Path

INDEXES = ('tasks.jsonl','episodes.jsonl','episodes_stats.jsonl')


def ensure_empty_episode_indexes(root):
    root = Path(root)
    info_path = root/'meta/info.json'
    if not info_path.exists():
        return []  # A fresh metadata constructor has not run yet.
    info = json.loads(info_path.read_text())
    missing = [root/'meta'/name for name in INDEXES if not (root/'meta'/name).exists()]
    if not missing:
        return []
    if info.get('codebase_version') != 'v2.1':
        raise ValueError('只能补齐 LeRobot v2.1 空数据集索引')
    counters = ('total_episodes','total_frames','total_tasks','total_videos','total_chunks')
    if any(type(info.get(key)) is not int or info[key] != 0 for key in counters):
        raise ValueError('已有数据或计数异常且元数据索引缺失；不会伪造索引，请检查: '+str(root))
    if (root/'verification_failed.json').exists():
        raise ValueError('存在保存校验失败标记，不自动补齐索引: '+str(root))
    for path in root.rglob('*'):
        if path.is_file() and (path.suffix.lower() in ('.mp4','.parquet','.jpg','.jpeg','.png')
                               or path.name.startswith('chenyz_episode_')
                               or path.name.startswith('chenyz_grasp_')
                               or path.name.startswith('chenyz_timing_')):
            raise ValueError('空计数数据集存在录制文件，不自动补齐索引: '+str(path))
    for name in INDEXES:
        path = root/'meta'/name
        if path.exists() and (not path.is_file() or path.stat().st_size != 0):
            raise ValueError('空计数数据集存在非空索引，不自动修复: '+str(path))
    created = []
    for path in missing:
        with path.open('x'):
            pass
        created.append(path.name)
    return created


def install_empty_metadata_indexes(metadata_class):
    """Create all empty indexes before any video writer or episode is opened."""
    if getattr(metadata_class,'_chenyz_empty_indexes_installed',False):
        return
    original = metadata_class.create.__func__

    def create(cls,*args,**kwargs):
        result = original(cls,*args,**kwargs)
        ensure_empty_episode_indexes(result.root)
        return result

    metadata_class.create = classmethod(create)
    metadata_class._chenyz_empty_indexes_installed = True
