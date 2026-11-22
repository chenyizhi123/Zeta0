"""Retired SONIC frame adapter, retained for historical dataset tooling.

Only hardware schema/provenance live here. Encoding, statistics, parquet,
episode indexes, discard and finalization are owned by LeRobot.
"""
import copy
import importlib.metadata
import json
from pathlib import Path


LEROBOT_VERSION = '0.6.1'
LEROBOT_COMMIT = '7e241bd630a3719a56157a497ce5d08f244784f1'


class LeRobotExporter:
    backend = 'lerobot'

    def __init__(self, root, fps, features, modalities, task, script_config):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.configs.video import rgb_encoder_defaults

        if importlib.metadata.version('lerobot') != LEROBOT_VERSION:
            raise RuntimeError('采集需要固定的官方 LeRobot '+LEROBOT_VERSION+'；运行 bash scripts/install_lerobot.sh')
        self.root = Path(root)
        self.task = task
        # Official defaults: AV1 / CRF 30 / GOP 2 / yuv420p / preset 12.
        # Use the official default PNG-first path. Streaming can drop individual
        # camera frames when its queue fills, which violates our paired rows.
        self.encoder = rgb_encoder_defaults()
        options = dict(root=self.root, rgb_encoder=self.encoder,
                       video_backend='pyav', streaming_encoding=False,
                       batch_encoding_size=1, encoder_threads=4,
                       image_writer_threads=4*sum(f['dtype']=='video' for f in features.values()))
        repo_id = 'local/sonic_collection'
        if (self.root/'meta/info.json').exists():
            info=json.loads((self.root/'meta/info.json').read_text())
            if info.get('codebase_version')!='v3.0':
                raise ValueError('官方后端只续采 v3；旧任务请用 --backend legacy')
            for key,feature in info['features'].items():
                if feature['dtype']!='video':continue
                saved=feature.get('info',{})
                expected={'video.codec':'av1','video.pix_fmt':'yuv420p',
                          'video.g':self.encoder.g,'video.crf':self.encoder.crf,
                          'video.preset':self.encoder.preset}
                if any(saved.get(k)!=value for k,value in expected.items()):
                    raise ValueError('已有任务的视频编码参数与官方默认配置不同，请用新任务名：'+key)
            self.dataset = LeRobotDataset.resume(repo_id, **options)
        else:
            self.dataset = LeRobotDataset.create(repo_id, fps=fps,
                features=copy.deepcopy(features), robot_type='g1_sonic_brainco', **options)
        self.meta = self.dataset.meta
        # Official DatasetInfo intentionally has a closed schema. Keep device
        # provenance beside it instead of patching upstream metadata classes.
        (self.root/'meta/chenyz_config.json').write_text(json.dumps(dict(
            script_config=script_config,backend=dict(name='lerobot',version=LEROBOT_VERSION,
                source_commit=LEROBOT_COMMIT,streaming_encoding=False,encoder_threads=4)),
            ensure_ascii=False,indent=2))
        (self.root/'meta/modality.json').write_text(json.dumps(modalities,ensure_ascii=False,indent=2))

    @property
    def features(self):
        return self.dataset.features

    @property
    def episode_buffer(self):
        return self.dataset.writer.episode_buffer

    def add_frame(self, frame):
        self.dataset.add_frame({**frame, 'task':self.task})

    def save_episode(self):
        from lerobot.datasets.io_utils import load_episodes
        # The receiver and official image writer are threaded. Avoid forking
        # their live threads into the upstream parallel encoder process pool.
        self.dataset.save_episode(parallel_encoding=False)
        # Footer/index flush is mandatory before verifying or publishing.
        self.dataset.finalize()
        self.meta.episodes = load_episodes(self.root)

    def cancel(self):
        self.dataset.clear_episode_buffer()
        self.dataset.finalize()

    def close(self):
        self.dataset.finalize()

    def metrics_snapshot(self):
        import av
        return dict(backend='lerobot',version=LEROBOT_VERSION,streaming_encoding=False,
                    source_commit=LEROBOT_COMMIT,encoder_threads=4,encoder=self.encoder.__dict__,
                    pyav_version=av.__version__,ffmpeg_libraries=av.library_versions)
