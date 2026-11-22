import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/data"
sys.path.insert(0, str(SCRIPTS))
from convert_hiw500_multinode import balance_shards, check_coverage
from convert_hiw500_mcap import encode_video


def test_balancing_is_deterministic_disjoint_and_size_aware():
    entries = [{"path": str(i), "size": size} for i, size in enumerate([100, 90, 80, 20, 10, 5])]
    shards, loads = balance_shards(entries, 3)
    assert sorted(e["path"] for shard in shards for e in shard) == [str(i) for i in range(6)]
    assert max(loads) - min(loads) <= 5
    assert (shards, loads) == balance_shards(list(reversed(entries)), 3)


def test_coverage_requires_every_episode_exactly_once_including_failures():
    records = [{"source": {"relative_path": "a"}}]
    check_coverage(["a", "b"], records, [{"source": "b"}])
    with pytest.raises(ValueError):
        check_coverage(["a", "b"], records, [])
    with pytest.raises(ValueError):
        check_coverage(["a"], records, [{"source": "a"}])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_mjpeg_copy_preserves_jpeg_payload_and_frame_identity(tmp_path):
    import cv2
    av = pytest.importorskip("av")
    samples = []
    for index in range(4):
        pixels = np.full((32, 64, 3), index * 40, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", pixels)
        assert ok
        samples.append((index, jpeg.tobytes()))
    path = tmp_path / "video.mp4"
    assert encode_video(path, samples, np.arange(4), crf=18, threads=1, codec="mjpeg") == [32, 64, 3]
    with av.open(str(path)) as container:
        packets = [bytes(packet) for packet in container.demux(video=0) if packet.size]
    assert packets == [jpeg for _, jpeg in samples]


def test_three_node_launcher_dry_run_never_starts_conversion():
    env = dict(os.environ, PET_NNODES="3", PET_NODE_RANK="2", PET_NPROC_PER_NODE="8",
               MASTER_ADDR="127.0.0.1", MASTER_PORT="29891", DRY_RUN="1")
    result = subprocess.run(["bash", str(SCRIPTS / "run_hiw500_3node_convert.sh")],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "workers_per_node=128 codec=mjpeg-copy" in result.stdout
    assert "GPU_LOAD" not in result.stdout and "zhanka" not in result.stdout
    assert "rank=2" in result.stdout
    assert "runtime_limit=none progress_every=10" in result.stdout
    env["PET_NNODES"] = "4"
    result = subprocess.run(["bash", str(SCRIPTS / "run_hiw500_3node_convert.sh")],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2


def test_progress_heartbeat_continues_without_completed_episodes(capsys):
    import time
    from convert_hiw500_multinode import LiveProgress
    class Store:
        def set(self, key, value):
            pass
    reporter = LiveProgress(1, 3, Store(), interval=0.01)
    reporter.update(stage="converting", completed=0, selected=20)
    reporter.started -= 3700
    reporter.start()
    try:
        time.sleep(0.05)
    finally:
        reporter.close()
    output = capsys.readouterr().out
    assert output.count('"event": "heartbeat"') >= 2
    assert '"completed": 0' in output
    assert '"target_exceeded": true' in output
    assert not reporter.thread.is_alive()


def test_waiting_for_healthy_long_running_peer_has_no_runtime_deadline(monkeypatch):
    import types
    import time
    import convert_hiw500_multinode as module
    class Store:
        attempts = 0
        def set(self, key, value):
            pass
        def check(self, keys):
            if keys[0].startswith("ready/"):
                self.attempts += 1
                return self.attempts >= 4
            return True
        def get(self, key):
            return str(time.time()).encode()
    store = Store()
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    module.wait_for_stage(store, "converted", 0, 3, types.SimpleNamespace(error=None), stale_seconds=1)
    assert store.attempts == 4
