"""Exercise AV1 frame addressing and the cache/current-reader boundary on CPU."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest
import torch

from psi.data.predictive_world.dataset import PredictiveWorldRowReader
from psi.data.predictive_world.video import decode_torchcodec_frames, decode_frame_job, initialize_decode_worker


@pytest.fixture
def av1_video(tmp_path):
    pytest.importorskip("torchcodec")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is needed to create the real AV1 fixture")
    path = tmp_path / "av1.mp4"
    frames = np.zeros((8, 48, 64, 3), dtype=np.uint8)
    for index in range(8):
        frames[index, :, :, 0] = index * 30
        frames[index, :, :, 1] = np.arange(64)[None, :] * 3
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x48",
        "-r", "50", "-i", "pipe:0", "-c:v", "libaom-av1", "-cpu-used", "8", "-threads", "1",
        str(path),
    ], input=frames.tobytes(), check=True)
    return path


def test_av1_random_access_matches_sequential_and_cache_reader(av1_video):
    path = av1_video
    sequential = decode_torchcodec_frames(str(path), range(8))
    assert all(frame.shape == (3, 48, 64) and frame.dtype == torch.float32 for frame in sequential)
    assert [round(frame[0].mean().item()*255) for frame in sequential] == pytest.approx(list(range(0, 240, 30)), abs=3)
    selected = decode_torchcodec_frames(str(path), [7, 0, 3, 3])
    for frame, index in zip(selected, [7, 0, 3, 3]):
        torch.testing.assert_close(frame, sequential[index], rtol=0, atol=0)
    reader = PredictiveWorldRowReader.__new__(PredictiveWorldRowReader)
    reader.video_decoder = "torchcodec"
    current = reader._decode_frames(str(path), [3])[0]
    cache = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/data/cache_predictive_world_features.py"))
    future = cache['_decode_frame'](str(path), 3, 'torchcodec')
    torch.testing.assert_close(current, future, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='out of range'):
        decode_torchcodec_frames(str(path), [8])
    with pytest.raises(RuntimeError, match='out of range'):
        decode_torchcodec_frames(str(path), [-1])


def test_spawned_decode_matches_exact_frames_and_propagates_errors(av1_video):
    keys = [(str(av1_video), index) for index in [7, 0, 3, 3]]
    expected = decode_torchcodec_frames(str(av1_video), [7, 0, 3, 3])
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_decode_worker) as pool:
        futures = [pool.submit(decode_frame_job, keys, 'torchcodec') for _ in range(3)]
        for future in futures:
            frames, elapsed = future.result(timeout=120)
            assert elapsed >= 0
            for key, reference in zip(keys, expected):
                assert frames[key].dtype == np.uint8
                torch.testing.assert_close(torch.from_numpy(frames[key]).float().div_(255),
                                           reference, rtol=0, atol=0)
        bad = pool.submit(decode_frame_job, [(str(av1_video), 8)], 'torchcodec')
        with pytest.raises(RuntimeError, match='out of range'):
            bad.result(timeout=120)


def test_three_camera_decoder_handles_are_reused(av1_video, tmp_path):
    from psi.data.predictive_world import video
    video._DECODERS.clear()
    paths = [av1_video]
    for name in ['left.mp4', 'right.mp4']:
        path = tmp_path / name
        shutil.copyfile(av1_video, path)
        paths.append(path)
    for path in paths:
        decode_torchcodec_frames(str(path), [0])
    handles = dict(video._DECODERS)
    for path in paths:
        decode_torchcodec_frames(str(path), [1])
        assert video._DECODERS[str(path)] is handles[str(path)]
    assert len(video._DECODERS) == 3
    decode_torchcodec_frames(str(paths[-1]), [2], cache_capacity=1)
    assert len(video._DECODERS) == 1


@pytest.mark.parametrize('workers,shared', [(0, False), (2, False), (2, True)])
def test_cache_pipeline_resumes_committed_slots_with_real_decoding(av1_video, tmp_path, monkeypatch, workers, shared):
    """Exercise real spawn/AV1/dedup/NPY journal/publication; substitute only teacher/index."""
    import transformers
    from psi.data.predictive_world import provenance
    from psi.data.predictive_world.cache import verify_predictive_world_feature_manifest

    entry = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/data/cache_predictive_world_features.py'))
    scope = entry['main'].__globals__
    rows = [dict(future_visual_indices=indices, future_visual_valid=valid,
                 materialization={'video_paths': {'front': str(av1_video)}},
                 source_capability={'future_visual_keys': ['front']})
            for indices, valid in [([0, 1, 2], [True]*3), ([1, 2, 3], [True]*3),
                                   ([4, 5, 6], [True]*3), ([5, 6, 7], [True, True, False])]]
    monkeypatch.setitem(scope, '_index_binding', lambda *_a, **_kw: ({'test': 'binding'}, 4, 3))
    monkeypatch.setitem(scope, '_initialize_distributed', lambda _: (0, 1, torch.device('cpu')))
    monkeypatch.setattr(provenance, 'verify_qwen_provenance', lambda *_: None)
    monkeypatch.setattr(transformers.Qwen3VLForConditionalGeneration, 'from_pretrained',
                        lambda *_a, **_kw: torch.nn.Identity())
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda *_a, **_kw: object())
    encoded = []

    def extract(_model, _processor, images, **kwargs):
        values = images[:, 0].mean(dim=(1, 2)).to(torch.float16)
        encoded.extend(values.tolist())
        return values[:, None, None].expand(-1, 16, 2048).contiguous()

    monkeypatch.setitem(scope, 'extract_endpoint_vlm_features', extract)
    spec = tmp_path / 'provenance.json'
    spec.write_text(json.dumps(dict(encoder_id='fixture', processor_id='fixture',
        feature_layer='post_merger', dtype='float16', image_preprocessing='aspect_v1', video_decoder='torchcodec')))
    output = tmp_path / f'cache-{workers}'
    monkeypatch.setattr(sys, 'argv', ['cache', '--index', str(tmp_path / 'index'),
        '--provenance', str(spec), '--model', 'fixture', '--output', str(output),
        '--batch-size', '2', '--checkpoint-rows', '2', '--decode-workers', str(workers),
        '--decode-batch-size', '1', '--decode-prefetch-batches', '3'])
    if shared:
        # Keep the AF_UNIX path below the OS length limit; service exits when idle.
        directory = tempfile.mkdtemp(prefix='ovla-cache-test-')
        monkeypatch.setattr(sys, 'argv', [*sys.argv, '--decode-pool', directory + '/pool.sock'])

    def interrupted(*_):
        yield from enumerate(rows[:2])
        raise RuntimeError('simulated interruption after a committed checkpoint')

    monkeypatch.setitem(scope, '_iter_rows', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        entry['main']()
    array_path, progress_path, _ = entry['_rank_paths'](output, 0, 1)
    before = np.load(array_path)[:2].copy()
    assert np.load(progress_path)[:2].all()
    assert not np.load(progress_path)[2:].any()
    assert len(encoded) == 4  # duplicate frame requests were coalesced
    assert not (output / 'feature_manifest.json').exists()

    encoded.clear()
    monkeypatch.setitem(scope, '_iter_rows', lambda *_: enumerate(rows))
    entry['main']()
    actual = np.load(array_path)
    np.testing.assert_array_equal(actual[:2], before)
    assert len(encoded) == 3  # only the missing suffix was encoded
    assert np.load(progress_path).all()
    references = decode_torchcodec_frames(str(av1_video), range(8))
    for row_index, row in enumerate(rows):
        for group, (index, valid) in enumerate(zip(row['future_visual_indices'], row['future_visual_valid'])):
            expected = references[index][0].mean().to(torch.float16).item() if valid else 0
            np.testing.assert_array_equal(actual[row_index, group], np.full((16, 2048), expected, np.float16))
    verify_predictive_world_feature_manifest(output / 'feature_manifest.json', verify_shard_hashes=True)


def test_shared_pool_reassigns_work_after_client_exit_and_rejects_budget_mismatch(av1_video, monkeypatch):
    from psi.data.predictive_world import decode_pool
    from psi.data.predictive_world.decode_pool import SharedDecodeClient
    with tempfile.TemporaryDirectory(prefix='ovla-pool-test-') as directory:
        address = directory + '/pool.sock'
        with ThreadPoolExecutor(max_workers=2) as starters:
            connections = [starters.submit(SharedDecodeClient, address, workers=2, max_pending=8,
                                          idle_seconds=1) for _ in range(2)]
            first, second = [future.result(timeout=120) for future in connections]
        try:
            assert first.server_pid == second.server_pid
            with pytest.raises(ValueError, match='2 total workers'):
                SharedDecodeClient(address, workers=3, max_pending=8)
            with monkeypatch.context() as context:
                context.setattr(decode_pool, '_decoder_identity', lambda: 'different-runtime')
                with pytest.raises(ValueError, match='source/environment differs'):
                    SharedDecodeClient(address, workers=2, max_pending=8)
            keys = [(str(av1_video), index) for index in range(8)]
            requests = [client.submit(decode_frame_job, keys, 'torchcodec')
                        for _ in range(4) for client in (first, second)]
            for future in requests:
                frames, _ = future.result(timeout=120)
                assert len(frames) == 8
            first.shutdown()
            # The remaining client can keep the entire shared pool busy, with
            # no reallocation, replacement service or per-task worker ceiling.
            remaining = [second.submit(decode_frame_job, keys, 'torchcodec') for _ in range(8)]
            for future in remaining:
                frames, _ = future.result(timeout=120)
                assert len(frames) == 8
            bad = second.submit(decode_frame_job, [(str(av1_video), 8)], 'torchcodec')
            with pytest.raises(RuntimeError, match='out of range'):
                bad.result(timeout=120)
            # A failed decode belongs to its request and does not kill the pool.
            assert len(second.submit(decode_frame_job, keys, 'torchcodec').result(timeout=120)[0]) == 8
        finally:
            first.shutdown(cancel_futures=True)
            second.shutdown(cancel_futures=True)
        deadline = time.monotonic() + 10
        while Path(address).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not Path(address).exists(), 'idle service must exit and release its socket'


def test_feature_cache_batches_distinct_frames_and_keeps_a_bounded_recent_lru():
    cache = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/data/cache_predictive_world_features.py"))
    calls = []

    def fake_decode(path, indices, backend):
        calls.append((path, list(indices), backend))
        return [torch.full((3, 2, 2), index, dtype=torch.float32) for index in indices]

    decode_batch = cache['_decode_frame_batch']
    original_decode = decode_batch.__globals__['_decode_frames']
    decode_batch.__globals__['_decode_frames'] = fake_decode
    try:
        decoded = decode_batch(
            [('video-a.mp4', 4), ('video-a.mp4', 9), ('video-b.mp4', 2)], backend='torchcodec'
        )
    finally:
        decode_batch.__globals__['_decode_frames'] = original_decode
    assert calls == [
        ('video-a.mp4', [4, 9], 'torchcodec'),
        ('video-b.mp4', [2], 'torchcodec'),
    ]
    assert decoded[('video-a.mp4', 9)][0, 0, 0].item() == 9

    recent = cache['_RecentFrameFeatures'](2)
    first = np.full((16, 2048), 1, dtype=np.float16)
    recent.put(('video-a.mp4', 1), first)
    recent.put(('video-a.mp4', 2), np.full_like(first, 2))
    assert recent.get(('video-a.mp4', 1))[0, 0] == 1  # marks frame 1 most-recent
    recent.put(('video-a.mp4', 3), np.full_like(first, 3))
    assert recent.get(('video-a.mp4', 2)) is None
    assert recent.get(('video-a.mp4', 1))[0, 0] == 1
