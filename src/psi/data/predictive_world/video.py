"""Exact CPU frame decoding shared by current observations and target caches."""
from __future__ import annotations

from collections import OrderedDict
import os
from time import perf_counter

import numpy as np
import torch

# Lazy, process-local handles; never carry decoder threads across DataLoader forks.
_DECODERS: OrderedDict = OrderedDict()
_PID: int | None = None


def _torchcodec_uint8(path: str, indices, *, cache_capacity: int = 8) -> torch.Tensor:
    global _PID
    if cache_capacity <= 0:
        raise ValueError("video decoder cache capacity must be positive")
    if _PID != os.getpid():
        _DECODERS.clear()
        _PID = os.getpid()
    from torchcodec.decoders import VideoDecoder

    if path in _DECODERS:
        decoder = _DECODERS.pop(path)
    else:
        decoder = VideoDecoder(path, device="cpu", num_ffmpeg_threads=1, seek_mode="exact")
    _DECODERS[path] = decoder
    while len(_DECODERS) > cache_capacity:
        _DECODERS.popitem(last=False)
    positions = [int(index) for index in indices]
    if any(index < 0 or index >= len(decoder) for index in positions):
        raise RuntimeError(f"predictive-world video frame is out of range: {path}:{positions}")
    if not positions:
        return torch.empty((0, 3, 0, 0), dtype=torch.uint8)
    return decoder.get_frames_at(indices=positions).data


def decode_torchcodec_frames(path: str, indices, *, cache_capacity: int = 8) -> list[torch.Tensor]:
    frames = _torchcodec_uint8(path, indices, cache_capacity=cache_capacity)
    return list(frames.float().div_(255).unbind(0))


def initialize_decode_worker() -> None:
    """Each spawned process owns its decoders and one CPU compute thread."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def decode_frame_job(keys, backend: str, cache_capacity: int = 8):
    """Return exact CHW uint8 frames through IPC, avoiding float32/shared-memory copies.

    Float conversion remains at the consumer boundary, identical to the synchronous
    decoder. Grouping by video preserves batched exact frame seeking in each worker.
    """
    started = perf_counter()
    by_path: dict[str, list[int]] = {}
    for path, index in keys:
        by_path.setdefault(path, []).append(index)
    decoded = {}
    for path, indices in by_path.items():
        if backend == "torchcodec":
            frames = _torchcodec_uint8(path, indices, cache_capacity=cache_capacity)
            decoded.update(((path, index), frame.numpy()) for index, frame in zip(indices, frames))
        elif backend == "opencv":
            import cv2

            cv2.setNumThreads(1)
            capture = cv2.VideoCapture(path)
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"cannot open predictive-world video: {path}")
                count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
                for index in indices:
                    source = count - 1 if count > 0 and index == count else index
                    if source < 0 or (count > 0 and source >= count):
                        raise RuntimeError(f"predictive-world frame is out of range {path}:{index}")
                    capture.set(cv2.CAP_PROP_POS_FRAMES, source)
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"cannot decode predictive-world frame {path}:{index}")
                    decoded[path, index] = np.ascontiguousarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    )
            finally:
                capture.release()
        else:
            raise ValueError(f"unsupported video decoder: {backend}")
    return decoded, perf_counter() - started
